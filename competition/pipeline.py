"""Full-scale pipeline (Submission 2 = the validated E-M1 design, F0.5 0.942 on validation).

Stages (each streams to WORK_DIR and skips work already on disk, so a crash costs one step):
  block    : label-free state partitions -> TF-IDF top-5 per record within partition, plus
             country-wide exact keys  -> work/pipe_{split}/tf/*.parquet, keys.parquet
  features : 34 pair features, in 32 buckets of whole records -> work/pipe_{split}/feat/*.parquet
  train    : one LightGBM on all validation-sample features (work/e_m1_feats_k5) -> model file
  decide   : M1 (+ --m2 stage 2) -> exclusive + expected-F0.5 decoding -> TSVs in output/sub2*/

Usage: python -m competition.pipeline --split test --stage block|features|train|decide|all
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from resolver.block import key_candidates, tfidf_candidates
from competition.config import OUTPUT_DIR, SEED, WORK_DIR
from resolver.decode import ef_decode, exclusive
from resolver.features import FEATURES, pair_features
from resolver.adaptive import GEN, generic_features, generic_tokens
from competition.address import CA, canon_addr, canon_features
from competition.vocab import VOC, token_df, vocab_features
from competition.groups import GRP, group_features
from competition.legal import LF, legal_features, legal_forms
from resolver.features2 import NEW, competition_features, diff_features, mask_keyonly, s1_maxima
from resolver.partition import infer_state, learn_aliases, s1_state

K_NAME, K_COMB, K_ADDR, K_KEEP = 10, 20, 10, 5
N_BUCKETS = 32
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=63, min_data_in_leaf=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=SEED, verbose=-1, num_threads=8)
ROUNDS = 400
BLOCK_COLS = ["entity_id", "country", "name_tok", "addr", "parts", "hn", "name_key", "name_ns"]
FEAT_COLS = ["entity_id", "name_tok", "addr", "nums", "hn", "name_ns", "name_key", "nonlatin", "empty_addr"]
MODEL = WORK_DIR / "model_m1.txt"
MODEL_M2 = WORK_DIR / "model_m2.txt"
PRUNE = 0.01  # stage-2 input: pairs with p1 >= PRUNE
# v1 = 34 pair features (E-M1); v2 = + 14 competition/difference features (E-F2: M1 0.9429 -> 0.9590)
V1 = dict(feat="feat", m1=MODEL, m1_cols=FEATURES, ens="ens", tag="sub2", train=WORK_DIR / "e_m1_feats_k5" / "*.parquet")
V2 = dict(feat="feat2", m1=WORK_DIR / "model_m1b.txt", m1_cols=FEATURES + NEW, ens="ens2", tag="sub3",
          train=WORK_DIR / "e_f2b_feats.parquet")  # E-F2b: record-side features blanked on key-only pairs (0.9585)
# v3 = v2 + optional no-state name search (E-B5) + optional transliteration features (E-T1); set from results
V3_NOSTATE, V3_TRANSLIT, V3_GENERIC = False, False, True  # E-B5 rejected (oracle +0.0008 only); E-T1 rejected (0.9577 < 0.9585); E-T2 kept (0.9594)
TR = ["tr_name_tset", "tr_name_ratio", "tr_name_jw", "tr_n_tok_extra"]
_V3_SRC = ("e_b5" if V3_NOSTATE else "e_f2b") + ("_tr" if V3_TRANSLIT else "") + ("_gen" if V3_GENERIC else "")
V3 = dict(feat="feat3", m1=WORK_DIR / "model_m1c.txt",
          m1_cols=FEATURES + NEW + (TR if V3_TRANSLIT else []) + (GEN if V3_GENERIC else []),
          ens="ens3" if V3_NOSTATE else "ens3g", tag="sub4", train=WORK_DIR / f"{_V3_SRC}_feats.parquet")
V3_CANON = True  # E-A1 kept at stage 2 (ens3gca mean3 0.9639 vs 0.9623, both folds up); canonical addresses: switches --v3 to feat4 / model_m1d / ens3gca / sub5 once validated
if V3_CANON:
    V3 = dict(V3, feat="feat4", m1=WORK_DIR / "model_m1d.txt", m1_cols=V3["m1_cols"] + CA, ens=V3["ens"] + "ca", tag="sub5",
              train=WORK_DIR / f"{_V3_SRC}_ca_feats.parquet")
V3_VOCAB = False  # E-T3 vocabulary class of unmatched name tokens: feat5 / model_m1e / ens3gcav / sub9 once validated
if V3_VOCAB:
    V3 = dict(V3, feat="feat5", m1=WORK_DIR / "model_m1e.txt", m1_cols=V3["m1_cols"] + VOC, ens=V3["ens"] + "v", tag="sub9",
              train=WORK_DIR / f"{_V3_SRC}_ca_voc_feats.parquet")
V3_LEGAL = True  # E-T4 kept: M1 0.9655 vs 0.9606 (0.9604 / 0.9689, both folds; dummy-column control = 0.9606); legal-form agreement from raw names: feat6 (= feat4 + LF) / model_m1f / ens3gcal / sub10 once validated
if V3_LEGAL:
    V3 = dict(V3, feat="feat6", m1=WORK_DIR / "model_m1f.txt", m1_cols=V3["m1_cols"] + LF, ens=V3["ens"] + "l", tag="sub10",
              train=WORK_DIR / f"{_V3_SRC}_ca_lf_feats.parquet")
# E-P2: stage-2 feature set. Twin / S1-side count features collapse when decoys double (x2 world 0.86)
V3_STAGE2 = "no_twins_grp"  # E-G2 kept (0.9620, x2 0.9534, both folds up). E-P2: as-is 0.9606, x2 0.9505 (all: 0.9628 / 0.8609); "all" | "no_twins" | "no_s1_side"  (ens dir gets the suffix; see experiments/e_p2_density.py)
if V3_STAGE2 != "all":
    from experiments.e_p2_density import SETS
    V3 = dict(V3, ens=V3["ens"] + "_" + V3_STAGE2, coll=SETS[V3_STAGE2], tag=V3["tag"] + V3_STAGE2.replace("_", ""))


def _dir(split: str, name: str):
    d = WORK_DIR / f"pipe_{split}" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _norm(split: str, src: int, cols: list[str]) -> pl.DataFrame:
    return pl.read_parquet(WORK_DIR / f"norm/{split}_s{src}.parquet", columns=cols)


def _norm_country(split: str, src: int, cols: list[str], country: str) -> pl.DataFrame:
    return pl.scan_parquet(WORK_DIR / f"norm/{split}_s{src}.parquet").select(cols) \
             .filter(pl.col("country") == country).collect()


def block(split: str) -> None:
    """One country at a time: aliases, keys and partitions never cross countries, and this
    keeps peak memory to the largest country (8 GB machine)."""
    t0 = time.time()
    countries = pl.read_parquet(WORK_DIR / f"norm/{split}_s1.parquet", columns=["country"])["country"].unique().sort()
    out = _dir(split, "tf")
    for country in countries:
        s1 = _norm_country(split, 1, BLOCK_COLS, country)
        recs = pl.concat([_norm_country(split, s, BLOCK_COLS, country) for s in (2, 3)])
        keys_path = _dir(split, "keys") / f"keys_{country}.parquet"
        if not keys_path.exists():
            key_candidates(s1, recs).write_parquet(keys_path)
        alias = learn_aliases(s1, recs)  # label-free, fitted on this split's own inputs
        s1 = s1.with_columns(s1_state(s1).alias("state")).drop("parts")
        recs = recs.with_columns(infer_state(recs, alias).alias("state")).drop("parts")
        parts = s1.group_by("state").len().drop_nulls().sort("len", descending=True).rows()
        for i, (state, n) in enumerate(parts):
            path = out / f"{country}_{i:05d}.parquet"
            if path.exists():
                continue
            g = s1.filter(pl.col("state") == state)
            q = recs.filter(pl.col("state") == state)
            c = tfidf_candidates(g, q, K_NAME, K_COMB, K_ADDR)
            c = c.with_columns(pl.col("cos_comb").rank("ordinal", descending=True).over("r").alias("rank_comb")) \
                 .filter(pl.col("rank_comb") <= K_KEEP)
            c.with_columns(pl.Series("s1", g["entity_id"].to_numpy()[c["s"].to_numpy()], dtype=pl.String),
                           pl.Series("rec", q["entity_id"].to_numpy()[c["r"].to_numpy()], dtype=pl.String)) \
             .select("s1", "rec", "cos_name", "cos_addr", "cos_comb").write_parquet(path)
            if n > 20_000:
                print(f"  {country} [{i + 1}/{len(parts)}] {state}: {n:,} S1 x {q.height:,} recs "
                      f"({time.time() - t0:.0f}s)", flush=True)
        del s1, recs
        print(f"{country} blocked ({time.time() - t0:.0f}s)", flush=True)
    print(f"block done in {time.time() - t0:.0f}s", flush=True)


def features(split: str) -> None:
    t0 = time.time()
    out = _dir(split, "feat")
    s1 = _norm(split, 1, FEAT_COLS)
    recs_lazy = pl.concat([pl.scan_parquet(WORK_DIR / f"norm/{split}_s{s}.parquet").select(FEAT_COLS) for s in (2, 3)])
    tf = pl.scan_parquet(_dir(split, "tf") / "*.parquet").with_columns(pl.lit(False).alias("in_keys"))
    keys = pl.scan_parquet(_dir(split, "keys") / "*.parquet")
    for b in range(N_BUCKETS):
        path = out / f"b{b:02d}.parquet"
        if path.exists():
            continue
        in_b = (pl.col("rec").hash(SEED) % N_BUCKETS) == b
        tb = tf.filter(in_b).collect()
        kb = keys.filter(in_b).collect().join(tb.select("s1", "rec"), on=["s1", "rec"], how="anti") \
                 .with_columns(pl.lit(None, pl.Float32).alias(c) for c in ("cos_name", "cos_addr", "cos_comb")) \
                 .with_columns(pl.lit(True).alias("in_keys")).select(tb.columns)
        cands = pl.concat([tb, kb], how="vertical_relaxed")
        recs_b = recs_lazy.filter((pl.col("entity_id").hash(SEED) % N_BUCKETS) == b).collect()
        f = pair_features(cands, s1, recs_b)
        f.select("s1", "rec", *[pl.col(x).cast(pl.Float32) for x in FEATURES]).write_parquet(path)
        print(f"  bucket {b + 1}/{N_BUCKETS}: {f.height:,} pairs ({time.time() - t0:.0f}s)", flush=True)
    print(f"features done in {time.time() - t0:.0f}s", flush=True)


def features2(split: str) -> None:
    """Add the E-F2 features to every v1 bucket (no re-blocking). S1-side maxima come from one
    streaming pass over all buckets; record-side features are exact per bucket (whole records)."""
    t0 = time.time()
    src, out = _dir(split, "feat"), _dir(split, "feat2")
    best = s1_maxima(pl.scan_parquet(src / "*.parquet")).collect(engine="streaming")
    s1 = _norm(split, 1, ["entity_id", "name_tok", "hn"])
    recs = pl.concat([pl.scan_parquet(WORK_DIR / f"norm/{split}_s{s}.parquet").select("entity_id", "name_tok", "hn") for s in (2, 3)])
    for b in range(N_BUCKETS):
        path = out / f"b{b:02d}.parquet"
        if path.exists():
            continue
        f = pl.read_parquet(src / f"b{b:02d}.parquet")
        comp = competition_features(f.select("s1", "rec", "cos_name", "cos_addr", "cos_comb"), best)
        recs_b = recs.filter((pl.col("entity_id").hash(SEED) % N_BUCKETS) == b).collect()
        diff = diff_features(f.select("s1", "rec"), s1, recs_b)
        f.join(comp, on=["s1", "rec"], how="left").join(diff, on=["s1", "rec"], how="left").write_parquet(path)
    print(f"features2 done in {time.time() - t0:.0f}s", flush=True)


def nostate(split: str) -> None:
    """E-B5 on this split: name-only top-5 S1 in the whole country for records block() could not
    place in a state (same per-country aliases as block())."""
    from experiments.e_b5_nostate import nostate_candidates
    t0 = time.time()
    path = _dir(split, "nostate") / "pairs.parquet"
    cols = ["entity_id", "country", "name_tok", "addr", "parts"]
    out = []
    for country in pl.read_parquet(WORK_DIR / f"norm/{split}_s1.parquet", columns=["country"])["country"].unique().sort():
        s1 = _norm_country(split, 1, cols, country)
        recs = pl.concat([_norm_country(split, x, cols, country) for x in (2, 3)])
        out.append(nostate_candidates(s1.drop("parts"), recs, learn_aliases(s1, recs)))
    pl.concat(out).write_parquet(path)
    print(f"nostate done in {time.time() - t0:.0f}s -> {path}", flush=True)


def features3(split: str) -> None:
    """feat3 = v1 features (+ no-state pairs, affected records re-featurised) + E-F2 features
    (+ transliteration features), key-only masked. v1/v2 buckets are left untouched."""
    import json
    from experiments.e_t1_translit import raw_names, tr_features
    t0 = time.time()
    src, ext, out = _dir(split, "feat"), _dir(split, "feat_ext"), _dir(split, "feat3")
    s1_all = _norm(split, 1, FEAT_COLS)
    recs_lazy = pl.concat([pl.scan_parquet(WORK_DIR / f"norm/{split}_s{x}.parquet").select(FEAT_COLS) for x in (2, 3)])
    new = pl.read_parquet(_dir(split, "nostate") / "pairs.parquet") if V3_NOSTATE else None
    for b in range(N_BUCKETS):                       # pass 1: candidate union + pair features
        path = ext / f"b{b:02d}.parquet"
        if path.exists():
            continue
        f = pl.read_parquet(src / f"b{b:02d}.parquet")
        if new is not None:
            nb = new.filter((pl.col("rec").hash(SEED) % N_BUCKETS) == b)
            aff = nb.select("rec").unique()
            old = f.join(aff, on="rec").select("s1", "rec", "cos_name", "cos_addr", "cos_comb", (pl.col("in_keys") > 0.5).alias("in_keys"))
            nb = nb.join(old.select("s1", "rec"), on=["s1", "rec"], how="anti").with_columns(pl.lit(False).alias("in_keys"))
            cands = pl.concat([old, nb.select(old.columns)], how="vertical_relaxed")
            recs_b = recs_lazy.join(aff.lazy().rename({"rec": "entity_id"}), on="entity_id").collect()
            g = pair_features(cands, s1_all, recs_b).select("s1", "rec", *[pl.col(x).cast(pl.Float32) for x in FEATURES])
            f = pl.concat([f.join(aff, on="rec", how="anti"), g.select(f.columns)], how="vertical_relaxed")
        f.write_parquet(path)
    best = s1_maxima(pl.scan_parquet(ext / "*.parquet")).collect(engine="streaming")
    s1n = _norm(split, 1, ["entity_id", "country", "name_tok", "hn"])
    gen = generic_tokens(s1n) if V3_GENERIC else None   # adaptive: this split's own S1 vocabulary, per country
    recs_h = pl.concat([pl.scan_parquet(WORK_DIR / f"norm/{split}_s{x}.parquet").select("entity_id", "name_tok", "hn", "nonlatin") for x in (2, 3)])
    if V3_TRANSLIT:
        mapping = json.loads((WORK_DIR / "translit_map_all.json").read_text())
        nl_ids = recs_h.filter(pl.col("nonlatin")).select("entity_id").collect()
        raw = pl.concat([raw_names(split, x, nl_ids) for x in (2, 3)])
    for b in range(N_BUCKETS):                       # pass 2: E-F2 (+ E-T1) features, mask
        path = out / f"b{b:02d}.parquet"
        if path.exists():
            continue
        f = pl.read_parquet(ext / f"b{b:02d}.parquet")
        rb = recs_h.filter((pl.col("entity_id").hash(SEED) % N_BUCKETS) == b).collect()
        f = f.join(competition_features(f.select("s1", "rec", "cos_name", "cos_addr", "cos_comb"), best), on=["s1", "rec"], how="left") \
             .join(diff_features(f.select("s1", "rec"), s1n, rb), on=["s1", "rec"], how="left")
        if V3_TRANSLIT:
            recs_tr = rb.select("entity_id", "nonlatin").join(raw, on="entity_id", how="left")
            f = f.join(tr_features(f.select("s1", "rec", "name_tset", "name_ratio", "name_jw", "n_tok_extra"), s1n, recs_tr, mapping),
                       on=["s1", "rec"], how="left")
        if V3_GENERIC:
            f = f.join(generic_features(f.select("s1", "rec"), s1n, rb, gen), on=["s1", "rec"], how="left")
        mask_keyonly(f).write_parquet(path)
    print(f"features3 done in {time.time() - t0:.0f}s", flush=True)


def features4(split: str) -> None:
    """feat4 = feat3 + canonical-address similarity (E-A1). S1 canon once per country, records per bucket."""
    t0 = time.time()
    src, out = _dir(split, "feat3"), _dir(split, "feat4")
    cols = ["entity_id", "country", "parts"]
    c1 = pl.concat([canon_addr(_norm_country(split, 1, cols, c))
                    for c in pl.read_parquet(WORK_DIR / f"norm/{split}_s1.parquet", columns=["country"])["country"].unique()])
    recs = pl.concat([pl.scan_parquet(WORK_DIR / f"norm/{split}_s{x}.parquet").select(cols) for x in (2, 3)])
    for b in range(N_BUCKETS):
        path = out / f"b{b:02d}.parquet"
        if path.exists():
            continue
        f = pl.read_parquet(src / f"b{b:02d}.parquet")
        c2 = canon_addr(recs.filter((pl.col("entity_id").hash(SEED) % N_BUCKETS) == b).collect())
        f.join(canon_features(f.select("s1", "rec"), c1, c2), on=["s1", "rec"], how="left").write_parquet(path)
    print(f"features4 done in {time.time() - t0:.0f}s", flush=True)


def features5(split: str) -> None:
    """feat5 = feat4 + vocabulary class of unmatched name tokens (E-T3), DF fitted on this split's S1."""
    t0 = time.time()
    src, out = _dir(split, "feat4"), _dir(split, "feat5")
    s1 = _norm(split, 1, ["entity_id", "country", "name_tok"])
    df, gen = token_df(s1), generic_tokens(s1)
    recs = pl.concat([pl.scan_parquet(WORK_DIR / f"norm/{split}_s{x}.parquet").select("entity_id", "name_tok") for x in (2, 3)])
    for b in range(N_BUCKETS):
        path = out / f"b{b:02d}.parquet"
        if path.exists():
            continue
        f = pl.read_parquet(src / f"b{b:02d}.parquet")
        rb = recs.filter((pl.col("entity_id").hash(SEED) % N_BUCKETS) == b).collect()
        f.join(vocab_features(f.select("s1", "rec"), s1, rb, df, gen), on=["s1", "rec"], how="left").write_parquet(path)
    print(f"features5 done in {time.time() - t0:.0f}s", flush=True)


def features6(split: str) -> None:
    """feat6 = feat4 + legal-form agreement (E-T4) from the raw names of this split."""
    from competition.config import DATA_DIR
    t0 = time.time()
    src, out = _dir(split, "feat4"), _dir(split, "feat6")
    raw = lambda x: pl.read_csv(DATA_DIR / split / f"{split}_source{x}.tsv", separator="\t", quote_char=None, columns=["entity_id", "business_name"])
    lf1 = legal_forms(raw(1))
    lf2 = legal_forms(pl.concat([raw(2), raw(3)]))
    for b in range(N_BUCKETS):
        path = out / f"b{b:02d}.parquet"
        if path.exists():
            continue
        f = pl.read_parquet(src / f"b{b:02d}.parquet")
        f.join(legal_features(f.select("s1", "rec"), lf1, lf2), on=["s1", "rec"], how="left").write_parquet(path)
    print(f"features6 done in {time.time() - t0:.0f}s", flush=True)


def train(v2: bool = False, v3: bool = False) -> None:
    cfg = V3 if v3 else V2 if v2 else V1
    feats = pl.read_parquet(cfg["train"], columns=cfg["m1_cols"] + ["y"])
    model = lgb.train(PARAMS, lgb.Dataset(feats.select(cfg["m1_cols"]).to_numpy(), feats["y"].to_numpy()), ROUNDS)
    model.save_model(str(cfg["m1"]))
    print(f"trained on {feats.height:,} pairs -> {cfg['m1']}", flush=True)


def _write(split: str, matches: pl.DataFrame, out_dir) -> None:
    """Both TSVs in test_source1 order into out_dir (never the bare output/, which holds Submission 1)."""
    cands = pl.scan_parquet(_dir(split, "feat") / "*.parquet").select("s1", "rec")
    ids = pl.read_parquet(WORK_DIR / f"norm/{split}_s1.parquet", columns=["entity_id"]) \
            .rename({"entity_id": "source1_entity_id"})
    out_dir.mkdir(parents=True, exist_ok=True)
    shared = WORK_DIR / f"pipe_{split}" / "candidate_pairs.tsv"   # same blocking for every version: write once, hard-link
    for name, col, pairs in (("matching_results.tsv", "matched_entity_ids", matches.lazy()),
                             ("candidate_pairs.tsv", "candidate_entity_ids", cands)):
        if name == "candidate_pairs.tsv" and shared.exists():
            (out_dir / name).unlink(missing_ok=True)
            (out_dir / name).hardlink_to(shared)
            continue
        lists = pairs.group_by("s1").agg(pl.col("rec").unique().sort().str.join(",").alias(col)) \
                     .rename({"s1": "source1_entity_id"}).collect()
        ids.join(lists, on="source1_entity_id", how="left", maintain_order="left").fill_null("") \
           .write_csv(out_dir / name, separator="\t", quote_style="never")


def decide(split: str, m2: bool = False, ens: bool = False, v2: bool = False, v3: bool = False) -> None:
    """M1 (+ optional stage 2) -> exclusive assignment -> expected-F0.5 decoding (src/decode.py).
    Stage 2 = M2 LightGBM, or with ens=True the E-E1 average of LightGBM + ExtraTrees + MLP.
    Validation (out-of-fold): M1 0.9429, M2 0.9488, ensemble 0.9510."""
    cfg = V3 if v3 else V2 if v2 else V1
    v2 = v2 or v3
    if v2 and m2 and not ens:
        raise ValueError("v2 has no single-M2 model; use --ens")
    m2 = m2 or ens
    t0 = time.time()
    m1 = lgb.Booster(model_file=str(cfg["m1"]))
    ex, pruned = [], []
    s2_dir = _dir(split, "stage2" + ("_v3" if v3 else "_v2" if v2 else ""))
    for path in sorted(_dir(split, cfg["feat"]).glob("*.parquet")):
        f = pl.read_parquet(path)
        if v2:
            f = mask_keyonly(f)   # idempotent (feat3 is already masked)
        f = f.with_columns(pl.Series("p", m1.predict(f.select(cfg["m1_cols"]).to_numpy())))
        if m2:  # stage 2 only needs pairs with p1 >= PRUNE (validated in E-M2b)
            f = f.filter(pl.col("p") >= PRUNE)
            f.write_parquet(s2_dir / path.name)
            pruned.append(f.select("s1", "rec", "p", "name_tset", "addr_tset"))
        else:  # buckets hold whole records, so the per-record argmax is exact inside a bucket
            ex.append(exclusive(f.select("s1", "rec", "p")))
    if m2:
        from resolver.collective import components
        from resolver.collective import COLLECTIVE, collective
        recs = pl.concat([_norm(split, s, ["entity_id", "country", "name_key", "name_ns", "hn", "addr"]) for s in (2, 3)])
        comp = components(recs)
        del recs
        pruned = pl.concat(pruned)
        col = collective(pruned.select("s1", "rec", "p"), comp).with_columns(pl.col(c).cast(pl.Float32) for c in COLLECTIVE)
        del comp
        if any(c in GRP for c in cfg.get("coll", [])):   # E-G2 entity-group gains, exact per S1 -> chunk by S1
            recs_g = pl.concat([_norm(split, s, ["entity_id", "name_tok", "addr", "hn"]) for s in (2, 3)]) \
                       .join(pruned.select(pl.col("rec").alias("entity_id")).unique(), on="entity_id")
            grp = pl.concat([group_features(pruned.filter(pl.col("s1").hash(SEED) % 8 == k).rename({"p": "p1"}), recs_g) for k in range(8)])
            col = col.join(grp, on=["s1", "rec"], how="left").with_columns(pl.col(c).fill_null(0) for c in GRP)
            del recs_g, grp
        del pruned
        if ens:
            import joblib
            ens_dir = WORK_DIR / cfg["ens"]
            weights = joblib.load(ens_dir / "final_weights.joblib")
            models = {n: joblib.load(ens_dir / f"final_{n}.joblib") for n in weights}
        else:
            model2 = lgb.Booster(model_file=str(MODEL_M2))
        for path in sorted(s2_dir.glob("*.parquet")):
            f = pl.read_parquet(path).drop("p").join(col, on=["s1", "rec"])
            X = f.select(cfg["m1_cols"] + cfg.get("coll", COLLECTIVE)).to_numpy().astype(np.float32)
            if ens:
                p2 = sum(w * models[n].predict_proba(X)[:, 1] for n, w in weights.items())
            else:
                p2 = model2.predict(X)
            ex.append(exclusive(f.select("s1", "rec").with_columns(pl.Series("p", p2))))
    matches = ef_decode(pl.concat(ex))
    out_dir = OUTPUT_DIR / (cfg["tag"] + ("d_ens_ef" if ens else "c_m2_ef" if m2 else "b_m1_ef"))
    _write(split, matches, out_dir)
    s1c = pl.read_parquet(WORK_DIR / f"norm/{split}_s1.parquet", columns=["entity_id", "country"])
    has = s1c.with_columns(pl.col("entity_id").is_in(matches["s1"].unique().implode()).alias("has_match"))
    print(has.group_by("country").agg(pl.col("has_match").mean()).sort("country"))
    print(f"decide done in {time.time() - t0:.0f}s: {matches.height:,} matched pairs -> {out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--stage", required=True, choices=["block", "features", "features2", "nostate", "features3", "features4", "features5", "features6", "train", "decide", "all"])
    ap.add_argument("--v3", action="store_true", help="use feat3 / model_m1c / ens3 (E-B5 + E-T1 per V3_* flags)")
    ap.add_argument("--v2", action="store_true", help="use the E-F2 feature set (feat2, model_m1b, ens2)")
    ap.add_argument("--m2", action="store_true", help="decide with the stage-2 (collective) model")
    ap.add_argument("--ens", action="store_true", help="decide with the stage-2 ensemble (E-E1)")
    a = ap.parse_args()
    stages = ["block", "features", "train", "decide"] if a.stage == "all" else [a.stage]
    for s in stages:
        {"block": lambda: block(a.split), "features": lambda: features(a.split),
         "features2": lambda: features2(a.split), "nostate": lambda: nostate(a.split),
         "features3": lambda: features3(a.split), "features4": lambda: features4(a.split), "features5": lambda: features5(a.split), "features6": lambda: features6(a.split), "train": lambda: train(a.v2, a.v3),
         "decide": lambda: decide(a.split, a.m2, a.ens, a.v2, a.v3)}[s]()


if __name__ == "__main__":
    main()
