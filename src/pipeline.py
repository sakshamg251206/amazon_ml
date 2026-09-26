"""Full-scale pipeline (Submission 2 = the validated E-M1 design, F0.5 0.942 on validation).

Stages (each streams to WORK_DIR and skips work already on disk, so a crash costs one step):
  block    : label-free state partitions -> TF-IDF top-5 per record within partition, plus
             country-wide exact keys  -> work/pipe_{split}/tf/*.parquet, keys.parquet
  features : 34 pair features, in 32 buckets of whole records -> work/pipe_{split}/feat/*.parquet
  train    : one LightGBM on all validation-sample features (work/e_m1_feats_k5) -> model file
  decide   : M1 (+ --m2 stage 2) -> exclusive + expected-F0.5 decoding -> TSVs in output/sub2*/

Usage: python -m src.pipeline --split test --stage block|features|train|decide|all
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from src.block import key_candidates, tfidf_candidates
from src.config import OUTPUT_DIR, SEED, WORK_DIR
from src.decode import ef_decode, exclusive
from src.features import FEATURES, pair_features
from src.partition import infer_state, learn_aliases, s1_state

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


def train() -> None:
    feats = pl.read_parquet(WORK_DIR / "e_m1_feats_k5" / "*.parquet", columns=FEATURES + ["y"])
    model = lgb.train(PARAMS, lgb.Dataset(feats.select(FEATURES).to_numpy(), feats["y"].to_numpy()), ROUNDS)
    model.save_model(str(MODEL))
    print(f"trained on {feats.height:,} pairs -> {MODEL}", flush=True)


def _write(split: str, matches: pl.DataFrame, out_dir) -> None:
    """Both TSVs in test_source1 order into out_dir (never the bare output/, which holds Submission 1)."""
    cands = pl.scan_parquet(_dir(split, "feat") / "*.parquet").select("s1", "rec")
    ids = pl.read_parquet(WORK_DIR / f"norm/{split}_s1.parquet", columns=["entity_id"]) \
            .rename({"entity_id": "source1_entity_id"})
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, col, pairs in (("matching_results.tsv", "matched_entity_ids", matches.lazy()),
                             ("candidate_pairs.tsv", "candidate_entity_ids", cands)):
        lists = pairs.group_by("s1").agg(pl.col("rec").unique().sort().str.join(",").alias(col)) \
                     .rename({"s1": "source1_entity_id"}).collect()
        ids.join(lists, on="source1_entity_id", how="left", maintain_order="left").fill_null("") \
           .write_csv(out_dir / name, separator="\t", quote_style="never")


def decide(split: str, m2: bool = False, ens: bool = False) -> None:
    """M1 (+ optional stage 2) -> exclusive assignment -> expected-F0.5 decoding (src/decode.py).
    Stage 2 = M2 LightGBM, or with ens=True the E-E1 average of LightGBM + ExtraTrees + MLP.
    Validation (out-of-fold): M1 0.9429, M2 0.9488, ensemble 0.9510."""
    m2 = m2 or ens
    t0 = time.time()
    m1 = lgb.Booster(model_file=str(MODEL))
    ex, pruned = [], []
    s2_dir = _dir(split, "stage2")
    for path in sorted(_dir(split, "feat").glob("*.parquet")):
        f = pl.read_parquet(path)
        f = f.with_columns(pl.Series("p", m1.predict(f.select(FEATURES).to_numpy())))
        if m2:  # stage 2 only needs pairs with p1 >= PRUNE (validated in E-M2b)
            f = f.filter(pl.col("p") >= PRUNE)
            f.write_parquet(s2_dir / path.name)
            pruned.append(f.select("s1", "rec", "p"))
        else:  # buckets hold whole records, so the per-record argmax is exact inside a bucket
            ex.append(exclusive(f.select("s1", "rec", "p")))
    if m2:
        from experiments.e_g1_twins import components
        from experiments.e_m2_collective import COLLECTIVE, collective
        recs = pl.concat([_norm(split, s, ["entity_id", "country", "name_key", "name_ns", "hn", "addr"]) for s in (2, 3)])
        comp = components(recs)
        del recs
        col = collective(pl.concat(pruned), comp).with_columns(pl.col(c).cast(pl.Float32) for c in COLLECTIVE)
        del comp, pruned
        if ens:
            import joblib
            from experiments.e_e1_ensemble import ENS
            weights = joblib.load(ENS / "final_weights.joblib")
            models = {n: joblib.load(ENS / f"final_{n}.joblib") for n in weights}
        else:
            model2 = lgb.Booster(model_file=str(MODEL_M2))
        for path in sorted(s2_dir.glob("*.parquet")):
            f = pl.read_parquet(path).drop("p").join(col, on=["s1", "rec"])
            X = f.select(FEATURES + COLLECTIVE).to_numpy().astype(np.float32)
            if ens:
                p2 = sum(w * models[n].predict_proba(X)[:, 1] for n, w in weights.items())
            else:
                p2 = model2.predict(X)
            ex.append(exclusive(f.select("s1", "rec").with_columns(pl.Series("p", p2))))
    matches = ef_decode(pl.concat(ex))
    out_dir = OUTPUT_DIR / ("sub2d_ens_ef" if ens else "sub2c_m2_ef" if m2 else "sub2b_m1_ef")
    _write(split, matches, out_dir)
    s1c = pl.read_parquet(WORK_DIR / f"norm/{split}_s1.parquet", columns=["entity_id", "country"])
    has = s1c.with_columns(pl.col("entity_id").is_in(matches["s1"].unique().implode()).alias("has_match"))
    print(has.group_by("country").agg(pl.col("has_match").mean()).sort("country"))
    print(f"decide done in {time.time() - t0:.0f}s: {matches.height:,} matched pairs -> {out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--stage", required=True, choices=["block", "features", "train", "decide", "all"])
    ap.add_argument("--m2", action="store_true", help="decide with the stage-2 (collective) model")
    ap.add_argument("--ens", action="store_true", help="decide with the stage-2 ensemble (E-E1)")
    a = ap.parse_args()
    stages = ["block", "features", "train", "decide"] if a.stage == "all" else [a.stage]
    for s in stages:
        {"block": lambda: block(a.split), "features": lambda: features(a.split),
         "train": train, "decide": lambda: decide(a.split, a.m2, a.ens)}[s]()


if __name__ == "__main__":
    main()
