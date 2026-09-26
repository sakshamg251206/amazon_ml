"""E-B5: name-only country-wide retrieval for records whose state cannot be inferred.

Finding (audit): 62.5% of the 44.8k true pairs missed by blocking have a record with no inferred
state (mostly empty addresses). Such records are never searched by partitioned TF-IDF; only exact
keys can reach them, and one typo breaks a key. ~3% of all records have no state.

Test: for each no-state record, top-K S1 of its country by name char-3-gram TF-IDF. Measure
(1) blocking recall on the validation sample, (2) out-of-fold M1 + expected-F F0.5 with the same
folds/params as E-M1 (baseline 0.9429), recomputing features for every affected record.

Usage: python -m experiments.e_b5_nostate [retrieve|eval|all]
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import src.block as B
from experiments.e_d1_decode import f05, ntrue_table
from experiments.e_m1_lgbm import PARAMS, ROUNDS
from src.config import SEED, WORK_DIR
from src.decode import ef_decode, exclusive
from src.evalx import candidate_metrics, log_experiment, truth_pairs
from src.features import FEATURES, pair_features
from src.features2 import NEW, competition_features, diff_features, mask_keyonly
from src.partition import infer_state

K = 5
QUERY_ROWS = 400  # rows per top-k chunk against a country-wide index
NEW = WORK_DIR / "e_b5_new_pairs.parquet"
FEATS = WORK_DIR / "e_b5_feats.parquet"
FEAT_COLS = ["entity_id", "name_tok", "addr", "nums", "hn", "name_ns", "name_key", "nonlatin", "empty_addr"]


def nostate_candidates(s1: pl.DataFrame, recs: pl.DataFrame, alias: pl.DataFrame, k: int = K) -> pl.DataFrame:
    """(s1, rec, cos_name, cos_addr, cos_comb) for records with no inferred state, per country."""
    recs = recs.with_columns(infer_state(recs, alias).alias("state")).filter(pl.col("state").is_null())
    out = []
    for (country,), q in recs.group_by("country"):
        g = s1.filter(pl.col("country") == country)
        if g.height == 0:
            continue
        B.MAX_CHUNK_CELLS = QUERY_ROWS * g.height
        n1, nr = B._fit(g["name_tok"].to_list(), q["name_tok"].to_list(), analyzer="char_wb",
                        ngram_range=(3, 3), min_df=1, max_df=B.MAX_DF, sublinear_tf=True)
        pairs = B._topk(nr, n1, k, "x").select("r", "s")
        r, s = pairs["r"].to_numpy(), pairs["s"].to_numpy()
        # cosine features from full (unpruned) vectors, as in tfidf_candidates
        n1, nr = B._fit(g["name_tok"].to_list(), q["name_tok"].to_list(), analyzer="char_wb", ngram_range=(3, 3), min_df=1, sublinear_tf=True)
        a1, ar = B._fit(g["addr"].to_list(), q["addr"].to_list(), analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        cn, ca = B._pair_cos(nr, n1, r, s), B._pair_cos(ar, a1, r, s)
        out.append(pl.DataFrame({"s1": g["entity_id"].to_numpy()[s], "rec": q["entity_id"].to_numpy()[r],
                                 "cos_name": cn, "cos_addr": ca, "cos_comb": (B.W_NAME * cn + B.W_ADDR * ca).astype(np.float32)}))
        print(f"  {country}: {q.height:,} no-state records -> {out[-1].height:,} pairs", flush=True)
    return pl.concat(out)


def retrieve() -> None:
    t0 = time.time()
    cols = ["entity_id", "country", "name_tok", "addr", "parts"]
    s1 = pl.read_parquet(WORK_DIR / "norm/train_s1.parquet", columns=cols)
    recs = pl.concat([pl.read_parquet(WORK_DIR / f"norm/train_s{s}.parquet", columns=cols) for s in (2, 3)])
    new = nostate_candidates(s1, recs, pl.read_parquet(WORK_DIR / "alias_train.parquet"))
    sample = pl.read_parquet(WORK_DIR / "e_b3_sample_s1.parquet")
    new = new.join(sample.select(pl.col("entity_id").alias("s1")), on="s1")  # validation S1 only, like E-B3
    new.write_parquet(NEW)
    old = pl.scan_parquet(WORK_DIR / "e_m1_feats_k5/*.parquet").select("s1", "rec").collect()
    truth, ids = truth_pairs(), sample["entity_id"]
    log_experiment("E-B5:cands_old", candidate_metrics(old, truth, ids), t0)
    log_experiment("E-B5:cands_old+nostate", candidate_metrics(pl.concat([old, new.select("s1", "rec")]).unique(), truth, ids), t0)


def build_feats() -> pl.DataFrame:
    """E-F2b features over (old candidates ∪ no-state pairs): pair features recomputed for every record
    that gained candidates, then competition/difference features recomputed on the full union
    (S1-side maxima can change), then the key-only mask."""
    base = pl.read_parquet(WORK_DIR / "e_f2b_feats.parquet", columns=["s1", "rec", "y", "country", "state", *FEATURES])
    new = pl.read_parquet(NEW)
    aff = new.select("rec").unique()
    old_aff = base.join(aff, on="rec").select("s1", "rec", "cos_name", "cos_addr", "cos_comb", (pl.col("in_keys") > 0.5).alias("in_keys"))
    new = new.join(old_aff.select("s1", "rec"), on=["s1", "rec"], how="anti").with_columns(pl.lit(False).alias("in_keys"))
    cands = pl.concat([old_aff, new.select(old_aff.columns)], how="vertical_relaxed")
    s1 = pl.read_parquet(WORK_DIR / "norm/train_s1.parquet", columns=FEAT_COLS).join(cands.select(pl.col("s1").alias("entity_id")).unique(), on="entity_id")
    recs = pl.concat([pl.read_parquet(WORK_DIR / f"norm/train_s{s}.parquet", columns=FEAT_COLS) for s in (2, 3)]).join(aff.rename({"rec": "entity_id"}), on="entity_id")
    f = pair_features(cands, s1, recs).select("s1", "rec", *[pl.col(x).cast(pl.Float32) for x in FEATURES])
    sample = pl.read_parquet(WORK_DIR / "e_b3_sample_s1.parquet").select(pl.col("entity_id").alias("s1"), "country", "state")
    t = truth_pairs().drop_nulls().with_columns(pl.lit(1, pl.Int8).alias("y"))
    f = f.join(t, on=["s1", "rec"], how="left").with_columns(pl.col("y").fill_null(0)).join(sample, on="s1")
    union = pl.concat([base.join(aff, on="rec", how="anti"), f.select(base.columns)], how="vertical_relaxed")
    del base, f
    comp = competition_features(union.select("s1", "rec", "cos_name", "cos_addr", "cos_comb"))
    ids = union.select("s1", "rec")
    cols = ["entity_id", "name_tok", "hn"]
    s1t = pl.read_parquet(WORK_DIR / "norm/train_s1.parquet", columns=cols).join(ids.select(pl.col("s1").alias("entity_id")).unique(), on="entity_id")
    rt = pl.concat([pl.read_parquet(WORK_DIR / f"norm/train_s{s}.parquet", columns=cols) for s in (2, 3)]) \
           .join(ids.select(pl.col("rec").alias("entity_id")).unique(), on="entity_id")
    diff = diff_features(ids, s1t, rt)
    return mask_keyonly(union.join(comp, on=["s1", "rec"]).join(diff, on=["s1", "rec"]))


def evaluate() -> None:
    """M1 (FEATURES + NEW, masked) out-of-fold on the union; baseline E-F2b 0.9585 (same folds)."""
    t0 = time.time()
    feats = build_feats()
    feats.write_parquet(FEATS)
    parts = sorted(set(feats.select("country", "state").unique().rows()))
    fold_of = {p: int(f) for p, f in zip(parts, np.random.default_rng(SEED).permutation(len(parts)) % 2)}  # E-M1 folds
    feats = feats.join(pl.DataFrame({"country": [p[0] for p in parts], "state": [p[1] for p in parts],
                                     "fold": [fold_of[p] for p in parts]}), on=["country", "state"])
    cols = FEATURES + NEW
    oof = []
    for f in (0, 1):
        tr, va = feats.filter(pl.col("fold") != f), feats.filter(pl.col("fold") == f)
        model = lgb.train({**PARAMS, "num_threads": 8}, lgb.Dataset(tr.select(cols).to_numpy(), tr["y"].to_numpy()), ROUNDS)
        oof.append(va.select("s1", "rec", "country", "y").with_columns(pl.Series("p", model.predict(va.select(cols).to_numpy()))))
        del tr, va, model
    scored = pl.concat(oof)
    scored.write_parquet(WORK_DIR / "e_b5_m1_oof.parquet")
    nt = ntrue_table()
    m = f05(ef_decode(exclusive(scored.select(pl.col("s1").hash(), pl.col("rec").hash(), "y", "p"))), nt, by_country=True)
    log_experiment("E-B5:m1v2_masked+nostate+ef", {**m, "n_pairs": scored.height}, t0)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].isdigit() else "all"
    if stage in ("retrieve", "all"):
        retrieve()
    if stage in ("eval", "all"):
        evaluate()
