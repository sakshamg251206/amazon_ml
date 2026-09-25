"""E-M1: learned matcher on the E-B3 candidate set.

Hypothesis: country-agnostic fuzzy + number + rank features scored by LightGBM, decoded with
"each record -> its best S1 if p >= t", beat v0_rule (0.5950) by a wide margin.
Validation: 2-fold split BY PARTITION (fit on half of the sampled states, predict the other half,
swap), so every S1 is scored out-of-fold. The same OOF scores are saved for decoding experiments.
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from src.config import SEED, WORK_DIR
from src.evalx import candidate_metrics, log_experiment, macro_f05_pairs, truth_pairs
from src.features import FEATURES, pair_features

K_TFIDF = int(sys.argv[1]) if len(sys.argv) > 1 else 5
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=63, min_data_in_leaf=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=SEED, verbose=-1, num_threads=8)
ROUNDS = 400
NORM = ["entity_id", "name_tok", "addr", "nums", "hn", "name_ns", "name_key", "nonlatin", "empty_addr"]


def decode(scored: pl.DataFrame, t: float) -> pl.DataFrame:
    """Each record keeps only its highest-probability S1, if p >= t."""
    best = scored.sort("p", descending=True).group_by("rec", maintain_order=True).first()
    return best.filter(pl.col("p") >= t).select("s1", "rec")


def main() -> None:
    """Stage 1: candidates -> labelled features on disk."""
    t0 = time.time()
    sample = pl.read_parquet(WORK_DIR / "e_b3_sample_s1.parquet")
    ids = sample["entity_id"]
    truth = truth_pairs()
    tf = pl.read_parquet(WORK_DIR / "e_b3_tfidf_cands.parquet")
    union = pl.read_parquet(WORK_DIR / "e_b3_union_cands.parquet")
    keys = union.join(tf.select("s1", "rec"), on=["s1", "rec"], how="anti").with_columns(pl.lit(True).alias("in_keys"))
    cands = pl.concat([
        tf.filter(pl.col("rank_comb") <= K_TFIDF).select("s1", "rec", "cos_name", "cos_addr", "cos_comb")
          .with_columns(pl.lit(False).alias("in_keys")),
        keys.with_columns(pl.lit(None, pl.Float32).alias(c) for c in ("cos_name", "cos_addr", "cos_comb"))
            .select("s1", "rec", "cos_name", "cos_addr", "cos_comb", "in_keys"),
    ], how="vertical_relaxed")
    cm = candidate_metrics(cands, truth, ids)
    log_experiment(f"E-M1:cands_k{K_TFIDF}", {**cm, "n_pairs": cands.height}, t0)

    s1 = pl.read_parquet(WORK_DIR / "norm/train_s1.parquet", columns=NORM).join(sample.select("entity_id"), on="entity_id")
    recs = pl.concat([pl.read_parquet(WORK_DIR / f"norm/train_s{s}.parquet", columns=NORM) for s in (2, 3)])
    recs = recs.join(cands.select(pl.col("rec").alias("entity_id")).unique(), on="entity_id")
    tf_ = time.time()
    # Features in chunks of whole records (a record's candidates stay together, so its
    # rank/margin features are exact); keep only compact numeric columns (8 GB machine).
    cands = cands.with_columns((pl.col("rec").hash(SEED) % 8).alias("chunk"))
    t_pairs = truth.drop_nulls().join(sample.select(pl.col("entity_id").alias("s1")), on="s1") \
                   .with_columns(pl.lit(1, pl.Int8).alias("y"))
    out_dir = WORK_DIR / f"e_m1_feats_k{K_TFIDF}"
    out_dir.mkdir(parents=True, exist_ok=True)
    part_of = sample.select(pl.col("entity_id").alias("s1"), "country", "state")
    n_pairs = 0
    for (i,), c in cands.group_by("chunk"):
        f = pair_features(c.drop("chunk"), s1, recs)
        f = f.select("s1", "rec", *[pl.col(x).cast(pl.Float32) for x in FEATURES]) \
             .join(t_pairs, on=["s1", "rec"], how="left").with_columns(pl.col("y").fill_null(0)) \
             .join(part_of, on="s1")
        f.write_parquet(out_dir / f"part_{i}.parquet")  # stream to disk: never hold all chunks
        n_pairs += f.height
        del f
    print(f"features: {n_pairs:,} pairs in {time.time() - tf_:.0f}s", flush=True)


def train_eval() -> None:
    """Stage 2 (fresh process): 2-fold-by-partition LightGBM, OOF decoding, threshold sweep."""
    t0 = time.time()
    feats = pl.read_parquet(WORK_DIR / f"e_m1_feats_k{K_TFIDF}" / "*.parquet")
    sample = pl.read_parquet(WORK_DIR / "e_b3_sample_s1.parquet")
    ids = sample["entity_id"]

    parts = sorted(set(feats.select("country", "state").unique().rows()))
    rng = np.random.default_rng(SEED)
    fold_of = {p: int(f) for p, f in zip(parts, rng.permutation(len(parts)) % 2)}
    folds = pl.DataFrame({"country": [p[0] for p in parts], "state": [p[1] for p in parts],
                          "fold": [fold_of[p] for p in parts]}, schema_overrides={"fold": pl.Int8})
    feats = feats.join(folds, on=["country", "state"])

    oof = []
    for f in (0, 1):
        tr, va = feats.filter(pl.col("fold") != f), feats.filter(pl.col("fold") == f)
        X = tr.select(FEATURES).to_numpy().astype(np.float32)
        model = lgb.train(PARAMS, lgb.Dataset(X, tr["y"].to_numpy()), ROUNDS)
        p = model.predict(va.select(FEATURES).to_numpy().astype(np.float32))
        oof.append(va.select("s1", "rec", "country", "y").with_columns(pl.Series("p", p)))
        if f == 0:
            imp = sorted(zip(FEATURES, model.feature_importance("gain")), key=lambda x: -x[1])[:10]
            print("top gain:", [(n, round(g / sum(model.feature_importance('gain')), 3)) for n, g in imp], flush=True)
    scored = pl.concat(oof)
    scored.write_parquet(WORK_DIR / f"e_m1_oof_k{K_TFIDF}.parquet")
    del feats, tr, va, X
    truth = truth_pairs()

    best = None
    for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        m = macro_f05_pairs(decode(scored, t), truth, ids)
        print(f"t={t:.2f} f05={m['f05']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}", flush=True)
        if best is None or m["f05"] > best[1]["f05"]:
            best = (t, m)
    t, m = best
    log_experiment(f"E-M1:lgbm_k{K_TFIDF}", {**m, "t": t}, t0)
    pred = decode(scored, t)
    for (c,), g in sample.group_by("country"):
        log_experiment(f"E-M1:lgbm_k{K_TFIDF}:{c}", macro_f05_pairs(pred, truth, g["entity_id"]), t0)


if __name__ == "__main__":
    if not (WORK_DIR / f"e_m1_feats_k{K_TFIDF}").exists():
        main()
    else:
        train_eval()
