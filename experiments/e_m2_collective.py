"""E-M2: stage-2 matcher with collective (entity-level) and graph-twin features.

Hypothesis: the M1 gap to the ceiling (0.942 vs 0.990) is mostly distractor discrimination, and
evidence from OTHER records improves it: how strongly other records claim the same S1, where this
record ranks among all claimants, and whether this record's exact-key twins (E-G1 components)
point to the same S1 (graph evidence used softly, as features, not as hard merges).

Stacking uses M1's out-of-fold stage-1 probability p1 and the SAME partition folds, so no pair's
features are computed from a model that saw its own label.
"""
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from experiments.e_g1_twins import components
from experiments.e_m1_lgbm import PARAMS, ROUNDS, decode
from src.config import SEED, WORK_DIR
from src.evalx import log_experiment, macro_f05_pairs, truth_pairs
from src.features import FEATURES

K = 5
COLLECTIVE = ["p1", "p1_rank_rec", "p1_margin_rec", "p1_sum_rec", "claims_s1", "best_other_p1_s1",
              "rank_among_claimants", "n_twins", "twins_agree", "twins_best_p1_same"]


def collective(scored: pl.DataFrame, comp: pl.DataFrame) -> pl.DataFrame:
    s = scored.select("s1", "rec", pl.col("p").alias("p1"))
    s = s.with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("rec").alias("p1_rank_rec"),
        pl.col("p1").sum().over("rec").alias("p1_sum_rec"),
    ).with_columns(
        (pl.col("p1") - pl.col("p1").filter(pl.col("p1_rank_rec") == 2).first().over("rec")).fill_null(pl.col("p1"))
        .alias("p1_margin_rec"),
        pl.col("p1").rank("ordinal", descending=True).over("s1").alias("rank_among_claimants"),
    )
    # claims: records whose BEST candidate is this S1 with p1 >= 0.5 (excluding this record)
    best = s.filter(pl.col("p1_rank_rec") == 1)
    claims = best.filter(pl.col("p1") >= 0.5).group_by("s1").len("claims_all")
    s = s.join(claims, on="s1", how="left").with_columns(
        (pl.col("claims_all").fill_null(0) - ((pl.col("p1_rank_rec") == 1) & (pl.col("p1") >= 0.5)).cast(pl.UInt32))
        .alias("claims_s1"))
    # best p1 of any OTHER record on this S1: top-2 per S1, take the one that is not this record
    top2 = s.sort("p1", descending=True).group_by("s1", maintain_order=True).head(2) \
            .group_by("s1").agg(pl.col("rec"), pl.col("p1").alias("tp"))
    s = s.join(top2, on="s1").with_columns(
        pl.when(pl.col("rec") == pl.col("rec_right").list.get(0))
          .then(pl.col("tp").list.get(1, null_on_oob=True)).otherwise(pl.col("tp").list.get(0))
          .fill_null(0.0).alias("best_other_p1_s1")).drop("rec_right", "tp")
    # graph twins: do this record's twins pick the same S1 as their best candidate?
    tw = comp.select(pl.col("entity_id").alias("rec"), "comp")
    best_s1 = best.select("rec", pl.col("s1").alias("twin_best"), pl.col("p1").alias("twin_p1")).join(tw, on="rec")
    pairs_tw = s.select("s1", "rec").join(tw, on="rec").join(best_s1.rename({"rec": "twin"}), on="comp") \
                .filter(pl.col("twin") != pl.col("rec"))
    agg = pairs_tw.group_by("s1", "rec").agg(
        pl.len().alias("n_twins"),
        (pl.col("twin_best") == pl.col("s1")).sum().alias("twins_agree"),
        pl.when(pl.col("twin_best") == pl.col("s1")).then(pl.col("twin_p1")).max().alias("twins_best_p1_same"))
    s = s.join(agg, on=["s1", "rec"], how="left").with_columns(
        pl.col("n_twins", "twins_agree").fill_null(0), pl.col("twins_best_p1_same").fill_null(0.0))
    return s.drop("claims_all")


def main() -> None:
    t0 = time.time()
    scored = pl.read_parquet(WORK_DIR / f"e_m1_oof_k{K}.parquet")
    recs = pl.concat([pl.read_parquet(WORK_DIR / f"norm/train_s{s}.parquet",
                                      columns=["entity_id", "country", "name_key", "name_ns", "hn", "addr"]) for s in (2, 3)])
    comp = components(recs)
    del recs
    col = collective(scored, comp).with_columns(pl.col(c).cast(pl.Float32) for c in COLLECTIVE)
    del comp, scored
    feats = pl.read_parquet(WORK_DIR / f"e_m1_feats_k{K}" / "*.parquet").join(col, on=["s1", "rec"])
    del col
    parts = sorted(set(feats.select("country", "state").unique().rows()))
    rng = np.random.default_rng(SEED)
    fold_of = {p: int(f) for p, f in zip(parts, rng.permutation(len(parts)) % 2)}  # same folds as M1
    folds = pl.DataFrame({"country": [p[0] for p in parts], "state": [p[1] for p in parts],
                          "fold": [fold_of[p] for p in parts]}, schema_overrides={"fold": pl.Int8})
    feats = feats.join(folds, on=["country", "state"])
    cols = FEATURES + COLLECTIVE
    oof = []
    for f in (0, 1):
        tr, va = feats.filter(pl.col("fold") != f), feats.filter(pl.col("fold") == f)
        model = lgb.train(PARAMS, lgb.Dataset(tr.select(cols).to_numpy(), tr["y"].to_numpy()), ROUNDS)
        oof.append(va.select("s1", "rec", "country", "y").with_columns(pl.Series("p", model.predict(va.select(cols).to_numpy()))))
        if f == 0:
            g = model.feature_importance("gain")
            print("top gain:", [(n, round(v / g.sum(), 3)) for n, v in sorted(zip(cols, g), key=lambda x: -x[1])[:10]], flush=True)
        del tr, va
    scored2 = pl.concat(oof)
    scored2.write_parquet(WORK_DIR / f"e_m2_oof_k{K}.parquet")
    del feats
    sample = pl.read_parquet(WORK_DIR / "e_b3_sample_s1.parquet")
    ids, truth = sample["entity_id"], truth_pairs()
    best = None
    for t in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        m = macro_f05_pairs(decode(scored2, t), truth, ids)
        print(f"t={t:.2f} f05={m['f05']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}", flush=True)
        if best is None or m["f05"] > best[1]["f05"]:
            best = (t, m)
    t, m = best
    log_experiment("E-M2:collective", {**m, "t": t}, t0)
    pred = decode(scored2, t)
    for (c,), g in sample.group_by("country"):
        log_experiment(f"E-M2:collective:{c}", macro_f05_pairs(pred, truth, g["entity_id"]), t0)


if __name__ == "__main__":
    main()
