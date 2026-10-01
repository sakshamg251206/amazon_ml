"""E-A1: canonical-address similarity features (src/address.py) on top of a base feature set.

Usage: python -m experiments.e_a1_address <base>   (default e_f2b_gen; baseline = its M1 OOF, 0.9594)
Validated on US/India (France has no labels); kept only if F0.5 rises in both folds.
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from experiments.e_d1_decode import f05, ntrue_table
from experiments.e_m1_lgbm import PARAMS, ROUNDS
from resolver.adaptive import GEN
from competition.address import CA, canon_addr, canon_features
from competition.config import SEED, WORK_DIR
from resolver.decode import ef_decode, exclusive
from competition.evalx import log_experiment
from resolver.features import FEATURES
from resolver.features2 import NEW


def fold_scores(scored: pl.DataFrame, nt: pl.DataFrame) -> dict:
    sample = pl.read_parquet(WORK_DIR / "e_b3_sample_s1.parquet")
    parts = sorted(set(sample.select("country", "state").unique().rows()))
    fold_of = {p: int(f) for p, f in zip(parts, np.random.default_rng(SEED).permutation(len(parts)) % 2)}
    sf = sample.select(pl.col("entity_id").hash().alias("s1"),
                       pl.struct("country", "state").map_elements(lambda r: fold_of[(r["country"], r["state"])], return_dtype=pl.Int8).alias("fold"))
    pred = ef_decode(exclusive(scored.select(pl.col("s1").hash(), pl.col("rec").hash(), "y", "p")))
    out = f05(pred, nt, by_country=True)
    for f in (0, 1):
        ids = sf.filter(pl.col("fold") == f).select("s1")
        out[f"f05_fold{f}"] = f05(pred.join(ids, on="s1"), nt.join(ids, on="s1"))["f05"]
    return out


def main(base: str) -> None:
    t0 = time.time()
    feats = pl.read_parquet(WORK_DIR / f"{base}_feats.parquet")
    ids = feats.select("s1", "rec")
    cols = ["entity_id", "country", "parts"]
    s1 = pl.read_parquet(WORK_DIR / "norm/train_s1.parquet", columns=cols).join(ids.select(pl.col("s1").alias("entity_id")).unique(), on="entity_id")
    recs = pl.concat([pl.read_parquet(WORK_DIR / f"norm/train_s{s}.parquet", columns=cols) for s in (2, 3)]) \
             .join(ids.select(pl.col("rec").alias("entity_id")).unique(), on="entity_id")
    feats = feats.join(canon_features(ids, canon_addr(s1), canon_addr(recs)), on=["s1", "rec"])
    out = f"{base}_ca"
    feats.write_parquet(WORK_DIR / f"{out}_feats.parquet")
    parts = sorted(set(feats.select("country", "state").unique().rows()))
    fold_of = {p: int(f) for p, f in zip(parts, np.random.default_rng(SEED).permutation(len(parts)) % 2)}
    feats = feats.join(pl.DataFrame({"country": [p[0] for p in parts], "state": [p[1] for p in parts],
                                     "fold": [fold_of[p] for p in parts]}), on=["country", "state"])
    use = FEATURES + NEW + (GEN if "_gen" in base else []) + CA
    oof = []
    for f in (0, 1):
        tr, va = feats.filter(pl.col("fold") != f), feats.filter(pl.col("fold") == f)
        model = lgb.train({**PARAMS, "num_threads": 8}, lgb.Dataset(tr.select(use).to_numpy(), tr["y"].to_numpy()), ROUNDS)
        oof.append(va.select("s1", "rec", "country", "y").with_columns(pl.Series("p", model.predict(va.select(use).to_numpy()))))
        if f == 0:
            g = model.feature_importance("gain")
            print("CA gain shares:", {n: round(float(v / g.sum()), 3) for n, v in zip(use, g) if n in CA}, flush=True)
        del tr, va, model
    scored = pl.concat(oof)
    scored.write_parquet(WORK_DIR / f"{out}_m1_oof.parquet")
    nt = ntrue_table()
    log_experiment(f"E-A1:m1[{out}]+ef", fold_scores(scored, nt), t0)
    base_oof = pl.read_parquet(WORK_DIR / f"{base}_m1_oof.parquet")
    log_experiment(f"E-A1:baseline[{base}]", fold_scores(base_oof, nt), t0)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "e_f2b_gen")
