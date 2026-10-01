"""E-T2: adaptive generic/rare name-token features (src/adaptive.py) on top of a base feature set.

Usage: python -m experiments.e_t2_adaptive <base>   (base = e_f2b or e_f2b_tr; baseline = its M1 OOF)
Generic vocabulary is fitted label-free on all train S1 (per country). Same folds / params as E-M1.
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from experiments.e_d1_decode import f05, ntrue_table
from experiments.e_m1_lgbm import PARAMS, ROUNDS
from resolver.adaptive import GEN, generic_features, generic_tokens
from competition.config import SEED, WORK_DIR
from resolver.decode import ef_decode, exclusive
from competition.evalx import log_experiment
from resolver.features import FEATURES
from resolver.features2 import NEW

TR = ["tr_name_tset", "tr_name_ratio", "tr_name_jw", "tr_n_tok_extra"]


def main(base: str) -> None:
    t0 = time.time()
    feats = pl.read_parquet(WORK_DIR / f"{base}_feats.parquet")
    s1 = pl.read_parquet(WORK_DIR / "norm/train_s1.parquet", columns=["entity_id", "country", "name_tok"])
    gen = generic_tokens(s1)
    ids = feats.select("s1", "rec")
    recs = pl.concat([pl.read_parquet(WORK_DIR / f"norm/train_s{s}.parquet", columns=["entity_id", "name_tok"]) for s in (2, 3)]) \
             .join(ids.select(pl.col("rec").alias("entity_id")).unique(), on="entity_id")
    feats = feats.join(generic_features(ids, s1, recs, gen), on=["s1", "rec"])
    out = f"{base}_gen"
    feats.write_parquet(WORK_DIR / f"{out}_feats.parquet")
    parts = sorted(set(feats.select("country", "state").unique().rows()))
    fold_of = {p: int(f) for p, f in zip(parts, np.random.default_rng(SEED).permutation(len(parts)) % 2)}
    feats = feats.join(pl.DataFrame({"country": [p[0] for p in parts], "state": [p[1] for p in parts],
                                     "fold": [fold_of[p] for p in parts]}), on=["country", "state"])
    cols = FEATURES + NEW + (TR if "_tr" in base else []) + GEN
    oof = []
    for f in (0, 1):
        tr, va = feats.filter(pl.col("fold") != f), feats.filter(pl.col("fold") == f)
        model = lgb.train({**PARAMS, "num_threads": 8}, lgb.Dataset(tr.select(cols).to_numpy(), tr["y"].to_numpy()), ROUNDS)
        oof.append(va.select("s1", "rec", "country", "y").with_columns(pl.Series("p", model.predict(va.select(cols).to_numpy()))))
        if f == 0:
            g = model.feature_importance("gain")
            print("GEN gain shares:", {n: round(float(v / g.sum()), 3) for n, v in zip(cols, g) if n in GEN}, flush=True)
        del tr, va, model
    scored = pl.concat(oof)
    scored.write_parquet(WORK_DIR / f"{out}_m1_oof.parquet")
    nt = ntrue_table()
    m = f05(ef_decode(exclusive(scored.select(pl.col("s1").hash(), pl.col("rec").hash(), "y", "p"))), nt, by_country=True)
    log_experiment(f"E-T2:m1[{out}]+ef", m, t0)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "e_f2b")
