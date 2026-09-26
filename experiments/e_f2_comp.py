"""E-F2: M1 with competition + difference features (src/features2.py) vs M1 baseline.

Same candidates (E-M1 validation features), same partition folds, same LightGBM params; scored
out-of-fold with exclusive + expected-F decoding. Baseline: M1 + EF = 0.9429 (E-D1).
Writes work/e_f2_feats.parquet (features + NEW) and work/e_f2_m1_oof.parquet for later stages.
"""
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from experiments.e_d1_decode import f05, ntrue_table
from experiments.e_m1_lgbm import PARAMS, ROUNDS
from src.config import SEED, WORK_DIR
from src.decode import ef_decode, exclusive
from src.evalx import log_experiment
from src.features import FEATURES
from src.features2 import NEW, competition_features, diff_features, mask_keyonly

FEATS = WORK_DIR / "e_f2_feats.parquet"


def build() -> pl.DataFrame:
    feats = pl.read_parquet(WORK_DIR / "e_m1_feats_k5" / "*.parquet")
    comp = competition_features(feats.select("s1", "rec", "cos_name", "cos_addr", "cos_comb"))
    cols = ["entity_id", "name_tok", "hn"]
    ids = feats.select("s1", "rec")
    s1 = pl.read_parquet(WORK_DIR / "norm/train_s1.parquet", columns=cols).join(ids.select(pl.col("s1").alias("entity_id")).unique(), on="entity_id")
    recs = pl.concat([pl.read_parquet(WORK_DIR / f"norm/train_s{s}.parquet", columns=cols) for s in (2, 3)]) \
             .join(ids.select(pl.col("rec").alias("entity_id")).unique(), on="entity_id")
    diff = diff_features(ids, s1, recs)
    return feats.join(comp, on=["s1", "rec"]).join(diff, on=["s1", "rec"])


def main(mask: bool = False) -> None:
    """mask=True (E-F2b): reuse E-F2 features, blank record-side features on key-only pairs."""
    t0 = time.time()
    tag = "e_f2b" if mask else "e_f2"
    if mask:
        feats = mask_keyonly(pl.read_parquet(FEATS))
        feats.write_parquet(WORK_DIR / f"{tag}_feats.parquet")
    else:
        feats = build()
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
        if f == 0:
            g = model.feature_importance("gain")
            print("top gain:", [(n, round(float(v / g.sum()), 3)) for n, v in sorted(zip(cols, g), key=lambda x: -x[1])[:12]], flush=True)
        del tr, va, model
    scored = pl.concat(oof)
    scored.write_parquet(WORK_DIR / f"{tag}_m1_oof.parquet")
    nt = ntrue_table()
    ex = exclusive(scored.select(pl.col("s1").hash(), pl.col("rec").hash(), "y", "p"))
    name = "E-F2b:m1+new+mask" if mask else "E-F2:m1+new"
    log_experiment(f"{name}+ef", f05(ef_decode(ex), nt, by_country=True), t0)
    log_experiment(f"{name}+thr0.7", f05(ex.filter(pl.col("p") >= 0.7), nt), t0)


if __name__ == "__main__":
    import sys
    main(mask="mask" in sys.argv[1:])
