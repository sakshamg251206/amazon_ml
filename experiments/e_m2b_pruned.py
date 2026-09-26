"""E-M2b: M2 restricted to pairs with stage-1 p1 >= PRUNE (needed to fit the test set in 8 GB),
validated with the same partition folds + expected-F decoding; then the final M2 is trained on
all sample rows and saved for the pipeline. Pass if F0.5 stays at E-M2 + EF level (0.9482)."""
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from experiments.e_d1_decode import f05, ntrue_table
from experiments.e_g1_twins import components
from experiments.e_m1_lgbm import PARAMS, ROUNDS
from experiments.e_m2_collective import COLLECTIVE, collective
from src.config import SEED, WORK_DIR
from src.decode import ef_decode, exclusive
from src.evalx import log_experiment
from src.features import FEATURES

PRUNE = 0.01
MODEL_M2 = WORK_DIR / "model_m2.txt"


def main() -> None:
    t0 = time.time()
    scored = pl.read_parquet(WORK_DIR / "e_m1_oof_k5.parquet").filter(pl.col("p") >= PRUNE)
    print(f"pairs kept at p1>={PRUNE}: {scored.height:,}", flush=True)
    recs = pl.concat([pl.read_parquet(WORK_DIR / f"norm/train_s{s}.parquet",
                                      columns=["entity_id", "country", "name_key", "name_ns", "hn", "addr"]) for s in (2, 3)])
    comp = components(recs)
    del recs
    col = collective(scored, comp).with_columns(pl.col(c).cast(pl.Float32) for c in COLLECTIVE)
    del comp, scored
    feats = pl.scan_parquet(WORK_DIR / "e_m1_feats_k5" / "*.parquet").join(col.lazy(), on=["s1", "rec"]).collect()
    del col
    parts = sorted(set(feats.select("country", "state").unique().rows()))
    rng = np.random.default_rng(SEED)
    fold_of = {p: int(f) for p, f in zip(parts, rng.permutation(len(parts)) % 2)}  # same folds as M1
    feats = feats.join(pl.DataFrame({"country": [p[0] for p in parts], "state": [p[1] for p in parts],
                                     "fold": [fold_of[p] for p in parts]}), on=["country", "state"])
    cols = FEATURES + COLLECTIVE
    oof = []
    for f in (0, 1):
        tr, va = feats.filter(pl.col("fold") != f), feats.filter(pl.col("fold") == f)
        model = lgb.train(PARAMS, lgb.Dataset(tr.select(cols).to_numpy(), tr["y"].to_numpy()), ROUNDS)
        oof.append(va.select(pl.col("s1").hash(), pl.col("rec").hash(), "y")
                     .with_columns(pl.Series("p", model.predict(va.select(cols).to_numpy()))))
        del tr, va, model
    m = f05(ef_decode(exclusive(pl.concat(oof))), ntrue_table(), by_country=True)
    log_experiment(f"E-M2b:pruned{PRUNE}+ef", m, t0)
    final = lgb.train(PARAMS, lgb.Dataset(feats.select(cols).to_numpy(), feats["y"].to_numpy()), ROUNDS)
    final.save_model(str(MODEL_M2))
    print(f"final M2 trained on {feats.height:,} pairs -> {MODEL_M2}", flush=True)


if __name__ == "__main__":
    main()
