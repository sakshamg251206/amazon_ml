"""E-P1: test-like decoy density. Test has ~2.3 decoy records per S1 vs 1.22 in train (records per S1
5.5-5.8 vs 4.68, copies per S1 3.46 in both train countries). Simulate by duplicating every decoy
record in validation; compare stage-2 LightGBM trained with decoy weight w (same folds, E-E2 data)."""
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import experiments.e_e2_ensemble_v2 as V
from experiments.e_d1_decode import f05, ntrue_table
from resolver.decode import ef_decode, exclusive
from competition.evalx import log_experiment, truth_pairs

V.configure(sys.argv[1] if len(sys.argv) > 1 else "e_f2b_gen", sys.argv[2] if len(sys.argv) > 2 else "ens3g")
E = V.E


def worlds(d: pl.DataFrame):
    dup = d.filter(~pl.col("owned")).with_columns((pl.col("rec") + 1).alias("rec"))
    return {"as_is": d, "decoys_x2": pl.concat([d, dup])}


def score(x, nt, sf, a=1.0):
    x = x.with_columns((a * pl.col("p") / (a * pl.col("p") + 1 - pl.col("p"))).alias("p"))
    pred = ef_decode(exclusive(x.select("s1", "rec", "y", "p")))
    out = {"f05": f05(pred, nt)["f05"]}
    for f in (0, 1):
        ids = sf.filter(pl.col("fold") == f).select("s1")
        out[f"fold{f}"] = f05(pred.join(ids, on="s1"), nt.join(ids, on="s1"))["f05"]
    return out


def main() -> None:
    t0 = time.time()
    d = pl.read_parquet(E.DATA)
    owned = truth_pairs().drop_nulls().select(pl.col("rec").hash().alias("rec")).unique().with_columns(pl.lit(True).alias("owned"))
    d = d.join(owned, on="rec", how="left").with_columns(pl.col("owned").fill_null(False))
    X, y, fold = d.select(E.COLS).to_numpy().astype(np.float32), d["y"].to_numpy(), d["fold"].to_numpy()
    nt, sf = ntrue_table(), d.select("s1", "fold").unique("s1")
    base = d.select("s1", "rec", "y", "fold", "owned")
    for w in (1.0, 2.0, 3.0):
        p = np.zeros(len(y))
        wt = np.where(d["owned"].to_numpy(), 1.0, w)
        for f in (0, 1):
            tr, va = fold != f, fold == f
            m = E._lgb(dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8), 700)
            m.b = lgb.train(m.params, lgb.Dataset(X[tr], y[tr], weight=wt[tr]), m.rounds)
            p[va] = m.b.predict(X[va])
        pl.DataFrame({"p": p}).write_parquet(E.ENS / f"oof_lgbm_decoyw{w:g}.parquet")
        for wn, world in worlds(base.with_columns(pl.Series("p", p))).items():
            log_experiment(f"E-P1[{E.ENS.name}]:lgbm_decoyw{w:g}:{wn}", score(world, nt, sf), t0)


if __name__ == "__main__":
    main()
