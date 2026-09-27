"""E-G2: entity-group gain features (src/groups.py) at stage 2, scored as-is and in the doubled-decoy world.

Base = stage-2 data of ens3gca (v5 M1 OOF p1 >= 0.01) with the E-P2 'no_twins' collective set. In the x2 world a
duplicated decoy is an exact copy, so its gains equal its original's and the original gains a sibling with gain 0:
group features of every x2 row = those of its original (no recomputation needed).
Keep only if as-is F0.5 rises in both folds without losing the x2 score.

Usage: python -m experiments.e_g2_groups
"""
import time

import numpy as np
import polars as pl

import experiments.e_p2_density as P
from experiments.e_d1_decode import ntrue_table
from src.config import WORK_DIR
from src.evalx import log_experiment, truth_pairs
from src.groups import GRP, group_features

E = P.E
OUT = E.ENS / "grp.parquet"


def build() -> None:
    t0 = time.time()
    sc = pl.read_parquet(E.SRC_OOF).filter(pl.col("p") >= 0.01).select("s1", "rec", pl.col("p").alias("p1"))
    sc = sc.join(pl.scan_parquet(E.SRC_FEATS).select("s1", "rec", "name_tset", "addr_tset").collect(), on=["s1", "rec"])
    recs = pl.concat([pl.read_parquet(WORK_DIR / f"norm/train_s{s}.parquet", columns=["entity_id", "name_tok", "addr", "hn"]) for s in (2, 3)]) \
             .join(sc.select(pl.col("rec").alias("entity_id")).unique(), on="entity_id")
    g = group_features(sc, recs)
    owned = truth_pairs().drop_nulls().select("rec").unique()
    dec = g.join(owned, on="rec", how="anti").with_columns((pl.col("rec") + "#d").alias("rec"))   # x2 duplicates
    pl.concat([g, dec]).select(pl.col("s1").hash(), pl.col("rec").hash(), *GRP).unique(["s1", "rec"]).write_parquet(OUT)
    print(f"group features: {g.height:,} pairs, {(g['g_p1_gain'] > 0).sum():,} with a better sibling ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    t0 = time.time()
    if not OUT.exists():
        build()
    g = pl.read_parquet(OUT)
    d = pl.read_parquet(E.DATA).join(g, on=["s1", "rec"], how="left").with_columns(pl.col(c).fill_null(0) for c in GRP)
    d2 = pl.read_parquet(P.X2).join(g, on=["s1", "rec"], how="left").with_columns(pl.col(c).fill_null(0) for c in GRP)
    nt = ntrue_table()
    y, fold, fold2 = d["y"].to_numpy(), d["fold"].to_numpy(), d2["fold"].to_numpy()
    for name, extra in (("no_twins+grp", GRP),):
        cols = P.BASE + P.SETS["no_twins"] + extra
        X, X2m = d.select(cols).to_numpy().astype(np.float32), d2.select(cols).to_numpy().astype(np.float32)
        p, p2 = np.zeros(len(y)), np.zeros(d2.height)
        for f in (0, 1):
            m = E.GRID["lgbm"]["b"]().fit(X[fold != f], y[fold != f])
            p[fold == f] = m.predict_proba(X[fold == f])[:, 1]
            p2[fold2 == f] = m.predict_proba(X2m[fold2 == f])[:, 1]
            if f == 0:
                gain = m.b.feature_importance("gain")
                print("GRP gain shares:", {n: round(float(v / gain.sum()), 3) for n, v in zip(cols, gain) if n in GRP}, flush=True)
        log_experiment(f"E-G2:{name}:as_is", E.score(p, d, nt), t0)
        log_experiment(f"E-G2:{name}:x2", E.score(p2, d2, nt), t0)


if __name__ == "__main__":
    main()
