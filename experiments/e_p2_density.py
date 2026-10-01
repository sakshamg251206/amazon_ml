"""E-P2: which stage-2 feature set survives test's decoy density? (labelled, on validation)

Test has ~2x the decoy records per S1. World 'x2': every decoy record (owned by no S1) that is a candidate of a
validation S1 gets an exact duplicate; twin components and ALL collective features are recomputed, so S1-side
counts (claims_s1, twins, ranks) inflate exactly as they would on test. Pair features and p1 are unchanged.
Stage-2 LightGBM (cfg b) per feature set is trained on the normal world (fold f) and scored on fold 1-f of
both worlds. Pick the set with the best x2 score that does not lose on the normal world.

Usage: python -m experiments.e_p2_density
"""
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import experiments.e_e2_ensemble_v2 as V
from experiments.e_d1_decode import ntrue_table
from experiments.e_g1_twins import COLS as TWIN_COLS, components
from experiments.e_m2_collective import COLLECTIVE, collective
from competition.config import SEED, WORK_DIR
from competition.evalx import log_experiment, truth_pairs

V.configure("e_f2b_gen_ca", "ens3gca")
E = V.E
BASE = [c for c in E.COLS if c not in COLLECTIVE]
S1_SIDE = ["claims_s1", "rank_among_claimants", "best_other_p1_s1", "n_twins", "twins_agree", "twins_best_p1_same"]
SETS = {
    "all": COLLECTIVE,
    "no_s1_side": [c for c in COLLECTIVE if c not in S1_SIDE],
    "no_counts": [c for c in COLLECTIVE if c not in ("claims_s1", "n_twins", "twins_agree")],
    "no_twins": [c for c in COLLECTIVE if c not in ("n_twins", "twins_agree", "twins_best_p1_same")],
}
from competition.groups import GRP  # noqa: E402  E-G2: entity-group gains replace the twin features (0.9620 / x2 0.9534)
SETS["no_twins_grp"] = SETS["no_twins"] + GRP
X2 = E.ENS / "data_x2.parquet"


def build_x2() -> None:
    scored = pl.read_parquet(E.SRC_OOF).filter(pl.col("p") >= 0.01).select("s1", "rec", "p")
    owned = truth_pairs().drop_nulls().select("rec").unique()
    dec = scored.select("rec").unique().join(owned, on="rec", how="anti")
    dup = scored.join(dec, on="rec").with_columns((pl.col("rec") + "#d").alias("rec"))
    recs = pl.concat([pl.read_parquet(WORK_DIR / f"norm/train_s{s}.parquet", columns=TWIN_COLS) for s in (2, 3)])
    recs = pl.concat([recs, recs.join(dec.rename({"rec": "entity_id"}), on="entity_id").with_columns(pl.col("entity_id") + "#d")])
    comp = components(recs)
    del recs
    col = collective(pl.concat([scored, dup]), comp).with_columns(pl.col(c).cast(pl.Float32) for c in COLLECTIVE)
    del comp
    col = col.with_columns(pl.col("rec").str.strip_suffix("#d").alias("orig"), pl.col("rec").str.ends_with("#d").alias("is_dup"))
    feats = pl.scan_parquet(E.SRC_FEATS).select("s1", pl.col("rec").alias("orig"), "country", "state", "y", *BASE) \
              .join(col.select("s1", "orig").unique().lazy(), on=["s1", "orig"], how="semi").collect()
    d = col.join(feats, on=["s1", "orig"])
    d = d.with_columns(pl.when(pl.col("is_dup")).then(0).otherwise(pl.col("y")).cast(pl.Int8).alias("y"))
    parts = sorted(set(d.select("country", "state").unique().rows()))
    fold_of = {p: int(f) for p, f in zip(parts, np.random.default_rng(SEED).permutation(len(parts)) % 2)}
    d = d.join(pl.DataFrame({"country": [p[0] for p in parts], "state": [p[1] for p in parts],
                             "fold": [fold_of[p] for p in parts]}), on=["country", "state"])
    d.select(pl.col("s1").hash(), pl.col("rec").hash(), "y", "fold", "is_dup", *E.COLS).write_parquet(X2)
    print(f"x2 world: {d.height:,} rows ({d['is_dup'].sum():,} duplicated decoy pairs)", flush=True)


def main(only: list[str] | None = None) -> None:
    t0 = time.time()
    if not X2.exists():
        build_x2()
    d, d2, nt = pl.read_parquet(E.DATA), pl.read_parquet(X2), ntrue_table()
    y, fold, fold2 = d["y"].to_numpy(), d["fold"].to_numpy(), d2["fold"].to_numpy()
    for name, coll in SETS.items():
        if only and name not in only:
            continue
        cols = BASE + coll
        X, X2m = d.select(cols).to_numpy().astype(np.float32), d2.select(cols).to_numpy().astype(np.float32)
        p, p2 = np.zeros(len(y)), np.zeros(d2.height)
        for f in (0, 1):
            m = E.GRID["lgbm"]["b"]().fit(X[fold != f], y[fold != f])
            p[fold == f] = m.predict_proba(X[fold == f])[:, 1]
            p2[fold2 == f] = m.predict_proba(X2m[fold2 == f])[:, 1]
        log_experiment(f"E-P2:{name}:as_is", E.score(p, d, nt), t0)
        log_experiment(f"E-P2:{name}:x2", E.score(p2, d2, nt), t0)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
