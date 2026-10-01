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

from experiments.e_m1_lgbm import PARAMS, ROUNDS, decode
from competition.config import SEED, WORK_DIR
from competition.evalx import log_experiment, macro_f05_pairs, truth_pairs
from resolver.collective import COLLECTIVE, collective, components  # noqa: F401 (re-exported)
from resolver.features import FEATURES

K = 5


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
