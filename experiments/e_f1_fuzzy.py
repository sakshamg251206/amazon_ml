"""E-F1: fuzzy matching WITHOUT learning, on the same candidates as E-M1.

Hypothesis: most of the gain over v0_rule comes from candidate generation; a hand-set fuzzy score
(no training) already recovers much of it. Isolates candidate-set gain vs learned-matcher gain.
Decision: each record -> its best S1 by the score, kept if score >= t (t swept).
"""
import time

import polars as pl

from experiments.e_m1_lgbm import decode
from competition.config import WORK_DIR
from competition.evalx import log_experiment, macro_f05_pairs, truth_pairs

SCORES = {
    "cos_comb": pl.col("cos_comb").fill_null(0.0),
    "fz_tokenset": pl.col("fz") / 100,
    "fz_plus_number": (pl.col("fz") / 100) * 0.8 + 0.2 * pl.col("hn_eq"),
}


def main() -> None:
    t0 = time.time()
    feats = pl.read_parquet(WORK_DIR / "e_m1_feats_k5" / "*.parquet", columns=["s1", "rec", "cos_comb", "fz", "hn_eq"])
    sample = pl.read_parquet(WORK_DIR / "e_b3_sample_s1.parquet")
    ids, truth = sample["entity_id"], truth_pairs()
    for name, expr in SCORES.items():
        scored = feats.select("s1", "rec", expr.alias("p"))
        best = max(((t, macro_f05_pairs(decode(scored, t), truth, ids)) for t in
                    (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)), key=lambda x: x[1]["f05"])
        log_experiment(f"E-F1:{name}", {**best[1], "t": best[0]}, t0)


if __name__ == "__main__":
    main()
