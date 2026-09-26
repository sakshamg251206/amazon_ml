"""E-B1: candidate recall of exact blocking keys (and their union) on the FULL train set.

Hypothesis: a union of cheap exact keys reaches far higher candidate recall than v0's single
(name_key, house_no) key. Keys whose S1 block exceeds CAP entities are skipped (chains).
"""
import time

import polars as pl

from src.block import CAP, KEYS, with_keys
from src.config import WORK_DIR
from src.evalx import candidate_metrics, log_experiment, truth_pairs

def main() -> None:
    t0 = time.time()
    s1 = with_keys(pl.read_parquet(WORK_DIR / "norm/train_s1.parquet"))
    recs = pl.concat([with_keys(pl.read_parquet(WORK_DIR / f"norm/train_s{s}.parquet")) for s in (2, 3)])
    truth = truth_pairs()
    ids = s1["entity_id"]
    n_rec = recs.height
    union = []
    for k in KEYS:
        tk = time.time()
        blocks = s1.filter(pl.col(k).is_not_null()).select(pl.col("entity_id").alias("s1"), k)
        blocks = blocks.filter(pl.len().over(k) <= CAP)
        cand = recs.filter(pl.col(k).is_not_null()).select(pl.col("entity_id").alias("rec"), k).join(blocks, on=k).select("s1", "rec")
        union.append(cand)
        m = candidate_metrics(cand, truth, ids)
        m["pairs_per_rec"] = cand.height / n_rec
        log_experiment(f"E-B1:{k}", m, tk)
    cand = pl.concat(union).unique()
    m = candidate_metrics(cand, truth, ids)
    m["pairs_per_rec"] = cand.height / n_rec
    log_experiment("E-B1:union", m, t0)
    # per-country union recall
    ctry = s1.select(pl.col("entity_id").alias("s1"), "country")
    for (country,), g in ctry.group_by("country"):
        log_experiment(f"E-B1:union:{country}", candidate_metrics(cand, truth, g["s1"]), t0)


if __name__ == "__main__":
    main()
