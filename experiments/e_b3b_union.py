"""E-B3b: finish E-B3 cheaply from its saved TF-IDF candidates: union(top-K TF-IDF, exact keys),
per-country recall, and the sample/union files E-M1 consumes. Streams only what it needs."""
import sys
import time

import polars as pl

from experiments.e_b1_keys import CAP, KEYS, with_keys
from experiments.e_b3_tfidf import COLS, sample_partitions
from competition.config import WORK_DIR
from competition.evalx import candidate_metrics, log_experiment, truth_pairs
from resolver.partition import s1_state

K = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def main() -> None:
    t0 = time.time()
    s1 = pl.read_parquet(WORK_DIR / "norm/train_s1.parquet", columns=COLS)
    s1 = s1.with_columns(s1_state(s1).alias("state"))
    parts = sample_partitions(s1)
    in_sample = pl.DataFrame({"country": [c for c, _ in parts], "state": [s for _, s in parts]})
    s1_s = s1.join(in_sample, on=["country", "state"])
    ids = s1_s["entity_id"]
    s1_s.select("entity_id", "country", "state").write_parquet(WORK_DIR / "e_b3_sample_s1.parquet")

    s1k = with_keys(s1)
    recs = pl.concat([pl.read_parquet(WORK_DIR / f"norm/train_s{s}.parquet", columns=COLS) for s in (2, 3)])
    rk = with_keys(recs)
    del recs
    keys = []
    for k in KEYS:
        b = s1k.filter(pl.col(k).is_not_null()).select(pl.col("entity_id").alias("s1"), k).filter(pl.len().over(k) <= CAP)
        b = b.filter(pl.col("s1").is_in(ids.implode()))
        keys.append(rk.filter(pl.col(k).is_not_null()).select(pl.col("entity_id").alias("rec"), k).join(b, on=k).select("s1", "rec"))
    keys = pl.concat(keys).unique()
    del rk, s1k

    tf = pl.scan_parquet(WORK_DIR / "e_b3_tfidf_cands.parquet").filter(pl.col("rank_comb") <= K).collect()
    tf.write_parquet(WORK_DIR / f"e_b3_tfidf_top{K}.parquet")
    union = pl.concat([tf.select("s1", "rec"), keys]).unique()
    union.write_parquet(WORK_DIR / "e_b3_union_cands.parquet")
    truth = truth_pairs()
    log_experiment(f"E-B3:union_top{K}+keys", candidate_metrics(union, truth, ids), t0)
    for (country,), g in s1_s.group_by("country"):
        log_experiment(f"E-B3:union_top{K}+keys:{country}", candidate_metrics(union, truth, g["entity_id"]), t0)


if __name__ == "__main__":
    main()
