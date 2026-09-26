"""E-B4: benchmark the scipy chunked top-k (replacement for sparse_dot_topn) before running at scale.

1. Equivalence + recall on 3 train partitions from the E-B3 sample: compare new top-5 candidates
   with the saved sparse_dot_topn ones (work/e_b3_tfidf_top5.parquet): pair overlap, candidate
   recall of each, runtime, peak memory.
2. Stability: the test partition that crashed sparse_dot_topn (France / hauts de france).
"""
import sys
import time

import polars as pl

from src.block import tfidf_candidates
from src.config import WORK_DIR
from src.evalx import candidate_metrics, log_experiment, peak_mem_mb, truth_pairs
from src.partition import infer_state, learn_aliases, s1_state

COLS = ["entity_id", "country", "name_tok", "addr", "parts"]
K_NAME, K_COMB, K_ADDR, K_KEEP = 10, 20, 10, 5


def run_partition(g: pl.DataFrame, q: pl.DataFrame) -> pl.DataFrame:
    c = tfidf_candidates(g, q, K_NAME, K_COMB, K_ADDR)
    c = c.with_columns(pl.col("cos_comb").rank("ordinal", descending=True).over("r").alias("rank_comb")) \
         .filter(pl.col("rank_comb") <= K_KEEP)
    return c.with_columns(pl.Series("s1", g["entity_id"].to_numpy()[c["s"].to_numpy()], dtype=pl.String),
                          pl.Series("rec", q["entity_id"].to_numpy()[c["r"].to_numpy()], dtype=pl.String)).select("s1", "rec")


def train_sample() -> None:
    sample = pl.read_parquet(WORK_DIR / "e_b3_sample_s1.parquet")
    alias = pl.read_parquet(WORK_DIR / "alias_train.parquet")
    old = pl.read_parquet(WORK_DIR / "e_b3_tfidf_top5.parquet", columns=["s1", "rec"])
    truth = truth_pairs()
    sizes = sample.group_by("country", "state").len().sort("len")
    picks = [sizes.row(i) for i in (len(sizes) // 3, len(sizes) // 2, 2 * len(sizes) // 3)]
    s1_all = pl.read_parquet(WORK_DIR / "norm/train_s1.parquet", columns=COLS)
    s1_all = s1_all.with_columns(s1_state(s1_all).alias("state"))
    recs_all = pl.concat([pl.read_parquet(WORK_DIR / f"norm/train_s{s}.parquet", columns=COLS) for s in (2, 3)])
    recs_all = recs_all.with_columns(infer_state(recs_all, alias).alias("state"))
    for country, state, n in picks:
        t0 = time.time()
        g = s1_all.filter((pl.col("country") == country) & (pl.col("state") == state))
        q = recs_all.filter((pl.col("country") == country) & (pl.col("state") == state))
        new = run_partition(g, q)
        o = old.join(g.select(pl.col("entity_id").alias("s1")), on="s1")
        overlap = new.join(o, on=["s1", "rec"]).height / max(o.height, 1)
        ids = g["entity_id"]
        log_experiment(f"E-B4:df{__import__('src.block').block.MAX_DF}:{country}/{state}", {
            "n_s1": n, "n_rec": q.height, "pair_overlap_with_old": overlap,
            "recall_new": candidate_metrics(new, truth, ids)["cand_recall"],
            "recall_old": candidate_metrics(o, truth, ids)["cand_recall"]}, t0)


def france_crash_partition() -> None:
    t0 = time.time()
    s1 = pl.scan_parquet(WORK_DIR / "norm/test_s1.parquet").select(COLS).filter(pl.col("country") == "France").collect()
    recs = pl.concat([pl.scan_parquet(WORK_DIR / f"norm/test_s{s}.parquet").select(COLS)
                      .filter(pl.col("country") == "France").collect() for s in (2, 3)])
    alias = learn_aliases(s1, recs)
    s1 = s1.with_columns(s1_state(s1).alias("state"))
    recs = recs.with_columns(infer_state(recs, alias).alias("state"))
    g = s1.filter(pl.col("state") == "hauts de france")
    q = recs.filter(pl.col("state") == "hauts de france")
    t1 = time.time()
    c = run_partition(g, q)
    log_experiment("E-B4:France/hauts de france", {"n_s1": g.height, "n_rec": q.height, "pairs": c.height,
                                                   "topk_s": time.time() - t1, "peak_mem_mb": peak_mem_mb()}, t0)


if __name__ == "__main__":
    import src.block as B
    if len(sys.argv) > 2:
        B.MAX_DF = float(sys.argv[2])
    print(f"MAX_DF={B.MAX_DF}", flush=True)
    {"train": train_sample, "france": france_crash_partition}[sys.argv[1]]()
