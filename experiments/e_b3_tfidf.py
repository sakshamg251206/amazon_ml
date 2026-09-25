"""E-B3: candidate recall of partitioned TF-IDF retrieval, alone and unioned with exact keys.

Evaluation sample = whole (country, state) partitions, randomly chosen until >=15% of each
country's S1 is covered. Sampling whole partitions keeps the evaluation unbiased: every sampled
S1 competes against all its real neighbours, exactly as on test. Records whose inferred state
is wrong/missing are counted as TF-IDF misses (the exact-key channel is country-wide).
"""
import random
import time

import polars as pl

from experiments.e_b1_keys import CAP, KEYS, with_keys
from src.block import tfidf_candidates
from src.config import SEED, WORK_DIR
from src.evalx import candidate_metrics, log_experiment, truth_pairs
from src.partition import infer_state, s1_state

COLS = ["entity_id", "country", "name_tok", "addr", "parts", "hn", "name_key", "name_ns"]
FRAC = 0.15
K_NAME, K_COMB = 20, 50


def sample_partitions(s1: pl.DataFrame) -> set[tuple[str, str]]:
    rng = random.Random(SEED)
    chosen = set()
    for (country,), g in s1.sort("country").group_by("country", maintain_order=True):  # deterministic
        sizes = g.group_by("state").len().drop_nulls().sort("state").rows()
        rng.shuffle(sizes)
        total, got = g.height, 0
        for state, n in sizes:
            if got >= FRAC * total:
                break
            chosen.add((country, state))
            got += n
    return chosen


def main() -> None:
    t0 = time.time()
    alias = pl.read_parquet(WORK_DIR / "alias_train.parquet")
    s1 = pl.read_parquet(WORK_DIR / "norm/train_s1.parquet", columns=COLS)
    s1 = s1.with_columns(s1_state(s1).alias("state"))
    recs = pl.concat([pl.read_parquet(WORK_DIR / f"norm/train_s{s}.parquet", columns=COLS) for s in (2, 3)])
    recs = recs.with_columns(infer_state(recs, alias).alias("state"))
    parts = sample_partitions(s1)
    in_sample = pl.DataFrame({"country": [c for c, _ in parts], "state": [s for _, s in parts]})
    s1_s = s1.join(in_sample, on=["country", "state"])
    ids = s1_s["entity_id"]
    truth = truth_pairs()
    print(f"sample: {len(parts)} partitions, {s1_s.height:,} S1", flush=True)

    # Exact-key channel (country-wide), restricted to sampled S1.
    tk = time.time()
    s1k, rk = with_keys(s1), with_keys(recs.select(COLS))  # chain cap uses ALL S1
    keys = []
    for k in KEYS:
        b = s1k.filter(pl.col(k).is_not_null()).select(pl.col("entity_id").alias("s1"), k).filter(pl.len().over(k) <= CAP)
        b = b.filter(pl.col("s1").is_in(ids.implode()))
        keys.append(rk.filter(pl.col(k).is_not_null()).select(pl.col("entity_id").alias("rec"), k).join(b, on=k).select("s1", "rec"))
    keys = pl.concat(keys).unique()
    del rk, s1k
    log_experiment("E-B3:keys_only", candidate_metrics(keys, truth, ids), tk)

    # TF-IDF channel, one partition at a time.
    tt = time.time()
    out, n_q = [], 0
    for (country, state), g in s1_s.group_by("country", "state"):
        q = recs.filter((pl.col("country") == country) & (pl.col("state") == state))
        n_q += q.height
        c = tfidf_candidates(g, q, K_NAME, K_COMB)
        out.append(c.with_columns(
            pl.Series("s1", g["entity_id"].to_numpy()[c["s"].to_numpy()]),
            pl.Series("rec", q["entity_id"].to_numpy()[c["r"].to_numpy()]),
        ).select("s1", "rec", "cos_name", "cos_addr", "cos_comb"))
    tf = pl.concat(out).with_columns(
        pl.col("cos_comb").rank("ordinal", descending=True).over("rec").alias("rank_comb"))
    tf.write_parquet(WORK_DIR / "e_b3_tfidf_cands.parquet")
    dt = time.time() - tt
    for k in (5, 10, 20, 50):
        m = candidate_metrics(tf.filter(pl.col("rank_comb") <= k), truth, ids)
        log_experiment(f"E-B3:tfidf_comb@{k}", m, tt)
    m = candidate_metrics(tf, truth, ids)
    m["ms_per_record"] = 1000 * dt / max(n_q, 1)
    log_experiment("E-B3:tfidf_all", m, tt)

    union = pl.concat([tf.select("s1", "rec"), keys]).unique()
    union.write_parquet(WORK_DIR / "e_b3_union_cands.parquet")
    log_experiment("E-B3:union", candidate_metrics(union, truth, ids), t0)
    for (country,), g in s1_s.group_by("country"):
        log_experiment(f"E-B3:union:{country}", candidate_metrics(union, truth, g["entity_id"]), t0)
    s1_s.select("entity_id", "country", "state").write_parquet(WORK_DIR / "e_b3_sample_s1.parquet")


if __name__ == "__main__":
    main()
