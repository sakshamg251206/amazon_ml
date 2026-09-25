"""E-L1: do train and test share an underlying pool of businesses?

Hypothesis: distractor records in one split are copies of S1 businesses of the OTHER split.
If so, train S1 is an extra reference that exposes test hard distractors (a dataset-construction
trait, which would explain public scores near 0.998).

Test (exact keys only, no model):
  * train distractors vs train matched records: how often does each hit a TEST S1 exactly?
    A much higher rate for distractors => distractors are copies of other-split entities.
  * test records: how often do they hit a TRAIN S1 exactly (vs hitting a test S1)?
  * test S1 vs train S1 exact duplicates.
"""
import time

import polars as pl

from src.config import WORK_DIR
from src.evalx import log_experiment, truth_pairs

COLS = ["entity_id", "country", "name_key", "hn", "name_ns", "addr"]


def keyed(split: str, src: int) -> pl.DataFrame:
    d = pl.read_parquet(WORK_DIR / f"norm/{split}_s{src}.parquet", columns=COLS)
    return d.select(
        "entity_id",
        pl.when((pl.col("hn") != "") & (pl.col("name_key") != ""))
          .then(pl.concat_str(["country", "name_key", "hn"], separator="|")).alias("k_nh"),
        pl.when(pl.col("addr") != "").then(pl.concat_str(["country", "name_ns", "addr"], separator="|")).alias("k_full"),
    )


def hit_rate(recs: pl.DataFrame, s1: pl.DataFrame, key: str) -> float:
    keys = s1.select(key).drop_nulls().unique()
    r = recs.filter(pl.col(key).is_not_null())
    return r.join(keys, on=key, how="semi").height / max(r.height, 1)


def main() -> None:
    t0 = time.time()
    tr1, te1 = keyed("train", 1), keyed("test", 1)
    tr = pl.concat([keyed("train", s) for s in (2, 3)])
    owned = truth_pairs().drop_nulls().select(pl.col("rec").alias("entity_id"))
    tr_m = tr.join(owned, on="entity_id", how="semi")
    tr_d = tr.join(owned, on="entity_id", how="anti")
    m = {}
    for key in ("k_nh", "k_full"):
        m[f"{key}:train_matched->train_s1"] = hit_rate(tr_m, tr1, key)
        m[f"{key}:train_distractor->train_s1"] = hit_rate(tr_d, tr1, key)
        m[f"{key}:train_matched->test_s1"] = hit_rate(tr_m, te1, key)
        m[f"{key}:train_distractor->test_s1"] = hit_rate(tr_d, te1, key)
    del tr, tr_m, tr_d
    te = pl.concat([keyed("test", s) for s in (2, 3)])
    for key in ("k_nh", "k_full"):
        m[f"{key}:test_rec->test_s1"] = hit_rate(te, te1, key)
        m[f"{key}:test_rec->train_s1"] = hit_rate(te, tr1, key)
        m[f"{key}:test_s1_dup_in_train_s1"] = hit_rate(te1, tr1, key)
    log_experiment("E-L1:overlap", m, t0)


if __name__ == "__main__":
    main()
