"""E-B2: label-free state inference. Coverage (share of records with a state) on train and test,
accuracy on train true pairs (record state == S1 state), and the state-confusion pairs that
should be merged into one blocking partition (e.g. Telangana/Andhra Pradesh)."""
import time

import polars as pl

from competition.config import WORK_DIR
from competition.evalx import log_experiment, truth_pairs
from resolver.partition import infer_state, learn_aliases, s1_state

COLS = ["entity_id", "country", "parts"]


def run(split: str) -> None:
    t0 = time.time()
    s1 = pl.read_parquet(WORK_DIR / f"norm/{split}_s1.parquet", columns=COLS)
    recs = pl.concat([pl.read_parquet(WORK_DIR / f"norm/{split}_s{s}.parquet", columns=COLS) for s in (2, 3)])
    alias = learn_aliases(s1, recs)
    alias.write_parquet(WORK_DIR / f"alias_{split}.parquet")
    s1 = s1.with_columns(s1_state(s1).alias("state"))
    recs = recs.with_columns(infer_state(recs, alias).alias("state"))
    for (c,), g in recs.group_by("country"):
        log_experiment(f"E-B2:{split}:coverage:{c}", {"coverage": g["state"].is_not_null().mean(),
                                                     "n_alias": alias.filter(pl.col("country") == c).height}, t0)
    if split != "train":
        return
    tp = truth_pairs().drop_nulls().join(s1.select(pl.col("entity_id").alias("s1"), pl.col("state").alias("s1_state"), "country"), on="s1") \
        .join(recs.select(pl.col("entity_id").alias("rec"), pl.col("state").alias("r_state")), on="rec")
    for (c,), g in tp.group_by("country"):
        known = g.filter(pl.col("r_state").is_not_null())
        log_experiment(f"E-B2:train:accuracy:{c}", {"acc_when_known": (known["r_state"] == known["s1_state"]).mean(),
                                                   "pairs_same_partition": (g["r_state"] == g["s1_state"]).fill_null(False).mean()}, t0)
    conf = (tp.filter(pl.col("r_state").is_not_null() & (pl.col("r_state") != pl.col("s1_state")))
              .group_by("country", "s1_state", "r_state").len("n").sort("n", descending=True))
    print(conf.head(15))


if __name__ == "__main__":
    run("train")
    run("test")
