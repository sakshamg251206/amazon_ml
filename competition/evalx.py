"""Vectorised evaluation for experiments: pair precision/recall, macro F0.5, candidate recall,
oracle F0.5, plus a logger that records runtime and peak memory.

Pairs are polars frames with columns (s1, rec). `s1_ids` restricts evaluation to a subset of
S1 (e.g. a validation sample); predictions on other S1 are ignored.
"""
import csv
import resource
import sys
import time
from datetime import datetime

import polars as pl

from competition import config
from competition.data import read_ground_truth

LOG = config.ROOT / "experiments" / "log.csv"


def truth_pairs() -> pl.DataFrame:
    """All true (s1, rec) pairs plus every S1 id (singletons carry rec = null)."""
    gt = read_ground_truth()
    s1s, recs = [], []
    for s1, ms in gt.items():
        if ms:
            for m in ms:
                s1s.append(s1)
                recs.append(m)
        else:
            s1s.append(s1)
            recs.append(None)
    return pl.DataFrame({"s1": s1s, "rec": recs}, schema={"s1": pl.String, "rec": pl.String})


def macro_f05_pairs(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series) -> dict[str, float]:
    """Macro F0.5 over s1_ids, plus pair-level precision and recall on the same S1 subset."""
    ids = pl.DataFrame({"s1": s1_ids.unique()})
    t = truth.join(ids, on="s1").filter(pl.col("rec").is_not_null())
    p = pred.select("s1", "rec").unique().join(ids, on="s1")
    tp = p.join(t, on=["s1", "rec"])
    per = (ids.join(t.group_by("s1").len("n_true"), on="s1", how="left")
              .join(p.group_by("s1").len("n_pred"), on="s1", how="left")
              .join(tp.group_by("s1").len("tp"), on="s1", how="left")
              .fill_null(0))
    f = pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64)).otherwise(
        1.25 * pl.col("tp") / (pl.col("n_pred") + 0.25 * pl.col("n_true")))
    return {
        "f05": per.select(f.mean()).item(),
        "precision": tp.height / max(p.height, 1),
        "recall": tp.height / max(t.height, 1),
        "n_s1": ids.height,
    }


def candidate_metrics(cand: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series) -> dict[str, float]:
    """Candidate recall (share of true pairs present) and oracle F0.5 (perfect matcher)."""
    ids = pl.DataFrame({"s1": s1_ids.unique()})
    c = cand.select("s1", "rec").unique().join(ids, on="s1")
    t = truth.join(ids, on="s1").filter(pl.col("rec").is_not_null())
    hit = c.join(t, on=["s1", "rec"])
    oracle = macro_f05_pairs(hit, truth, s1_ids)["f05"]
    return {"cand_recall": hit.height / max(t.height, 1), "oracle_f05": oracle,
            "cand_per_s1": c.height / ids.height}


def peak_mem_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3  # macOS reports bytes, Linux KiB


def log_experiment(name: str, metrics: dict[str, float], t0: float) -> None:
    metrics = {**metrics, "runtime_s": time.time() - t0, "peak_mem_mb": peak_mem_mb()}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "version", "seed", "metric", "value"])
        now = datetime.now().isoformat(timespec="seconds")
        for k, v in metrics.items():
            w.writerow([now, name, config.SEED, k, f"{v:.5f}"])
    print(name, {k: round(v, 4) for k, v in metrics.items()}, flush=True)
