"""E-D1: decoding-only improvements on the saved out-of-fold predictions (no retraining).

Compares, on the 414k-S1 validation sample (M1 and M2 OOF probabilities):
  thr       : each record -> its best S1, kept if p >= t (current rule), fine sweep of t
  thr_ctry  : same, one t per country
  ef        : exclusive assignment, then per S1 keep the top-k records that maximise EXPECTED F0.5
              under the model's probabilities; predicting nothing scores P(no true match).
Metric identical to competition.evalx.macro_f05_pairs (checked against it for the current rule).
"""
import sys
import time

import numpy as np
import polars as pl

from competition.config import WORK_DIR
from competition.evalx import log_experiment

NTRUE = WORK_DIR / "val_ntrue.parquet"


def ntrue_table() -> pl.DataFrame:
    """Per sample S1: number of true matches (0 = singleton). Cached (truth_pairs is large)."""
    if not NTRUE.exists():
        from competition.evalx import truth_pairs
        ids = pl.read_parquet(WORK_DIR / "e_b3_sample_s1.parquet").select(pl.col("entity_id").alias("s1"), "country")
        t = truth_pairs().join(ids, on="s1")
        t.group_by("s1", "country").agg(pl.col("rec").is_not_null().sum().alias("n_true")) \
         .select(pl.col("s1").hash().alias("s1"), "country", "n_true").write_parquet(NTRUE)
    return pl.read_parquet(NTRUE)


def load_oof(name: str) -> pl.DataFrame:
    return pl.read_parquet(WORK_DIR / f"{name}.parquet").select(
        pl.col("s1").hash(), pl.col("rec").hash(), "y", pl.col("p").cast(pl.Float64))


def exclusive(scored: pl.DataFrame) -> pl.DataFrame:
    return scored.sort("p", descending=True).group_by("rec", maintain_order=True).first()


def f05(pred: pl.DataFrame, nt: pl.DataFrame, by_country: bool = False) -> dict[str, float]:
    per = pred.group_by("s1").agg(pl.len().alias("n_pred"), pl.col("y").sum().alias("tp"))
    per = nt.join(per, on="s1", how="left").fill_null(0)
    f = pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64)) \
          .otherwise(1.25 * pl.col("tp") / (pl.col("n_pred") + 0.25 * pl.col("n_true")))
    out = {"f05": per.select(f.mean()).item(),
           "precision": pred["y"].sum() / max(pred.height, 1),
           "recall": pred["y"].sum() / max(nt["n_true"].sum(), 1)}
    if by_country:
        for (c,), g in per.with_columns(f.alias("f")).group_by("country"):
            out[f"f05_{c}"] = g["f"].mean()
    return out


def ef_decode(ex: pl.DataFrame, c: float = 0.0, floor: float = 0.0) -> pl.DataFrame:
    """Per S1 keep the top-k (by p) maximising E[F0.5] ~ 1.25*sum_topk(p) / (k + 0.25*(sum_all(p) + c)).
    k = 0 is chosen when P(no true match) = prod(1-p) beats every k >= 1. c = expected true matches
    the candidates missed (candidate recall < 1)."""
    d = ex.filter(pl.col("p") >= floor).sort(["s1", "p"], descending=[False, True]).with_columns(
        pl.int_range(1, pl.len() + 1).over("s1").alias("k"),
        pl.col("p").cum_sum().over("s1").alias("P"),
        pl.col("p").sum().over("s1").alias("E"),
    ).with_columns(
        (1 - pl.col("p").clip(0, 1 - 1e-9)).log().sum().over("s1").exp().alias("EF0"),
    ).with_columns((1.25 * pl.col("P") / (pl.col("k") + 0.25 * (pl.col("E") + c))).alias("EF"))
    d = d.with_columns(pl.col("EF").max().over("s1").alias("EFmax"),
                       pl.col("k").filter(pl.col("EF") == pl.col("EF").max()).first().over("s1").alias("kstar"))
    return d.filter((pl.col("k") <= pl.col("kstar")) & (pl.col("EFmax") > pl.col("EF0"))).select("s1", "rec", "y", "p")


def run(name: str) -> None:
    t0 = time.time()
    nt = ntrue_table()
    ex = exclusive(load_oof(name))
    ctry = ex.join(nt.select("s1", "country"), on="s1", how="left")
    res = {}
    for t in np.arange(0.50, 0.905, 0.02):
        res[round(float(t), 2)] = f05(ex.filter(pl.col("p") >= t), nt)["f05"]
    t_best = max(res, key=res.get)
    print(f"[{name}] thr sweep:", {k: round(v, 4) for k, v in res.items()}, flush=True)
    log_experiment(f"E-D1:{name}:thr", {**f05(ex.filter(pl.col("p") >= t_best), nt, True), "t": t_best}, t0)
    # per-country threshold
    tc = {}
    for c in ("US", "India"):
        sub_nt = nt.filter(pl.col("country") == c)
        sub = ctry.filter(pl.col("country") == c)
        tc[c] = max(np.arange(0.50, 0.905, 0.02), key=lambda t: f05(sub.filter(pl.col("p") >= t), sub_nt)["f05"])
    pred = ctry.filter(pl.col("p") >= pl.col("country").replace_strict(tc, default=t_best))
    log_experiment(f"E-D1:{name}:thr_ctry", {**f05(pred.drop("country"), nt, True),
                                             **{f"t_{k}": round(float(v), 2) for k, v in tc.items()}}, t0)
    # expected-F decoding
    for c in (0.0, 0.05, 0.15):
        for floor in (0.0, 0.05, 0.2):
            log_experiment(f"E-D1:{name}:ef_c{c}_floor{floor}", f05(ef_decode(ex, c, floor), nt, True), t0)


if __name__ == "__main__":
    for n in sys.argv[1:] or ["e_m1_oof_k5"]:
        run(n)
