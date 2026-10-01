"""Evaluation: the competition metric (macro F0.5 over every S1, singletons included) plus the
diagnostics used throughout the project: pair precision/recall, blocking recall, oracle F0.5,
singleton accuracy, and per-slice breakdowns. Pair frames are polars (s1, rec)."""
import numpy as np
import polars as pl

from resolver.decode import ef_decode, exclusive


def per_s1(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series) -> pl.DataFrame:
    """(s1, n_true, n_pred, tp, f05) for every S1 in s1_ids. truth: (s1, rec) true pairs only."""
    ids = pl.DataFrame({"s1": s1_ids.unique()})
    t = truth.select("s1", "rec").join(ids, on="s1")
    p = pred.select("s1", "rec").unique().join(ids, on="s1")
    tp = p.join(t, on=["s1", "rec"])
    d = (ids.join(t.group_by("s1").len("n_true"), on="s1", how="left")
            .join(p.group_by("s1").len("n_pred"), on="s1", how="left")
            .join(tp.group_by("s1").len("tp"), on="s1", how="left")
            .with_columns(pl.col("n_true", "n_pred", "tp").fill_null(0)))
    return d.with_columns(
        pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64))
          .otherwise(1.25 * pl.col("tp") / (pl.col("n_pred") + 0.25 * pl.col("n_true"))).alias("f05"))


def summary(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series) -> dict[str, float]:
    d = per_s1(pred, truth, s1_ids)
    tp, n_pred, n_true = d["tp"].sum(), d["n_pred"].sum(), d["n_true"].sum()
    single = d.filter(pl.col("n_true") == 0)
    f1 = d.select(pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64))
                  .otherwise(2 * pl.col("tp") / (pl.col("n_pred") + pl.col("n_true"))).mean()).item()
    return {
        "f05": float(d["f05"].mean()) if d.height else 0.0,
        "f1": float(f1 or 0.0),
        "precision": float(tp / n_pred) if n_pred else 0.0,
        "recall": float(tp / n_true) if n_true else 0.0,
        "n_s1": d.height,
        "n_pred_pairs": int(n_pred),
        "n_true_pairs": int(n_true),
        "singleton_share": single.height / max(d.height, 1),
        "singleton_accuracy": float((single["n_pred"] == 0).mean()) if single.height else 1.0,
        "exact_s1_share": float(((d["tp"] == d["n_true"]) & (d["n_pred"] == d["n_true"])).mean()) if d.height else 0.0,
    }


def candidate_metrics(cands: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series, n_recs: int) -> dict[str, float]:
    """Blocking quality: recall of true pairs, oracle F0.5 (a perfect matcher on these candidates),
    candidates per S1 and the reduction ratio vs the full S1 x record cross product."""
    ids = pl.DataFrame({"s1": s1_ids.unique()})
    c = cands.select("s1", "rec").unique().join(ids, on="s1")
    t = truth.select("s1", "rec").join(ids, on="s1")
    hit = c.join(t, on=["s1", "rec"])
    return {
        "cand_recall": hit.height / max(t.height, 1),
        "oracle_f05": summary(hit, truth, s1_ids)["f05"],
        "cand_pairs": c.height,
        "cand_per_s1": c.height / max(ids.height, 1),
        "reduction_ratio": 1 - c.height / max(ids.height * n_recs, 1),
    }


def decode_ef(scored: pl.DataFrame) -> pl.DataFrame:
    """(s1, rec, p) -> matches after exclusive assignment + expected-F0.5 decoding."""
    return ef_decode(exclusive(scored.select("s1", "rec", "p")))


def decode_threshold(scored: pl.DataFrame, t: float) -> pl.DataFrame:
    return exclusive(scored.select("s1", "rec", "p")).filter(pl.col("p") >= t)


def threshold_curve(scored: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series,
                    grid=tuple(np.round(np.arange(0.05, 1.0, 0.05), 2))) -> list[dict]:
    ex = exclusive(scored.select("s1", "rec", "p"))
    out = []
    for t in grid:
        m = summary(ex.filter(pl.col("p") >= t), truth, s1_ids)
        out.append({"t": float(t), "f05": m["f05"], "precision": m["precision"], "recall": m["recall"]})
    return out


def best_threshold(scored: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series) -> tuple[float, dict]:
    curve = threshold_curve(scored, truth, s1_ids, grid=tuple(np.round(np.arange(0.05, 1.0, 0.025), 3)))
    best = max(curve, key=lambda r: r["f05"])
    return best["t"], summary(decode_threshold(scored, best["t"]), truth, s1_ids)
