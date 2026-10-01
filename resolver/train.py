"""Training with honest, out-of-fold evaluation.

Validation follows the project's protocol (D2): S1 entities are grouped by their inferred
(country, state) partition and whole partitions are assigned to folds, so every S1 is scored by
models that never saw its state, at full candidate density. Stage 2 is stacked on OUT-OF-FOLD
stage-1 probabilities with the same folds, so no pair's collective features come from a model
that saw its own label. Final models are then refitted on everything.

The report holds the improvement ladder (rule -> fuzzy -> stage 1 -> stage 2), blocking quality,
per-country / per-source / script slices, a threshold curve and feature importance.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import polars as pl

from resolver import engine as E
from resolver.baseline import rule_pairs
from resolver.evaluate import (best_threshold, candidate_metrics, decode_ef, decode_threshold, summary,
                               threshold_curve)
from resolver.explain import FEATURE_INFO

log = logging.getLogger("resolver")


def assign_folds(s1_part: pl.DataFrame, k: int, seed: int = E.SEED) -> pl.DataFrame:
    """(entity_id, country, state) -> (s1, fold). Whole partitions per fold; S1 without a state are
    spread by id hash."""
    parts = s1_part.select("country", "state").drop_nulls().unique().sort("country", "state")
    perm = np.random.default_rng(seed).permutation(parts.height) % k
    pf = parts.with_columns(pl.Series("fold", perm, dtype=pl.Int8))
    return (s1_part.join(pf, on=["country", "state"], how="left")
                   .with_columns(pl.col("fold").fill_null((pl.col("entity_id").hash(seed) % k).cast(pl.Int8)))
                   .select(pl.col("entity_id").alias("s1"), "fold"))


def _slice_metrics(pred: pl.DataFrame, truth: pl.DataFrame, s1: pl.DataFrame, recs: pl.DataFrame) -> dict:
    """Per-country macro F0.5, and pair precision/recall by record source and script."""
    out = {"by_country": []}
    for (country,), g in s1.group_by("country"):
        m = summary(pred, truth, g["entity_id"])
        out["by_country"].append({"country": country, **{k: m[k] for k in ("f05", "precision", "recall", "n_s1")}})
    out["by_country"].sort(key=lambda r: r["country"])
    info = recs.select(pl.col("entity_id").alias("rec"), "source", "nonlatin")
    p, t = pred.select("s1", "rec").join(info, on="rec"), truth.join(info, on="rec")
    tp = p.join(t.select("s1", "rec"), on=["s1", "rec"])
    rows = []
    for label, cond in (("Source 2", pl.col("source") == 2), ("Source 3", pl.col("source") == 3),
                        ("Latin script", ~pl.col("nonlatin")), ("Non-Latin script", pl.col("nonlatin"))):
        n_p, n_t, n_tp = p.filter(cond).height, t.filter(cond).height, tp.filter(cond).height
        rows.append({"slice": label, "precision": n_tp / max(n_p, 1), "recall": n_tp / max(n_t, 1), "true_pairs": n_t})
    out["by_record"] = rows
    return out


def fit(ds, k_folds: int = 2, workers: int = 1) -> tuple[E.Matcher, dict]:
    if ds.truth is None:
        raise ValueError("training needs a ground-truth file")
    t_start = time.time()
    prep = E.prepare(ds, workers=workers)
    truth, s1_ids = ds.truth, ds.s1["entity_id"]
    feats = prep.feats.join(truth.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1", "rec"], how="left") \
                      .with_columns(pl.col("y").fill_null(0)) \
                      .join(assign_folds(prep.blocked.s1_part, k_folds), on="s1", how="left") \
                      .with_columns(pl.col("fold").fill_null(0))
    y, fold = feats["y"].to_numpy(), feats["fold"].to_numpy()
    X1 = E.to_matrix(feats, E.STAGE1)
    timings = dict(prep.timings)

    # ---- stage 1: out-of-fold, then final
    t0 = time.time()
    p1 = np.zeros(len(y))
    for f in range(k_folds):
        tr, va = fold != f, fold == f
        if va.any():
            p1[va] = E.train_stage1(X1[tr], y[tr]).predict(X1[va], num_threads=E.THREADS)
    stage1 = E.train_stage1(X1, y)
    timings["stage1_s"] = time.time() - t0
    log.info("stage 1 trained on %d pairs (%.1fs)", len(y), timings["stage1_s"])

    # ---- stage 2 on OOF p1 >= PRUNE, same folds
    t0 = time.time()
    keep = p1 >= E.PRUNE
    kept = feats.filter(pl.Series(keep)).with_columns(pl.Series("p1", p1[keep], dtype=pl.Float32))
    col = E.collective(kept.select("s1", "rec", pl.col("p1").alias("p")), prep.comp)
    d2 = kept.drop("p1").join(col, on=["s1", "rec"])
    X2, y2, f2 = E.to_matrix(d2, E.STAGE2), d2["y"].to_numpy(), d2["fold"].to_numpy()
    p2_members = {}
    for name in E.stage2_members(len(y2)):
        p = np.zeros(len(y2))
        for f in range(k_folds):
            tr, va = f2 != f, f2 == f
            if va.any() and tr.any() and len(np.unique(y2[tr])) == 2:
                p[va] = E.stage2_members(len(y2))[name].fit(X2[tr], y2[tr]).predict_proba(X2[va])[:, 1]
        p2_members[name] = p
    p2 = np.mean(list(p2_members.values()), axis=0)
    stage2 = {name: m.fit(X2, y2) for name, m in E.stage2_members(len(y2)).items()}
    timings["stage2_s"] = time.time() - t0
    log.info("stage 2 trained on %d pairs (%.1fs)", len(y2), timings["stage2_s"])

    # ---- out-of-fold evaluation
    t0 = time.time()
    oof1 = feats.select("s1", "rec").with_columns(pl.Series("p", p1))
    oof2 = d2.select("s1", "rec").with_columns(pl.Series("p", p2))
    final_pred = decode_ef(oof2)
    fuzzy = feats.select("s1", "rec", (pl.col("fz") / 100).alias("p"))
    t_fz, m_fz = best_threshold(fuzzy, truth, s1_ids)
    rule = rule_pairs(ds.s1, ds.recs)
    ladder = [
        {"step": "Exact-key rule", "detail": "name key + house number, no learning (v0)", **_ms(summary(rule, truth, s1_ids))},
        {"step": "Fuzzy score", "detail": f"0.6·name + 0.4·address token-set, tuned t={t_fz:.2f}", **_ms(m_fz)},
        {"step": "Stage 1 @ 0.5", "detail": "LightGBM, 54 features, each record to its best S1", **_ms(summary(decode_threshold(oof1, 0.5), truth, s1_ids))},
        {"step": "Stage 1 + expected-F", "detail": "expected-F0.5 decoding per S1", **_ms(summary(decode_ef(oof1), truth, s1_ids))},
    ]
    for name, p in p2_members.items():
        ladder.append({"step": f"Stage 2: {name}", "detail": "+10 collective / twin-graph features",
                       **_ms(summary(decode_ef(d2.select("s1", "rec").with_columns(pl.Series("p", p))), truth, s1_ids))})
    final = summary(final_pred, truth, s1_ids)
    ladder.append({"step": "Stage 2 ensemble", "detail": "mean of LightGBM + ExtraTrees + MLP, expected-F", **_ms(final)})
    folds = []
    fold_ids = assign_folds(prep.blocked.s1_part, k_folds)
    for f in range(k_folds):
        ids = fold_ids.filter(pl.col("fold") == f)["s1"]
        folds.append({"fold": f, "f05": summary(final_pred, truth, ids)["f05"], "n_s1": ids.len()})
    gain = stage1.feature_importance("gain")
    importance = sorted(({"feature": n, "gain": float(g / gain.sum()), **FEATURE_INFO.get(n, {})}
                         for n, g in zip(E.STAGE1, gain)), key=lambda r: -r["gain"])
    blocking = candidate_metrics(prep.blocked.pairs, truth, s1_ids, ds.recs.height)
    key_only = prep.blocked.pairs.filter(pl.col("in_keys"))
    blocking["tfidf_recall"] = candidate_metrics(prep.blocked.pairs.filter(~pl.col("in_keys")), truth, s1_ids, 1)["cand_recall"]
    blocking["key_only_pairs"] = key_only.height
    timings["evaluate_s"] = time.time() - t0
    report = {
        "dataset": ds.stats(),
        "protocol": {"folds": k_folds, "grouping": "(country, state) partitions",
                     "partitions": prep.blocked.s1_part.select("country", "state").drop_nulls().unique().height},
        "blocking": blocking,
        "final": final,
        "ladder": ladder,
        "folds": folds,
        "slices": _slice_metrics(final_pred, truth, ds.s1, prep.recn),
        "threshold_curve": threshold_curve(oof2, truth, s1_ids),
        "importance": importance,
        "pairs": {"stage1": int(len(y)), "stage2": int(len(y2)), "positives": int(y.sum())},
        "timings": {k: round(v, 2) for k, v in timings.items()},
        "total_s": round(time.time() - t_start, 1),
    }
    matcher = E.Matcher(stage1=stage1, stage2=stage2,
                        meta={"features_stage1": E.STAGE1, "features_stage2": E.STAGE2, "trained_on": ds.stats(),
                              "oof_f05": final["f05"], "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    log.info("OOF macro F0.5 %.4f (P %.4f R %.4f), blocking recall %.4f", final["f05"], final["precision"],
             final["recall"], blocking["cand_recall"])
    return matcher, report


def _ms(m: dict) -> dict:
    return {k: m[k] for k in ("f05", "precision", "recall")}


def evaluate(ds, matcher: E.Matcher, workers: int = 1) -> tuple[E.Resolution, dict]:
    """Resolve a labelled dataset with a fitted matcher (e.g. the held-out test split)."""
    res = E.resolve(ds, matcher, workers=workers)
    report = {"dataset": ds.stats(), "timings": {k: round(v, 2) for k, v in res.timings.items()}}
    if ds.truth is not None:
        s1_ids = ds.s1["entity_id"]
        report["final"] = summary(res.matches, ds.truth, s1_ids)
        report["blocking"] = candidate_metrics(res.prepared.blocked.pairs, ds.truth, s1_ids, ds.recs.height)
        report["slices"] = _slice_metrics(res.matches, ds.truth, ds.s1, res.prepared.recn)
        report["threshold_curve"] = threshold_curve(res.scored.select("s1", "rec", "p"), ds.truth, s1_ids)
    return res, report
