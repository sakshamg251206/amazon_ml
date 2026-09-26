# 02: SanthoshReddy352 src/features.py (primary code)

- **URL:** https://raw.githubusercontent.com/SanthoshReddy352/Amazon-ML-Challenge-2026/main/code/business_entity_resolution/src/features.py
- **Type:** primary / code. **Credibility:** high.

## Quotes

```
(pl.col("score") >= pl.col("rev_best")).alias("rev_is_best"),
pl.when(pl.col("score") >= pl.col("rev_best"))
  .then(pl.col("score") - pl.col("rev_second"))
  .otherwise(pl.col("score") - pl.col("rev_best"))
  .alias("rev_margin"),
(pl.col("score") / pl.col("rev_best")).alias("rev_score_rel"),
```

- "The rev_ features use the blocker's combined score and name_score computed per record (m, src) across all candidate S1s."
- Also `ctx_score_rel`, `ctx_score_gap`, `ctx_name_rel`, `ctx_addr_rel`, `ctx_n_cands`: "Within-S1 score relations".
