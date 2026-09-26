# 08: Our out-of-fold error profile (primary data, reproducible)

- **Inputs:** M1 OOF (`work/e_m1_oof_k5.parquet`) and features (`work/e_m1_feats_k5`). 2026-09-26 17:48.
- **Sample:** 26,381 confident FPs vs 1,275,530 confident TPs, after exclusive assignment at p ≥ 0.7.

| Signal | FP | TP |
|---|---|---|
| Record's cosine-best S1 is a different S1 | 18.5% | 0.7% |
| House numbers present and different | 44.5% | 11.7% |
| ctx_rel_cos median (this record's cosine vs S1's best record) | 0.655 | 0.920 |
| cos_comb median | 0.655 | 0.840 |

- Examples of confident FPs: "indo lotus enterprises" vs "indo memorial trust" at the same building (p = 1.0); "food india" vs "food builders" at the same address (p = 0.99); "nistha it" #4 vs "nistha son" #628.
- Blocking-miss audit: 44.8k true pairs missed (3.13%); 62.5% of those misses have a record with no inferred state (mostly empty addresses).
