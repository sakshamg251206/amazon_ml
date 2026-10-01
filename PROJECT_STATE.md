# Project State

> **v2.0 (Oct 2026):** the pipeline is now the `resolver` package with a web app, API, synthetic data and Docker deploy. See README.md and DECISIONS.md D30. The competition-era state below is unchanged; its commands now live under `competition/`.

**Last updated:** 26 Sep 2026, 18:35 IST
**Branch:** `phase-2-experiments`
**Baseline:** tag `v0_rule`

## Headline

- Best validated result: **macro F0.5 = 0.9614** (E-E2: stage-2 ensemble on top of E-F2b, `output/sub3d_ens_ef/`). Stage 1 alone: 0.9585 (E-F2b: M1 with 14 new competition/difference features, then expected-F decoding), up from 0.9510 (E-E1 ensemble on old features) and **v0_rule = 0.5950**. See `research/f05-gap/` and D17.
- It is measured out-of-fold at real density: 414k train S1 in 19 whole states, with states held out 2-fold.
- **This is a validation number, not a leaderboard score.** Expected leaderboard is about 0.93–0.94 (France unlabeled; test has more distractors).
- The candidate ceiling is 0.990 (oracle F0.5 on our candidates), so the remaining gap is the matcher, not blocking.

## Submission files (all pass `student_resource/utils/validate_submission.py`)

Only `matching_results.tsv` is scored on the leaderboard. `candidate_pairs.tsv` goes in the final zip.

| Folder under `output/` | Candidates | Scoring | Validation F0.5 |
|---|---|---|---|
| **`sub3d_ens_ef/`** | Full | M1 v2 → stage-2 ensemble (58 features) → expected-F | **0.9614** |
| `sub3b_m1_ef/` | Full | M1 v2 (48 features, key-only masked) → expected-F | 0.9585 |
| `sub3b_m1_ef_unmasked/` | Full | M1 v2 unmasked → expected-F | 0.9590 (validation artifact on key-only pairs) |
| `sub2d_ens_ef/` | Full (TF-IDF top-5 per state ∪ exact keys) | M1 → 3-model ensemble → expected-F | 0.9510 |
| `sub2b_m1_ef/` | Full | M1 → expected-F | 0.9429 |
| `quick_keys/sub2d_ens_ef/` | Exact keys only | Ensemble → expected-F | — (probe) |
| `quick_keys/sub2c_m2_ef/` | Exact keys only | M2 → expected-F | — (probe) |
| `sub2a_keys/` | Exact keys only | M1, t = 0.70 | — (probe) |
| `output/*.tsv` (top level) | v0 rule | Rule | Submission 1 (train 0.595) |

Test output sanity (ensemble file): 94.5–95.5% of S1 have a match in each country (train: 94.4%); 3.3 matches per S1 (validation 3.2); keeps 98.2% of the M1 file's matches.

## Reproduce the final file

```
.venv/bin/python -m competition.prep                                         # normalise raw TSVs -> work/norm/ (9 min)
.venv/bin/python -m competition.pipeline --split test --stage block          # ~4.5 h on the 8 GB M2 Mac
.venv/bin/python -m competition.pipeline --split test --stage features       # ~7 min
.venv/bin/python -m competition.pipeline --stage train                       # M1 -> work/model_m1.txt
.venv/bin/python -m competition.pipeline --split test --stage decide         # M1 file (sub2b_m1_ef)
.venv/bin/python -m competition.pipeline --split test --stage decide --ens   # v1 ensemble file (sub2d_ens_ef), ~30 min
.venv/bin/python -m experiments.e_f2_comp && .venv/bin/python -m experiments.e_f2_comp mask   # E-F2 / E-F2b validation features
.venv/bin/python -m competition.pipeline --split test --stage features2      # add the 14 features to test buckets (~2 min)
.venv/bin/python -m competition.pipeline --stage train --v2                  # M1 v2 on e_f2b_feats -> work/model_m1b.txt
.venv/bin/python -m competition.pipeline --split test --stage decide --v2    # sub3b_m1_ef (validation 0.9585)
.venv/bin/python -m experiments.e_e2_ensemble_v2                     # stage-2 v2 OOF + final models -> work/ens2/
.venv/bin/python -m competition.pipeline --split test --stage decide --v2 --ens   # sub3d_ens_ef (validation 0.9614)
```

The ensemble's final models are `work/ens/final_{lgbm,extratrees,mlp}.joblib` with `final_weights.joblib` (equal). They are trained by `from experiments.e_e1_ensemble import final; final()` — run via import, not `-m`, so they unpickle under `experiments.e_e1_ensemble`. Validation data for them: `experiments/e_m1_lgbm.py` (features) → `experiments/e_e1_ensemble.py build`.

## Results ladder

Validation sample: 19 whole states, 413,741 S1, out-of-fold (2 folds by state).

| Version | What | F0.5 | Precision | Recall |
|---|---|---|---|---|
| v0_rule | Exact (country, name_key, house_no) rule (full train) | 0.5950 | 0.983 | — |
| E-F1 | Fuzzy score, no learning | 0.799 | 0.920 | 0.681 |
| E-M1 | LightGBM, 34 features, t = 0.70 | 0.9420 | 0.980 | 0.891 |
| E-D1 | + expected-F decoding | 0.9429 | 0.979 | 0.892 |
| E-M2 | + stage-2 collective/twin features | 0.9471 | 0.980 | 0.906 |
| E-M2b | + pruning (p1 ≥ 0.01) + expected-F | 0.9488 | 0.981 | 0.904 |
| E-E1 | 3-model stage-2 ensemble + expected-F | 0.9510 | — | — |
| E-F2 | M1 + 14 competition/difference features + expected-F | 0.9590 | 0.984 | 0.920 |
| E-F2b | E-F2 with record-side features blanked on key-only pairs | 0.9585 | 0.984 | 0.920 |
| **E-E2** | **E-F2b + stage-2 ensemble (LightGBM + ExtraTrees + MLP mean) + expected-F** | **0.9614** | — | — |

### E-E1: five models, tuned individually (stage 2, same 44 features)

| Model | Best config | F0.5 | Fold 0 | Fold 1 |
|---|---|---|---|---|
| LightGBM | b: 127 leaves, lr 0.05, 700 rounds | 0.9498 | 0.9455 | 0.9532 |
| HistGradientBoosting | a | 0.9493 | 0.9448 | 0.9528 |
| ExtraTrees | a: 150 trees, leaf 40, max_features 0.5 | 0.9504 | 0.9455 | 0.9542 |
| Logistic regression | a | 0.9419 | 0.9340 | 0.9476 |
| MLP | b: (64, 32) | 0.9495 | 0.9445 | 0.9533 |
| 5-model weighted (cross-fitted) | — | 0.9508 | 0.9464 | 0.9542 |
| **3-model mean (LGBM + ET + MLP)** | — | **0.9510** | 0.9463 | 0.9546 |

Round 2 (a stronger config per model, and binned logistic regression) is logged in `experiments/log.csv` as `E-E1:*`. Its winner is recorded in `work/ens/best_cfg.json`.

### Decoding (E-D1)

| Decoding | M1 | M2 |
|---|---|---|
| t = 0.70 | 0.9420 | 0.9471 |
| Best global t | 0.9421 | 0.9472 |
| Per-country t | 0.9422 | 0.9473 |
| Expected-F | 0.9429 | 0.9482 |

Where M2 loses true pairs: 5.7% have p below threshold, and 0.4% lose to another S1 (exclusive assignment is almost always right). False positives are mostly distractor records, not swaps.

### Full validation report (M2 + expected-F)

| Metric | Value |
|---|---|
| Blocking recall | 0.9687 |
| Candidate pairs | 12.7M (30.7 per S1) |
| Reduction ratio | 0.999994 |
| Macro F0.5 / F1 | 0.9482 / 0.9269 |
| Pair precision / recall | 0.9795 / 0.9060 |
| FP rate on negative candidates | 0.0024 |
| Singletons | 5.6% of S1; 86.5% correctly empty |

### By slice (ExtraTrees stage 2, pairs with true match in candidates)

| Slice | Precision | Recall |
|---|---|---|
| Source 2 / Source 3 records | 0.982 / 0.983 | 0.935 / 0.935 |
| US | 0.986 | 0.939 |
| India | 0.977 | 0.928 |
| India, non-Latin script | 0.973 | 0.912 |

## Rejected (measured)

| ID | Idea | Result |
|---|---|---|
| E-G1 | Graph / BFS propagation as hard merges | +0.0003 ceiling; only 0.15% of 1.1M added pairs true → features only |
| E-L1 | Train/test overlap | None; also no ID or row-order leakage |
| E-D1 | Per-country thresholds | +0.0001 (noise) |
| — | Translation of non-Latin text | Not run. Non-Latin is 7% of true pairs; worst-case upside ≈ +0.003; no offline translator feasible |
| — | Transformers / cross-encoders | Not run: no GPU, no torch, 8 GB RAM |

## Known limitations

- France is unvalidated (no labels). Its test match rate (95.5%) looks like the other countries'.
- Models are trained on the 15% state sample, not all of train.
- The pipeline's TF-IDF blocking prunes features in more than 5% of a state's S1 (`MAX_DF`). Validation candidates were built without pruning; E-B4 showed identical recall on 3 states.
- `sparse_dot_topn` was replaced by pure scipy top-k (it segfaulted on Apple Silicon).

## Next steps (if time)

1. Train stage-1/2 models on all train states (needs full train blocking, ~4–6 h).
2. Indic transliteration dictionary (India non-Latin recall 0.912).
3. Leaderboard probes for a test-prior threshold shift.
