# Reproducing the competition submission

This is the original 8 GB-laptop workflow that produced the leaderboard files from the **real** competition data (not in git). It streams everything through `work/` on disk. The shared algorithms now live in `resolver/`; the competition-scale drivers live in `competition/`, and the experiments in `experiments/`.

For anything else (the demo, the API, new data in the same format) use the `python -m resolver` workflow described in the main README.

## 1. Setup (once)

```bash
git clone https://github.com/sakshamg251206/amazon_ml.git && cd amazon_ml
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # macOS: also `brew install libomp` (needed by LightGBM)
python -m pytest -q                      # should pass
```

**Data.** It is not in git. Put the competition files here:

```
student_resource/dataset/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv
student_resource/dataset/test/{test_source1,test_source2,test_source3}.tsv
```

(Other locations work via the `ER_DATA_DIR`, `ER_WORK_DIR` and `ER_OUTPUT_DIR` environment variables.)

**Machine.** This was built on an 8 GB Mac. 16 GB is more comfortable. You need about **40 GB free disk**: `work/` grows to ~25 GB and `output/` to ~7 GB.

**Warning.** Run **one heavy step at a time**. Two heavy jobs together can exhaust 8 GB; this machine rebooted once.

## 2. Full reproduction, in order

Times are for an 8 GB M2 Mac. Every step writes to `work/` and can be re-run.

### A. Validation data and models

Validation uses 19 whole states (414k train S1), with states split into 2 folds.

| # | Command | What it does | Time |
|---|---|---|---|
| 1 | `python -m competition.prep` | Normalise all TSVs → `work/norm/*.parquet` | ~9 min |
| 2 | `python -m experiments.e_b2_state` | Label-free state aliases → `work/alias_{train,test}.parquet` | ~2 min |
| 3 | `python -m experiments.e_b3_tfidf` | TF-IDF candidates for the validation states | ~20 min |
| 4 | `python -m experiments.e_b3b_union 5` | Top-5 ∪ exact keys; validation S1 list | ~5 min |
| 5 | `python -m experiments.e_m1_lgbm 5` (**run twice**) | 1st run: pair features. 2nd run: M1 out-of-fold (0.942) | ~5 + 7 min |
| 6 | `python -m experiments.e_f2_comp` then `python -m experiments.e_f2_comp mask` | Competition/difference features (0.9585) | ~10 min each |
| 7 | `python -m experiments.e_t2_adaptive e_f2b` | Adaptive generic/rare vocabulary (0.9594) | ~10 min |
| 8 | `python -m experiments.e_e2_ensemble_v2 e_f2b_gen ens3g` | Stage 2 (LightGBM + ExtraTrees + MLP) out-of-fold score + final models → `work/ens3g/` | ~40 min |

### B. Test set → submission

| # | Command | Time |
|---|---|---|
| 9 | `python -m competition.pipeline --split test --stage block` | **~4.5 h** (TF-IDF per state; resumable) |
| 10 | `python -m competition.pipeline --split test --stage features` | ~7 min |
| 11 | `python -m competition.pipeline --split test --stage features3` | ~10 min |
| 12 | `python -m competition.pipeline --stage train --v3` | ~5 min |
| 13 | `python -m competition.pipeline --split test --stage decide --v3 --ens` | ~15 min → `output/sub4d_ens_ef/` |
| 14 | `python student_resource/utils/validate_submission.py -m output/sub4d_ens_ef/matching_results.tsv -t student_resource/dataset/test` | Must print `PASS` |

**Upload only `matching_results.tsv`** to the leaderboard. `candidate_pairs.tsv` is for the final zip.

## 3. How to contribute an experiment

1. **Branch:** `git checkout -b exp-<idea>`.
2. **Check `DECISIONS.md`** to see whether the idea was already measured.
3. **Build the feature** in `resolver/` as a function over `(s1, rec)` pairs, and add a runnable self-check.
   - Examples: `resolver/features2.py`, `resolver/adaptive.py`.
4. **Measure** with a copy of `experiments/e_t2_adaptive.py`:
   - add your features to `work/e_f2b_gen_feats.parquet`;
   - retrain M1 on the same folds;
   - compare with **0.9594** (M1).
   - **Keep it only if F0.5 rises in both folds.**
5. **If it wins,** rerun stage 2 (step 8) against **0.9614**, then the test steps (11–14).
6. **Record the result:**
   - the experiment log goes to `experiments/log.csv` automatically;
   - add a `D<n>` entry to `DECISIONS.md` (the what, the numbers and the decision);
   - open a pull request into `main`.

**Rules that saved us:**
- **No external data or lookups** (competition rule).
- **Never tune on the test set.** Validation is out-of-fold by state.
- **Avoid features that count density or raw token frequency.** They broke France for another team.
- **Commit code, never data.** `work/` and `output/` are gitignored.
