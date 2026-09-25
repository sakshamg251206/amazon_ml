# Project State

**Last updated:** 26 Sep 2026, 00:40 IST
**Branch:** `phase-2-experiments`
**Baseline:** tag `v0_rule`

## Headline

- Best validated result: **macro F0.5 = 0.9471** (experiment E-M2), up from **v0_rule = 0.5950**.
- It is measured out-of-fold at real density: 414k train S1 in 19 whole states, with states held out 2-fold.
- **This is a validation number, not a leaderboard score.** The public leaderboard has run 0.01–0.02 below validation for other teams, mostly because of France, which our validation cannot cover.
- The full test pipeline (blocking + model on all 1.73M test S1) has **not been run yet**. Submission 1 on the leaderboard is still the v0 rule file.

## Results ladder

All rows use the same validation sample (19 whole states, 414,741 S1), except v0.

| Version | What | F0.5 | Precision | Recall | Candidate recall | Ceiling |
|---|---|---|---|---|---|---|
| v0_rule | Exact (country, name_key, house_no) rule (on full train) | **0.5950** | 0.983* | — | 0.39 | 0.62 |
| E-B1 | Union of 5 exact keys (candidates only, full train) | — | — | — | 0.80 | 0.90 |
| E-B3 | Partitioned TF-IDF top-5 ∪ exact keys (candidates only) | — | — | — | **0.970** | **0.9905** |
| E-F1 | Fuzzy score, no learning: 0.8·token-set + 0.2·house-no-equal, t = 0.80 | 0.799 | 0.920 | 0.681 | 0.969 | 0.990 |
| E-M1 | LightGBM (34 features) + each record → best S1 if p ≥ 0.70 | 0.942 | 0.980 | 0.891 | 0.969 | 0.990 |
| **E-M2** | + stage-2 collective and graph-twin features | **0.9471** | 0.980 | 0.906 | 0.969 | 0.990 |

\* Pair precision recomputed by the phase-1 reviewer.

Per country for E-M2: US 0.954, India 0.936.

## Experiments that were run and rejected

| ID | Hypothesis | Measurement | Decision |
|---|---|---|---|
| E-G1 | Transitive / BFS propagation along S2–S3 twin edges recovers missed matches | Candidate recall 0.9701 → 0.9713; ceiling +0.0003; 1.1M pairs added, only 0.15% true; 6.6% of twin components mix owners | **Rejected** as hard post-processing. Kept as soft features (E-M2) |
| E-G1 | Distractors form "phantom" multi-copy clusters | Only 0.6% of twin components are all-distractor | **Rejected** (agrees with the InvincibleAgam repo) |
| E-L1 | Train and test share businesses, so train S1 exposes test distractors (a possible explanation for a 0.998 leaderboard) | Distractor hit-rate on the other split's S1 is 0.7% vs 1.5% for matched records; 0.0% full name+address overlap | **Rejected** |
| Leakage | IDs or row order carry signal | corr ≈ 0.000; same-entity rows are about 1.1M apart | **None found** |

## Pipeline components (tested)

| Module | Job | Tests |
|---|---|---|
| `src/prep.py` | Normalizes all 26M rows to Parquet (`work/norm/`) in 9 min, in slices sized for 8 GB | — |
| `src/normalize.py` | v1 keys; v2: `name_tokens` (domains, `#tags`, legal forms), `nums`, `addr_parts`, `is_nonlatin` | 8 |
| `src/partition.py` | Label-free state inference: the S1 state is its most frequent address part; aliases learned from S1 plus record co-occurrence; US full-name override | 4 |
| `src/block.py` | TF-IDF retrieval within a partition: name char-3gram / address word / combined / address-only views | 5 |
| `src/features.py` | 34 country-agnostic pair features (fuzzy, number relation, rank/margin, flags) | 3 |
| `src/evalx.py` | Vectorized macro F0.5, candidate recall, oracle ceiling, experiment logger (runtime + peak memory) | 3 |
| `experiments/e_*.py` | One script per experiment. Every result is appended to `experiments/log.csv` | — |

Full suite: 34+ tests passing.

## Known limitations and open issues

- Validation covers **US and India only**. France has no labels.
  - Planned: label-free per-country checks and leave-one-country-out (LOCO).
- M1/M2 train and evaluate on a **15% state sample**. The learning curve and a full-train fit are not measured yet.
- The M2 feature-importance split looks odd: `addr_partial` has 0.64 of gain and `len_name_rec` 0.34. Needs a look.
- The `rev_rank_fz` and `n_cand_s1` features are only approximate under record-hash chunking. This is consistent between train and test.
- A few bogus S1 "states" exist (`unit b`, `greenville`: about 150 S1). A state-vocabulary filter would fix them.
- Indic-script names still go through anyascii only (`praivet`, `elelpi`), and India is 0.02 behind the US.
- **Memory:** the 8 GB Mac OOM-killed B3 and M1 several times. Every stage now streams to disk. A full test run will need chunking by partition, or Colab.

## Next steps

These are ordered by expected gain per hour.

1. **Test pipeline:** block the full test set, compute features, run the M2 model, and write both TSVs. Validate, then make **Submission 2**. This turns 0.947 on validation into a leaderboard number.
2. **Learned Indic transliteration dictionary** (token alignment on matched train pairs) plus legal-form variants (`praivet`, `elelpi`). Targets India (0.936) and the non-Latin blocking misses.
3. **House-number relation features:** prefix/truncation, same-length digit change, last-number relation. InvincibleAgam measured +0.0015 from these.
4. **A decoding rule tuned under a simulated test prior** (test has about 40% distractors vs 26% in train, per InvincibleAgam).
5. **France:** French abbreviation / department lexicon, initials (acronym) key, label-free sanity statistics, and France-only leaderboard probes.
