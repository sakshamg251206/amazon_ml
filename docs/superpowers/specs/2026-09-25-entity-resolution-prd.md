# PRD: Business Entity Resolution Pipeline (Amazon ML Challenge 2026)

**Owner:** Saksham Garg (solo)
**Date:** 25 Sep 2026, 21:00 IST
**Deadline:** 27 Sep 2026, 23:59 IST
**Background:** `AMAZON_ML_DEEP_RESEARCH.md`. Section numbers written "§n" below refer to that report.

---

## 1. Goal

Maximize **private-leaderboard macro F0.5**, and ship a package that reproduces it.

What we optimize:
- the **holdout test-mix F0.5**: a realistic-density holdout, reweighted to the test country mix;
- **robustness on France**, the unseen country.

We do not optimize the public leaderboard directly.

### Success criteria

| # | Criterion | Deadline |
|---|---|---|
| S1 | A valid submission on the leaderboard | 25 Sep, before 23:59 IST |
| S2 | Holdout test-mix F0.5 ≥ 0.97 | 26 Sep, 13:00 |
| S3 | The final package reproduces the submitted `matching_results.tsv` exactly, starting from the raw TSVs on a fresh Colab | Final submission |
| S4 (stretch) | Public leaderboard ≥ 0.965 | Final submission |

### Non-goals

- Embedding-based blocking. Measured recall is 0.70 against 0.97 for sparse TF-IDF (§6.3).
- Cross-encoders and LLMs.
- Boosting ensembles.
- Hyperparameter search.
- 5-fold CV.
- Any score measured against a sampled S1 index.

---

## 2. Constraints

### Rules [OFF]

- At most 5 submissions per day, over 3 days.
- No external data: no geocoding, registries or APIs.
- The final model must be MIT or Apache-2.0 licensed and have ≤8B parameters.
- Outputs:
  - `matching_results.tsv` is the only scored file;
  - `candidate_pairs.tsv` must be the *final* candidate set the model scores, and every match must appear in it.
- Every test S1 needs exactly one row.
- Deliverable zip: `output/`, `code/business_entity_resolution/{src,README.md,requirements.txt}`, and the filled-in `Documentation_template.md`.

### Compute

| Machine | Spec | Role |
|---|---|---|
| Mac | M2, 8 GB RAM, about 11 GB free disk | Development and tests on a 1% sample |
| Google Colab | Free: about 2 vCPU / 12 GB RAM | All full-scale stages |
| Google Drive | Free tier: 15 GB | Data and checkpoints |

### Data

The dataset is at `student_resource/dataset/{train,test}/`. It is gitignored. Row counts have been verified against the competitor EDA.

| | S1 | S2 | S3 |
|---|---|---|---|
| Train | 2,206,821 | 5,034,616 | 5,285,603 |
| Test | 1,732,544 | 4,887,273 | 5,082,316 |

Test S1 country mix: India 47%, US 38%, **France 15% (unseen in training)**.

---

## 3. Architecture

### 3.1 Five resumable stages

Each stage writes Parquet under `WORK_DIR` (Google Drive on Colab). A stage skips any output file that already exists, so a disconnect only costs the chunk that was in progress.

```
prep → block → features → train → decide
```

| Stage | Input | Output | Notes |
|---|---|---|---|
| `prep` | Raw TSVs | `norm/{split}_s{1,2,3}.parquet` | Read with `sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False` |
| `block` | Normalized records | `cand/{split}/{country}_{partition}.parquet` with columns `s1, rec, channel_flags, cos_name, cos_addr, cos_comb` | Details in §3.3 |
| `features` | Candidates + records | `feat/{split}/…parquet` | Details in §3.4 |
| `train` | Train features + ground truth | `model/lgbm.txt`, `model/threshold.json` | Details in §3.5 |
| `decide` | Test features + model | `output/matching_results.tsv`, `output/candidate_pairs.tsv` | Runs the official validator with `--check-ids` |

### 3.2 Normalization (`prep`)

The normalizer is **country-agnostic code driven by lexicons**. Steps:
- NFKC Unicode normalization.
- `anyascii` transliteration (a rule-based library, not a lookup).
- Lowercase.
- Strip junk (`<NULL>`, `N/A`, `##`) and number prefixes (`Door No`, `H.NO`, `No.`, `Unit`).
- Expand abbreviations for the US, India and France:
  - France: R / R. / AV / BD / IMP / ALL / RTE / PL / QUAI / FBG → full words; BIS / TER kept as number suffixes.
- Strip leading zeros from numbers.

Parsed fields per record:

| Field | Content |
|---|---|
| `name_core` | Name without legal forms or honorifics |
| `legal_form` | Canonical legal-form token (SARL, SAS, SASU, EURL, SCI, SA and the US/India equivalents) |
| `name_nospace` | Name with spaces removed (matches domain-style names) |
| `house_no` | First number |
| `nums` | Set of all numbers |
| `street_tokens`, `locality_tokens` | Address components |
| `state_canon` | Canonical state or region |
| `partition_key` | Blocking partition (§3.3) |
| `flags` | `empty_addr`, `nonlatin_name`, `domain_name` |

`state_canon` is swap-aware:
- US abbreviation ↔ full name.
- Indian states, including native-script names; Telangana and Andhra Pradesh are an "equivalence group".
- French department → region, plus the `hdf` / `naq` / `pdl` codes.
- A France city → region map, covering the ~15 cities, for records that carry no region.

The lexicons are hand-written generic domain knowledge. `normalize.py` documents where they come from.

### 3.3 Blocking (`block`) — Approach A

Retrieval direction is **S2/S3 → S1**, because each record has at most one owner. Three channels are unioned per record:

1. **Partitioned TF-IDF.** Partition = (country, `partition_key`), where `partition_key` is the state's equivalence group.
   - Name: char 3–4-gram vectors.
   - Address: word uni- and bi-grams.
   - Combined view: 0.6 × name + 0.4 × address, L2-normalized.
   - Keep the top k per view: default k_name = 20, k_comb = 50.
   - Top-k search uses `sparse_dot_topn`.
2. **Exact keys across the whole country:**
   - `name_nospace`;
   - `house_no` + first street token;
   - `name_core` + locality token.
3. **Fallback.** Records with no state or an ambiguous one are searched against the whole country's S1, combined view only, k = 20.

Memory rule: only one partition's matrices are held in memory at a time. Candidates are streamed to Parquet.

**Gate G-block:** holdout blocking recall ≥ 0.97 (measured against all 2.2M train S1) before training. If it falls short, raise k or add a channel first.

### 3.4 Features (`features`)

About 40 features, **all country-agnostic**. `country` is never a feature.

| Group | Features |
|---|---|
| Name | rapidfuzz ratio / token_sort / token_set / partial / Jaro-Winkler on `name_core`; `name_nospace` ratio; token Jaccard; first-token equality; legal-form agree / conflict / missing |
| Address | ratio / token_set / partial; number-set Jaccard; empty-address flags |
| Number relationship | Equal / dropped / added / differs; digit edit distance; prefix or substring (1600 vs 160); unit or slash parts equal |
| Context (per record) | Rank of this S1 among the record's candidates; gap to the best; **margin over the second-best**; candidate count; which channels proposed the pair; blocking cosines |
| Source | S2 or S3 |

Excluded (these broke France for other teams, §6.3):
- density or crowding features;
- raw token-frequency features;
- word-identity log-odds.

### 3.5 Training (`train`)

- **Model:** LightGBM, binary classification.
  - Settings: `num_leaves` 127, learning rate 0.05, early stopping on the validation fold. Seed fixed and logged.
- **Training rows:**
  - candidates of about 250k train S1, drawn from the non-holdout 90%;
  - blocked against **all** train S1, so the density is realistic;
  - subsample to at most 25M pairs if memory requires it.

### 3.6 Decoding (`decide`)

1. Each record keeps only its highest-probability S1.
2. Keep the pair if p ≥ t.
3. t is swept over [0.01, 0.95] on the holdout. §5.3 of the research predicts the optimum near 0.02.
4. `candidate_pairs.tsv` = all pairs that were scored.
5. Rows are written by iterating over the `test_source1` IDs, so every S1 (France included) gets exactly one row.

### 3.7 Code layout

```
src/config.py      paths, constants, seeds
src/normalize.py   cleaning, lexicons, field parsing
src/block.py       partitioned TF-IDF + exact keys + fallback
src/features.py    pair features
src/train.py       fit + save LightGBM
src/decide.py      assignment, threshold, TSV writing
src/metrics.py     exact macro F0.5 + blocking recall/oracle
src/run.py         CLI: run.py --stage {prep,block,features,train,decide} --split {train,test} [--sample 0.01]
tests/             pytest (metrics, normalize, decide)
experiments/log.csv   one row per run: version, config hash, seed, all §4 metrics, LB score
```

---

## 4. Validation

### 4.1 Holdout

- 10% of train S1 (about 220k), selected by a hash of the entity ID and stratified by country × match count.
- Holdout records are blocked against **all** train S1.

### 4.2 Metrics reported on every run

Each run appends a row to `experiments/log.csv`.

| Metric | Definition |
|---|---|
| Blocking recall | Share of true pairs present in the candidates |
| Oracle F0.5 | Score a perfect matcher would get on these candidates |
| Candidates per S1 | The reduction ratio the organisers audit |
| Holdout macro F0.5 | Measured after decoding |
| **Test-mix F0.5 (headline)** | Holdout F0.5 reweighted to India 47% / US 38%; France's 15% is filled with the leave-one-country-out estimate |
| Breakdowns | Country × source × true-set size (0 / 1–2 / 3+) |
| Leave-one-country-out | Train on US → score India, and India → US |

### 4.3 Label-free test checks (per country, before every upload)

- Predicted singleton rate.
- Matches per S1.
- Share of records assigned.
- Share of kept pairs with a house-number conflict.

**Rule:** if France deviates strongly from the US and India on these, investigate before uploading.

### 4.4 Scorer tests (`tests/test_metrics.py`)

| Input | Expected score |
|---|---|
| Empty prediction for every train S1 | 0.0558 (the singleton rate) |
| Perfect prediction | 1.0 |
| Worked example from the problem statement PDF | 0.714 |

---

## 5. Gated edges

Each edge is built **only** if its gate passes. Each is its own version in `experiments/log.csv`.

| Edge | Experiment (cost) | Gate | Build if it passes |
|---|---|---|---|
| **1. Metric-aware decoding** | **E7:** isotonic calibration per source; within each S1, keep the k-th assigned record only if p > (k−1)/((k−1)+0.25·n̂), where n̂ is the expected set size under the test prior (about 4.1). Re-decode only (minutes) | Test-mix F0.5 ≥ baseline + 0.002 | Replace the global t in `decide.py` |
| **2. France hardening** | French lexicon (always on); **E8:** leave-one-country-out per feature group (1 h); **E11:** France-only leaderboard probes (≤2 submissions) | E8: drop any group that widens the gap. E11: probe raises the public score | Keep the winning France rule |
| **3a. Phantom clusters** | **E1:** with ground truth, compare how often unmatched records share (core name, street, house number) with *other* records vs how often a true copy's deviation from S1 is shared (about 30 min) | A clear rate gap, e.g. ≥3× | Cluster features: "number of other records sharing my deviation from S1", "sources in my cluster" |
| **3b. Twin recall** | **E2:** share of blocking misses (non-Latin names, empty addresses) that have an exact-key twin which *is* retrieved correctly (about 30 min) | ≥20% of misses recoverable | S2/S3↔S2/S3 exact-key pass; records inherit their twin's candidate S1s |
| Duplicate collapse | **E4:** do exact or light duplicates within S2 always get the same owner? (10 min) | 100% consistency (or ≥99.9%) | Deduplicate before scoring |
| Rule lane | **E5:** precision of "exact normalized name + address + number, no rival" per country (20 min) | ≥99.9% in every country | Accept those pairs without the model |

---

## 6. Timeline and submission plan

All times are IST.

| When | Work | Submission |
|---|---|---|
| 25 Sep 21:00–00:00 | `metrics.py` + tests; E0 / E1 / E2 / E4; `prep` on full data; exact-key rule baseline | **Sub 1 before 23:59:** the rule baseline. It uses today's quota (which probably expires) and calibrates the leaderboard |
| 26 Sep 00:00–09:00 | `block` on holdout + training sample, then on test (Colab, overnight) | — |
| 26 Sep 09:00–13:00 | `features`, `train`, threshold sweep, holdout metrics | **Sub 2:** refined baseline |
| 26 Sep 13:00–20:00 | Edges 1 and 2 | Subs 3–4 (one France probe) |
| 26 Sep 20:00–00:00 | Edge 3 if E1/E2 passed; otherwise E4/E5 | Sub 5 if the holdout improved |
| 27 Sep 00:00–18:00 | Last improvement cycle; France probe 2 | Subs 6–8 |
| **27 Sep 18:00** | **Freeze.** Regenerate from raw on a fresh Colab; run the validator; fill in the documentation; build the zip | **Final upload by 21:00** |

### Submission rules

- One change per submission.
- Archive every uploaded TSV alongside its holdout metrics.
- Never follow the public leaderboard against the direction of the holdout.
- **Final pick:** the best test-mix holdout result that the public board does not contradict.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Colab disconnects | Every stage resumes from Drive checkpoints; the test stage runs in chunks |
| Colab RAM (about 12 GB) | One partition in memory at a time; streamed Parquet; integer IDs |
| Drive space (15 GB) | Integer IDs; delete intermediates once consumed; check usage after each stage |
| Mac disk (11 GB free) | Full-scale artifacts live on Drive only |
| State-swap noise loses recall | Equivalence groups + exact keys + fallback channel; Gate G-block |
| France has no labels | Country-agnostic features, leave-one-country-out, label-free checks, probes |
| Quota reset time and public/private split unknown | Assume a midnight IST reset; Sub 1 tonight reveals behaviour |
| Grey area on pseudo-labels | Not planned. If used, disclose in the documentation |
| Reproducibility review | Pinned `requirements.txt`; a single `run.py` entry point; the regenerate-from-raw check at freeze |

---

## 8. Open questions

These are to be resolved during execution.

1. Is the public leaderboard stratified by country? The France probes (E11) partly answer this.
2. When does the daily submission quota reset?
3. Is Colab Free enough for test blocking, or is Pro needed? Measure on the first India partition.
