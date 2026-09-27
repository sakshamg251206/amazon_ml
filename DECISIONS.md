# Decisions Log

Each entry records the decision, the evidence behind it, and what it would cost if it turns out to be wrong. Newest entries are last.

---

## D1 — Freeze v0_rule as the baseline

- **Decision:** tag `v0_rule` (train F0.5 0.5950), keep it reproducible, and do not merge anything into `main` during Phase 2.
- **Why:** every experiment is compared against it.
- **Cost if wrong:** none.

## D2 — Validate by sampling whole (country, state) partitions at full density

- **Decision:** the validation sample is 19 whole states covering 414k S1. Every sampled S1 competes with all of its real neighbours.
- **Why:** sampled S1 indexes overstate recall. Another team measured 0.997 on a sample vs 0.967 at full size.
- **Cost if wrong:** state-level variance. Two folds of about 200k S1 each keep that small.

## D3 — State partitions are inferred without labels

- **Decision:** an S1's state is its most frequent address component. Aliases are learned from S1 and from record co-occurrence, plus a US state-name override.
- **Evidence:**
  - The naive "last component is the state" gave 84–86% accuracy.
  - The frequency rule gives 98.8% (US) / 99.9% (India).
  - Towns named after states (Indiana PA, Maryland NY) required the override.
- **Cost if wrong:** about 4–6% of true pairs fall outside their partition. The exact-key channel is country-wide and catches part of them.

## D4 — Candidates = TF-IDF top-5 within partition ∪ five exact keys

- **Evidence:**
  - Keys alone: recall 0.786.
  - TF-IDF top-1: 0.916. Top-5: 0.926. Top-10: 0.929 (a plateau).
  - Union of top-5 and keys: **0.970** recall, ceiling 0.9905, at 31.6 candidates per S1.
  - This passes the PRD gate (≥ 0.97).
- **Cost if wrong:** about 3% recall. The next levers are transliteration, initials keys and neighbour states.

## D5 — Dense embeddings are not used for blocking

- **Decision:** no dense-embedding retrieval.
- **Evidence:** an external measurement found LSA-256 recall 0.70 vs 0.97 for sparse TF-IDF. The signal here is character-level.
- **Cost if wrong:** a small miss on non-Latin names. A learned dictionary is the cheaper fix.

## D6 — Each record goes to its best S1 (exclusive assignment); threshold tuned on macro F0.5

- **Why:** in ground truth, every S2/S3 record belongs to at most one S1.
- **Evidence:** the tuned t is 0.70, not low. At t = 0.01, precision is 0.78 because distractors get assigned. The optimum depends on how well the model separates distractors, so t must be re-swept for every model.
- **Cost if wrong:** a mis-set threshold. It is re-swept each run.

## D7 — LightGBM on country-agnostic features; `country` is never a feature

- **Evidence:** 0.799 (fuzzy, no learning) → 0.942 (LightGBM), on identical candidates.
- **Cost if wrong:** none observed. France behaviour is still unmeasured.

## D8 — Graph / transitive matching is used as soft features, not hard merges

- **Evidence:** propagating candidates along twin edges adds 1.1M pairs, only 0.15% of them true, and 6.6% of twin components mix owners.
- **As features:** +0.005 (0.942 → 0.947).
- **Cost if wrong:** a small amount of recall.

## D9 — Train/test overlap and ID/row-order leakage were ruled out

- **Evidence:** E-L1 and the leakage checks (see PROJECT_STATE.md).
- **Implication:** a leaderboard score of 0.998 is not explained by any structure we found. Treat it as unexplained. Do not assume it is reachable.
- **Cost if wrong:** we might miss a legitimate structural edge.

## D10 — Memory discipline on the 8 GB Mac

- **Decision:** everything is written as slice/chunk Parquet; integer ids are used while accumulating; ground truth is loaded only after training.
- **Why:** B3 and M1 were OOM-killed several times before this.
- **Cost if wrong:** none. Full test runs may still need Colab.

## D11 — Where the Gemini blueprint was not followed

| Gemini said | We did instead |
|---|---|
| Singletons are 20–40% | Measured 5.58% |
| Retrieve top-k per S1 | Retrieve per record, then assign each record to its best S1 |
| GroupKFold by `source1_id` | Fold by partition |
| Conservative t of 0.65–0.78 by rule | Swept on measured macro F0.5 |
| MiniLM embeddings for blocking | Rejected (see D5) |

## D12 — Replace `sparse_dot_topn` with chunked scipy top-k; prune ubiquitous features

- **Decision:** top-k via scipy sparse matmul plus a per-row sort, in chunks bounded to 5M cells and run on a thread pool. Features in more than 5% of a large partition's S1 are dropped for *retrieval* only; cosine *features* always use the full vectors.
- **Evidence:** `sparse_dot_topn` segfaulted (EXC_BAD_ACCESS) on France. E-B4 recall on 3 train states was unchanged (0.9307–0.9469 either way), runtime was 2× faster, and France's worst region ran in 17 min at 1.8 GB.
- **Cost if wrong:** about 14% different candidate pairs than the ones the model was trained on, at the same recall.

## D13 — Expected-F0.5 decoding instead of a single threshold

- **Evidence:** E-D1: +0.0009 (M1), +0.0011 (M2). Threshold and per-country threshold variants were ≤ +0.0002.
- **Why:** the metric is per-S1, and a singleton scores 1 only when nothing is predicted.

## D14 — Stage 2 restricted to pairs with p1 ≥ 0.01

- **Evidence:** E-M2b: 0.9488 vs 0.9482 unpruned. It keeps 15% of pairs, so the test set fits in 8 GB.

## D15 — Final scorer = plain average of LightGBM + ExtraTrees + MLP

- **Evidence:** E-E1 out-of-fold: best single model 0.9504 (ExtraTrees); 3-model mean 0.9510, higher in both folds. The mean beat fitted weights (0.9508) and a logistic-regression stacker (0.9507), so no weights are fitted and nothing overfits. Logistic regression and HistGB got about zero weight.
- **Cost if wrong:** ≤ 0.002. The M1 file is kept as a fallback.

## D16 — No transformers, translation or reinforcement learning

- **Why:** no GPU and no torch on an 8 GB machine, with the deadline in about 1.5 days. Non-Latin pairs are 7% of true pairs, with worst-case upside of about +0.003. This is supervised pair classification with labels, so reinforcement learning does not apply.

## D17 — Competition + difference features on the TF-IDF cosine (E-F2), with key-only masking (E-F2b)

- **Decision:** add 14 features (`src/features2.py`):
  - record-side margin / is-best / relative score on the combined and name cosine (`rev_*`);
  - S1-side relative scores (`ctx_*`);
  - name words added / missing;
  - house-number delta / abs-delta / containment.
- **Evidence:** M1 OOF 0.9429 → **0.9590**, with the same folds, parameters and decoding. `rev_margin_cos` takes 0.55 of gain. This matches SanthoshReddy's feature importance (`rev_margin` 0.66) and the research in `research/f05-gap/`.
- **Masking:** record-side features are blanked on key-only pairs (no cosine).
  - In validation, records outside the sampled states reach sample S1 only through keys, so every candidate of theirs has cosine 0 and looks "best". On test, those records also have TF-IDF candidates.
  - Masked: 0.9585 (−0.0005). This is what ships, because it means the same thing on test.
- **Checked:**
  - The bucketed test computation equals the validation computation exactly (max diff 0.0 on 12.7M rows).
  - On TF-IDF-retrieved pairs, test and validation distributions match ("is best" 0.202 everywhere; France looks like India).
- **Cost if wrong:** the unmasked test file is kept as `output/sub3b_m1_ef_unmasked/`.

## D18 — Stage-2 ensemble on the v2 features (E-E2)

- **Evidence:** same rows (1.70M pairs with p1 ≥ 0.01) and same folds. Baseline masked M1: 0.9585 (fold 0 0.9544 / fold 1 0.9617). Stage-2 models:

  | Model | F0.5 |
  |---|---|
  | LightGBM b | 0.9605 |
  | ExtraTrees a | 0.9612 |
  | MLP b | 0.9603 |
  | **Mean of the three** | **0.9614** (0.9570 / 0.9649) |

  The mean is up in both folds (+0.0026 / +0.0032).
- **Decision:** ship `decide --v2 --ens` → `output/sub3d_ens_ef/`. Final models are in `work/ens2/`.

## D19 — Learned transliteration map features: rejected (E-T1)

- **What was tried:** a map of 362 token mappings learned from 449k train true pairs (non-Latin record ↔ Latin S1; validation-sample S1 excluded), plus 4 features on the mapped name.
- **Result:** M1 out-of-fold 0.9577 vs 0.9585 baseline (**−0.0008**). India fell 0.9481 → 0.9456 and recall dropped.
- **Likely reason:** noisy mappings ("pra"→"pvt", "li"→"ltd", "mam"→"maa"), and making romanized fakes look *more* like their S1 costs precision.
- **Decision:** switched off (`V3_TRANSLIT = False`). The code is kept in `src/translit.py`.

## D20 — Name-only country-wide search for no-state records: rejected (E-B5, memory-safe rerun)

- **Result:** blocking recall 0.9687 → 0.9714; oracle ceiling 0.9901 → 0.9909 (**+0.0008 at best**). Test-side cost is about 1 h.
- **Why so small:** 3-grams in more than 2,000 S1 are dropped (needed for memory safety), and a top-5 among 1.3M names is crowded.
- **Also:** the first, uncapped run (5% DF cap) pushed memory to about 27 GB compressed; the Mac rebooted.
- **Agrees with:** SanthoshReddy v6 ("no gain").

## D21 — Adaptive generic / rare name vocabulary (E-T2, user's tip)

- **What:** per split and country, tokens in ≥ 0.1% of that split's S1 names are "generic" (label-free, so France gets its own list). Six features: extra/missing generic, extra/missing rare, core-name token-set similarity, core names equal.
- **Result:** M1 out-of-fold **0.9594 vs 0.9585** (fold 0 +0.0003, fold 1 +0.0013). India +0.0019; precision and recall both up.
- **Generic lists are stable train → test:** India 238 / 237, US 470 / 471.
- **Decision:** kept. v3 = v2 + generic features (`src/adaptive.py`, `features3`, `ens3g`).

## D22 — Canonical addresses (E-A1): pending stage 2

- **What:** `src/address.py`. State and department names/codes are mapped to one canonical form (e.g. Gironde → nouvelle aquitaine, kansas → ks). Street words go through country-aware abbreviation maps (FR r → rue, bd → boulevard; US st → street). Three features: `ca_tset`, `ca_sort`, `ca_jacc`.
- **M1 result:** out-of-fold **0.9606 vs 0.9594** (fold 0 −0.0006, fold 1 +0.0023; US +0.0018). Fails the both-folds rule at stage 1.
- **Next:** stage 2 (`ens3gca`) decides. Test path: `features4` + `V3_CANON` in `src/pipeline.py`, off until validated.

## D23 — Decoy-weighted stage 2 (E-P1): marginal

- **What:** stage-2 LightGBM trained with decoy records weighted ×w. It is scored as-is, and in a world where every decoy is duplicated to mimic test's 2× decoy density.
- **Result:**
  - w=2: as-is −0.0004; doubled decoys +0.0009.
  - w=3: as-is −0.0008; doubled decoys +0.0010.
- **Decision:** not shipped. Superseded by D24.

## D24 — Density-checked model consensus (E-C1) → `output/sub6_consensus/`

- **Evidence (public repo mayankgoplani431-del, which reports leaderboard scores):**
  - E02 0.964; S3 0.956 despite higher validation.
  - Keeping only pairs both models predict: **0.969**.
  - Removed pairs were about 81% false: test decoys are near-copies with a shifted house number, at 2× train density.
- **Our label-free audit:**
  - Vote pattern per test pair over [v1 ens, v2 ens, v3 ens, v2 M1], compared with validation OOF on the same patterns.
  - Cell `111.` (all stage-2 ensembles accept, stage-1 rejects), US: **0.092 per S1 on test vs 0.018 on validation (×5.2)**; house-number mismatch 0.90 vs 0.63.
  - India: every cell ≈ ×1.0. France `111.`: 1.3–3.4× the US/India validation rates; 34% hn mismatch vs 1% in agreed France pairs.
  - Cause: stage-2 collective features are raw counts of competing records (`claims_s1`, `p1_sum_rec`, `n_twins`), so they shift with decoy density.
- **Estimate:**
  - Per-cell test precision = validation precision × (validation rate ÷ test rate). US `111.` ≈ 0.17.
  - Monte Carlo per-S1 F0.5 vs sub3d (LB 0.952): sub4d +0.0037; **sub4d − US `111.` +0.0087** (US + India only).
- **Decision:** `sub6_consensus` = sub4d minus `111.` pairs in US (61,315) and France (15,438). Matches ⊆ candidates (0 violations); validator PASS.
- **Unmeasurable:** France is a bet backed by the house-number signature only.

## D25 — Canonical addresses kept at stage 2; 7-model combination (E-C2) → `output/sub7_stack/`

- **Stage 2 with CA (`ens3gca`):** mean3 **0.9639 vs 0.9623** (0.9593 / 0.9676 vs 0.9579 / 0.9658; both folds up). `V3_CANON = True`. Test outputs: `sub5d` (ensemble) and `sub5b` (M1).
- **Plain vote stacker:** 7 voters, q per (country, vote pattern), cross-fitted. Scores **0.9630, below the best voter** (v5ens 0.9639) in both folds, so it was not used.
- **Shipped rule (`sub7`):**
  - Start from v5ens; remove pairs whose pattern is over-represented on test, where q × min(1, val rate ÷ test rate) < T (0.762). The validation rate uses an upper bound (n + 2√n + 3).
  - On validation this is exactly v5ens; removals happen only on test.
  - Removed: US 75.7k, India 35.2k, France 28.2k. Validator PASS against its own candidates.
- **Label-free estimate vs sub3d (LB 0.952), a ranking not a score:** sub4d +0.0033, sub5d raw +0.0027, sub6 +0.0087, **sub7 +0.0100**.
- **Disk:** the disk filled and processes were killed. Identical `candidate_pairs.tsv` copies in `output/` are now hard links.

## D26 — Twin features collapse at test decoy density (E-P2) → stage 2 without twins

- **World x2 (labelled):**
  - Every decoy that is a candidate of a validation S1 is duplicated; twin components and all collective features are recomputed.
  - Stage-2 LightGBM: **all 0.9628 → 0.8609**; no_counts 0.9630 → 0.8539; **no_twins 0.9606 → 0.9505**; no_s1_side 0.9591 → 0.9528.
- **Why:** decoy businesses have their own copies. They agree as twins, and in training twin agreement meant a true match.
- **Label-free test check (pairs per S1, test ÷ validation, US):**
  - stage-1 models 1.015–1.018; stage-2 ensembles with twins 1.036–1.063;
  - **no-twins ensemble 1.015**. India 1.011.
- The no-twins ensemble (`ens3gca_no_twins`, mean3 0.9620) → `output/sub5notwinsd_ens_ef/`.

## D27 — Vocabulary class of unmatched name tokens (E-T3): rejected

- **Idea:** decoys swap in a *real* word ("daniels hawk" → "hanners hawk" at the same address); true copies add typos seen nowhere else. On same-address pairs the match rate is 0.82 without a real-word extra vs 0.22–0.37 with one.
- **M1 result:** 0.9610 vs 0.9606 (fold 0 +0.0017, fold 1 −0.0003); gain share 0.1%. Existing name similarity already carries the signal.
- **Decision:** off. Code: `src/vocab.py`, `features5`, `V3_VOCAB = False`.

## D28 — Entity-group gains replace twins (E-G2)

- **Group of (s1, r):** the S1's other candidates that are near-copies of r (name ≥ 85, address ≥ 80 or empty, same house number when both present).
- **Features are gains, never counts:** best sibling minus self for name, address and p1. A duplicated decoy adds a gain of 0.
- **Stage-2 LightGBM:** **0.9620** (0.9569 / 0.9659) vs no_twins 0.9606 (0.9559 / 0.9644), both folds up. x2 world **0.9534** vs 0.9505.
- **Test:** 33.8% of pairs have a better sibling vs 35.5% on validation, so the features are density-stable.
- **Decision:** kept, `V3_STAGE2 = "no_twins_grp"`. Code: `src/groups.py`, decide stage chunked by S1.

## D29 — Legal-form agreement (E-T4): kept → FINAL `output/sub10notwinsgrpb_m1_ef/`

- **Why:** half of the train S1 (1.11M of 2.2M) share their normalised name with another S1. They are different businesses that differ only in stripped tokens ("Crandall Enterprises P.C." vs "... Inc"). The legal form was invisible to every feature.
- **Features:** `src/legal.py` on raw names: `lf_equal`, `lf_diff`, `lf_one_side`.
- **M1 result:** **0.9655 vs 0.9606** (0.9604 / 0.9689; precision 0.9871, recall 0.9304).
  - The dummy-column control reproduces the baseline exactly (0.9606).
  - Gain share is tiny because the features break ties between rival S1 in exclusive assignment.
- **Final file:** legal-form M1 alone (`model_m1f`, `feat6`). It is the best validated model and more density-robust than every stage-2 ensemble (US test/val pairs per S1 1.010, India 1.003).
- **Checks:** validator PASS; 0 matches outside candidates.
- **Not finished (disk-full crashes):**
  - stage 2 on legal-form M1 (`ens3gcal_no_twins_grp`);
  - E-T5 raw-name similarity (`src/rawname.py`, `experiments/e_t5_rawname.py`).
