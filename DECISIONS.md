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
