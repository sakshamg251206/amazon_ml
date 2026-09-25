# Amazon ML Challenge 2026: Business Entity Resolution. Deep Research Report

*Written 25 Sep 2026, about 20:30 IST: day 1 of a 72-hour window that closes 27 Sep, 23:59 IST.*

## How to read this report: provenance tags

Every factual claim carries a tag saying where it comes from. Do not treat a lower tier as a higher one.

| Tag | Meaning | Trust |
|---|---|---|
| **[OFF]** | Official problem statement, guidelines PDF, validator code | Ground truth for rules |
| **[COMP:name]** | Measured on the real data by a competitor, taken from their public repo. **We have not re-verified it.** | High when two or more teams agree; otherwise medium |
| **[COMM]** | A claim a community member or competitor made that is not backed by a measurement | Low |
| **[INF]** | Our own inference from the evidence | Reasoned, not measured |
| **[HYP]** | A hypothesis we have not tested. Section 15 gives the test for each. | Unknown |
| **[MATH]** | Derived exactly from the metric definition | Exact |

**Data status.** The dataset is not in this folder yet; it is gitignored in the linked repo. All dataset statistics below are therefore **[COMP]**. They come mainly from `Abhigyanv23`, `Epic021`, `vishwajitsarnobat` and `SanthoshReddy352`, which agree with each other wherever they overlap. The first research task after the data arrives is to re-verify them (section 15, E0).

**Sources inspected:**
- The linked repo `Sugandh-vI/Amazon-ML-challenge`: README, problem-statement PDF, guidelines PDF, `student_resource/` (README, `Documentation_template.md`, `validate_submission.py`).
- A GitHub search found about 230 repos created for this challenge in the last 24 hours. We shallow-cloned 91 of them and mined all 91. We read about 10 in depth.

---

## 1. Executive Summary

**What the problem is.** Test S1 has 1.73M clean "reference" businesses. There are about 10M noisy S2/S3 records. For each S1, output the set of S2/S3 records that are the same business. Each S2/S3 record belongs to **at most one** S1 [COMP ×4]. About 26% of S2/S3 records belong to none [COMP]. The metric is F0.5 per S1, averaged over all S1 [OFF].

**What the data is.** A programmatic generator, not organically collected data [INF, strong evidence]:
- Every S1 gets a sampled number of noisy copies. That distribution is identical across countries: 5.6% / 5.4% / 17.0% / 24% / 22% / 26% for 0 / 1 / 2 / 3 / 4 / 5+ copies.
- Copies are formatted in a source-specific style: S2 is uppercase, "registry" style; S3 is "directory" style with full state names.
- Each copy then gets random noise operations: typos, homoglyphs, accent injection, Indic transliteration, word shuffle, legal-suffix swap, domain-style names, number corruption, dropped address components, and state or region swaps.
- Distractors are added on top. **44% of them are constructed "neighbour businesses":** a copy of an S1 with a word added and the house number changed [COMP:Epic021].

**What everyone is converging on** [COMP, 91 repos]:
- Normalize and transliterate.
- Sparse TF-IDF retrieval, per country, run from S2/S3 toward S1.
- LightGBM on string-similarity, house-number and rank/margin features.
- Decode with "each record goes to its best S1 if p ≥ t".

The best documented versions of this pipeline score **0.97–0.99 on validation** and **0.93–0.96 on the public leaderboard on day 1** [COMP]. The top of the leaderboard will be separated by thousandths [INF].

**Where the remaining score is** (measured, not guessed):
1. **France.** France is 15% of test S1 and has no labels. Two teams' leaderboard arithmetic implies France scores about **0.85–0.91**, against about 0.97 for US/India [COMP:Abhigyanv23, SanthoshReddy352]. One team's validation *rose* +0.0007 while its leaderboard *fell* −0.011, because a feature broke on France.
2. **Neighbour negatives.** Even among records whose best S1 looks ≥92% similar, **14% are negatives** [COMP:Epic021]. That band decides precision.
3. **Recall ceiling of blocking at full density.** At full index size, recall falls to 0.97–0.988; on small samples it is about 0.997 [COMP:vishwajit]. The misses concentrate in non-Latin names (recall 0.954) and empty addresses (0.920).

**The least-exploited structure** [COMP: 0 of 91 repos use graph or cluster methods]. The problem has **entity-level structure**: each S1 has on average 3.5 copies spread across two independently noised sources. Nobody uses the copies' agreement with *each other*, and nobody tests whether distractors form their own "phantom" clusters. This is the most plausible untested edge. It is testable in about an hour once the data is here (E1–E3).

**A metric insight most teams state backwards** [MATH]. "F0.5 is precision-heavy, so when unsure don't merge" is true only for S1s with *large* true sets.
- For an S1 with one true match, **a miss costs 1.0 and a false positive costs 0.44.**
- The first predicted match for any S1 is almost always worth making.
- The confidence bar should *rise* with each additional match: p > k/(k + 0.25n).

This is why the best team found its optimal global threshold at **t = 0.02**. The metric rewards S1-centric, rank-dependent decisions, not one global cutoff.

**Your binding constraint** [measured on this machine]: an Apple M2 with 8 cores, **8 GB of RAM**, and 17 GB of free disk. The best competitors are running on 48-core / 160 GB clusters. The raw data is about 2.4 GB. Unpartitioned full-density blocking will not fit, so compute strategy is part of the solution (sections 11 and 17).

---

## 2. Official Problem Statement

### Layman version

Amazon has three phone books of businesses.
- **Book 1 is clean.** Each business appears once, spelled properly.
- **Books 2 and 3 are messy.** They contain the same businesses written badly: typos, "Pvt" instead of "Private", Hindi script instead of English, missing street names, the words in the wrong order. A business can appear several times in the same messy book. The messy books also contain businesses that are *not* in Book 1 at all.

For every business in Book 1, list every entry in Books 2 and 3 that is the same business. If there are none, say so. You are graded business by business, and wrongly claiming a match hurts more than missing one.

### Research version

- **Task.** This is **one-sided record linkage** into a clean reference set [OFF]. S1 is deduplicated. S2 and S3 are not deduplicated: an S1 can have up to 5 S2 and 6 S3 matches [COMP:Abhigyanv23].
- **Output** [OFF]: a function f: S1 → 2^(S2 ∪ S3).
- **Label structure** [COMP ×4]:
  - the true map is a partial function g: S2 ∪ S3 → S1 ∪ {∅}. Each record has at most one owner, and ownership never crosses countries. So the output is really a **partition of the S2/S3 records** labelled by S1, plus an "unowned" bucket.
- **Fields** [OFF]: `entity_id`, `business_name`, `business_address`, `country`. That is all. There are no phone numbers, websites, categories or coordinates.
- **Metric** [OFF]: macro-averaged F0.5 over test S1. A singleton scores 1 for an empty prediction and 0 otherwise.
- **Constraints** [OFF]:
  - No external data: no geocoding, registries, APIs or internet augmentation.
  - The final model must be MIT or Apache-2.0 licensed and have ≤8B parameters.
  - At most 5 leaderboard submissions per day, for 3 days.
  - The public leaderboard is a subset of test; the private leaderboard is the rest and decides the ranking.
  - Required deliverables: `matching_results.tsv` (scored) and `candidate_pairs.tsv`. The candidate file is audited for "recall ceiling" and "reduction ratio". It must be the *final* candidate set that the model scores, and the matches must be a subset of it.
  - Reproducible code package plus a methodology document. The top-100 packages are reviewed.
- **Rules competitors cite that are not in the official files** [COMM, from Epic021; verify on the portal]:
  - ties go to the earlier submission;
  - the top 500 at the 48-hour mark get +$100 in AWS credit;
  - participants get $200 in AWS credit.
- **Validator facts** [OFF, from reading `validate_submission.py`]:
  - A matched ID that does not exist in test only *lowers your score*; it does not cause a rejection.
  - A missing S1 row, a duplicate row, a duplicate ID inside a list, or an S1 ID inside a list **will cause a rejection**.
  - The validator warns, but does not fail, if a match is not in `candidate_pairs.tsv`. The final-package audit will still check it.
- **Contradiction in the official material** [OFF]: the guidelines PDF says the document should be "1–2 pages", while the README says "no page limit". Aim for about 2 dense pages.

---

## 3. Dataset Anatomy

All figures are **[COMP:Abhigyanv23 EDA]** unless tagged otherwise. They are cross-checked against Epic021, vishwajit and SanthoshReddy wherever those overlap.

### 3.1 Sizes and composition

| | Train | Test |
|---|---|---|
| S1 | 2,206,821 (US 60.0% / India 40.0%) | 1,732,544 (**India 46.8% / US 38.3% / France 15.0%**) |
| S2 | 5,034,616 | 4,887,273 |
| S3 | 5,285,603 | 5,082,316 |
| S2+S3 per S1 | **4.68** | **5.75** (France 5.53, India 5.82, US 5.76) |
| Raw size on disk | about 1.2 GB | about 1.16 GB |

**The country mix shifts between train and test.** US is 60% of train but only 38% of test. Any score averaged over the train mix will not equal the leaderboard, even with zero France error. Abhigyanv23 estimates this reweighting alone is worth about 0.006 [COMP].

### 3.2 Label structure (train ground truth)

- **Matches per S1:**
  - 0 (singleton): 5.58%
  - 1: 5.40%
  - 2: 17.0%
  - 3: 24.1%
  - 4: 21.9%
  - 5+: 26.0% (maximum 11, mean 3.46)
- **The distribution is identical in India and the US** to 0.1pp. That is a generator parameter, not a property of the real world [INF].
- **Of matched S1s:** 1.78M have matches in both S2 and S3, 143k only in S2, and 165k only in S3.
- **Each S2/S3 record matches at most one S1.** Four teams measured this independently. Matches never cross countries.
- **Unmatched ("distractor") records:** 26.6% of S2 and 25.4% of S3.
- **Estimated test match rate:** Epic021 matched the distribution of best-candidate similarity between train and test. From that it estimates 71–74% of test S2/S3 are true matches, about the same as train. The extra test records per S1 would then mean **about 4.1–4.2 true matches per S1 in test, against 3.46 in train** [COMP:Epic021, a model-based estimate].

### 3.3 Missingness and formatting by source

| | S1 | S2 | S3 |
|---|---|---|---|
| Empty address | 0% | 3.4% (test 2.65%) | 3.3% (test 2.68%) |
| Non-ASCII name | 0% in train, **2.4% in test** (French accents) | 15.2% | 11.5% |
| ALL-CAPS address | 0% | **63%** | 0% |
| ALL-CAPS name | 0% | 18.9% | 3.0% |
| Double spaces in name | 0 | 554k | 576k |
| Names of 2 characters or fewer | 0 | 704 (test: 10,166) | 9,275 (test: 19,910) |

- **Source style** [COMP:Epic021]:
  - **S2 is "registry" style:** uppercase, `H.NO`/`##` prefixes, 9% Indic-script names.
  - **S3 is "directory" style:** full state names ("Ohio", "Kansas"), Indic-script *state* names, `NULL`/`N/A`, "CITY" suffixes.
- **Short names rise in test** (S2 ≤2-character names go from 704 to 10k, e.g. `cc`×302, `sc`×359) [INF]. This is a train/test difference nobody has explained yet. It could be acronym-style names generated for France ("CC" = Comité/Club?). Check it.

### 3.4 Duplicates and name collisions

- **S1:** zero exact duplicates, but **686k S1 rows share a normalized name with another S1**. Examples: "primary care group" ×253, "pediatric group" ×222. Chains and branches are everywhere, so the name alone is weak evidence.
- **Test S1** is dominated by French templates: "bordeaux club sarl" ×205, "nantes club sarl" ×157, "lille club sas" ×122.
- **Share of S1 whose core name appears ≥5 times:** France 28.5%, India 37.1%, US 20.8% [COMP:Abhigyanv23 country_diag].
- **Exact duplicates within S2** (same name, address, country): 25.9k in train. [HYP] These are the same entity or both distractors ("same fate"). Test this; if true, it is a free constraint.

### 3.5 How true pairs differ

Sources: Abhigyanv23 on 300k pairs and Epic021.
- **Light-normalized name identical:** 21.8% of pairs. **Address identical:** 8.4%.
- **Mean token Jaccard** (the share of tokens the two strings have in common): name 0.615, address 0.596.
- **14.5% of true pairs share no name token.** That is caused by transliteration, domain-style names ("konkandata.com") and renames. Only 0.01% share no name *and* no address token.
- **82% of positives have name and address similarity both ≥80.** 3.3% are renamed (name similarity <50). 1.9% have an empty address.
- **House number relationship in positives** [COMP:Epic021]:

  | Relationship | Share |
  |---|---|
  | Equal | 65% |
  | Candidate drops the number | 12% |
  | Numbers **differ** (typos: 31→30, 1600→160, 13741d) | 10% |
  | Candidate adds a number | 10% |

- **Words added in positives:** dba, aka, formerly, com, www, center, services, society, lp, smt, sri, shri, mr, `#35740`-style tags, brackets. **Words dropped in positives:** legal forms.

### 3.6 Constructed negatives: the core difficulty

Source: Epic021.
- **44% of unmatched records have an address ≥90 similar to some S1,** and 36% also have a similar name. Only 16% are unlike any S1.
- **House numbers in these near-address negatives:** different 88.2%, equal 2.5%, candidate adds a number 8%.
- **They add a business-type or geographic word:** holdings, group, enterprises, industries, exports, public, overseas, ventures, north/south/east/west, metro, central, valley, harbor. French examples: Développement, Participations.
- **Among records whose best S1 is ≥92 similar, 14% are negatives.**
- **The same pattern hits France.** SanthoshReddy's v2 lost 0.011 on the leaderboard by merging 35k French "sibling businesses": same core name plus a descriptor, same street, different house number.

### 3.7 France (test only, no labels)

Sources: Abhigyanv23, Epic021, SanthoshReddy.
- **Geography:** about 15 cities in 3 regions: Hauts-de-France (Lille, Roubaix, Tourcoing, Dunkerque, Calais), Nouvelle-Aquitaine (Bordeaux, Pessac, Mérignac, La Teste-de-Buch, Lège-Cap-Ferret) and Pays de la Loire (Nantes, Saint-Nazaire, Saint-Herblain, Pornic, La Baule). That makes **extremely dense streets.**
- **Region and department swaps:**
  - S1 uses the region.
  - S2/S3 swap it for the department: Nord / Pas-de-Calais ↔ Hauts-de-France, Gironde ↔ Nouvelle-Aquitaine, Loire-Atlantique ↔ Pays de la Loire.
  - They also use region codes `hdf` / `naq` / `pdl`.
  - The state is found in only 66–68% of French S2/S3 addresses, against 91% for India and 100% for the US.
- **Street abbreviations:** R, R., AV, BD, IMP, ALL, RTE, PL, QUAI, FBG, N°. **Number suffixes:** BIS, TER.
- **Name vocabulary:** association-like (Club, Amicale, Comité, Ecole, Union sportive) plus legal forms (SARL, SAS, SASU, EURL, SCI, SA). [INF] The base names look like association names with a random legal form appended.
- **Blocking top-1 similarity:** France mean 0.983, India 0.943, US 0.931. **France *looks* easier, and that is the trap.** High surface similarity combined with dense, templated neighbours produces confident false merges.

### 3.8 Leakage audit (what has been checked)

- **IDs, row order and file position:** no signal [COMP:Epic021].
- **Train and test share no IDs** [COMP:Abhigyanv23].
- **34% of test S1 names also appear in train** [COMP:Epic021]. This is vocabulary reuse by the generator, not entity overlap. [INF] It still means that token-level lexicons learned on train (transliteration, abbreviations) transfer well to US/India test.
- **Our position:** there is no known exploitable leakage. We still re-check ID-number correlation between each S1 and its matches in E0, because it costs about a minute.

---

## 4. Data-Generation Hypotheses

The evidence strongly suggests a **three-stage synthetic generator** [INF]. Knowing its stages tells us which invariants the model should learn.

1. **Base entities (S1).** These are real-looking seeds.
   - US addresses look real and names are synthesized (surname + word + legal form: "Yazzie Empire Enterprises LLC", "Munoz Onfolio LLC").
   - Indian records look like a company registry ("Silver Foundation Private Limited", "F-25, Asha Complex, Sec-18…").
   - French names look like associations with an appended legal form ("Shidokan Ecole SARL").
   - [HYP] Seeds come from public-registry-style lists. This is irrelevant for us because external lookup is banned, but it explains the name collisions.
2. **Positive copies.** The copy count is drawn from a fixed distribution (section 3.2). Each copy is assigned to S2 or S3 (at most 5 S2 and 6 S3 per S1), rendered in that source's template, and then passed through random noise operations (the catalogue below).
3. **Negatives:**
   - (a) **"Neighbour" constructions from an S1:** add a descriptor word and change the house number (44%).
   - (b) Records similar to an S1 in a weaker way.
   - (c) Unrelated records (16%).
   - (d) [HYP] **Singletons are S1s that got zero copies, and they may still attract neighbour negatives.** Two things are unknown: how many neighbours each seed gets, and whether a neighbour itself gets *multiple copies* across S2/S3 (a "phantom entity"). **Nobody has checked the second question** (E1).

**Noise-operation catalogue** (observed in published examples):

| Field | Operation | Example |
|---|---|---|
| Name | Case change, extra spaces | `YAZZIE EMPIRE  ENTERPRISES` |
| Name | Legal form dropped or swapped | `Pvt`↔`Private`, `(LLC)`, `[Limited]` |
| Name | Generic suffix added | `… Center`, `… Services`, `… Society` |
| Name | Word shuffle | `LIMITED SUNBEAM EAE`, `Estate Sunbeam Limited` |
| Name | Typo: swap, insert, delete | `Caare`, `Dentsiryt`, `Onfnoloi`, `Fsotoria` |
| Name | Homoglyph or digit swap | `5unbeam`, `Banga1ore`, `ATLANTlC` |
| Name | Accent injection | `Lìmited`, `Fóundation`, `TRÁNSALTA` |
| Name | Domain form | `tuckermetrotransalta.com`, `>> shreethreads.com` |
| Name | Tag suffix | `#35740` |
| Name | Indic transliteration (whole name) | Devanagari, Kannada, Tamil, Telugu |
| Name | Trade name | `Fluxmira trading as Lopez Select Motors`, aka / dba / fka |
| Address | Component dropped, reordered or repeated | `OH, EUCLID AVE, EUCLID` |
| Address | Abbreviate or expand (street, state) | `Ter`↔`Terrace`, `KS`↔`Kansas`, `UP`↔`Uttar Pradesh` |
| Address | State in native script | `उत्तर प्रदेश` |
| Address | **State swap** | Telangana→Andhra Pradesh (historical split), department↔region |
| Address | Locality substitution | `LUCKNOW HQ REGION`, `Nangloi`→`Kanjhawala` |
| Address | Number edits | Leading zero `02A/3`, suffix letter `3906a`, truncation, `B-d/382` |
| Address | Prefixes | `Door No ##435`, `No 276`, `H.NO` |
| Address | Blank | About 3% |

**Implications** [INF]:
- **Nearly every operation is country-agnostic.** Case, typos, word shuffle, number edits, suffix insertion and dropped components are applied the same way everywhere.
- **Only the lexicons are country-specific:** legal forms, street abbreviations, state names, transliteration.
- So France should be solvable by a model that learned *operations* rather than *tokens*. That is exactly the generalisation the hidden country tests.

---

## 5. Evaluation & F0.5

### 5.1 The formula, simplified

With P = TP/|pred| and R = TP/|true| for one S1:

> **F0.5 = 1.25·TP / (|pred| + 0.25·|true|)** [MATH]

Singleton special case: an empty prediction scores 1, and anything else scores 0. The PDF example checks out: 1.25·2 / (3 + 0.5) = 0.714.

**Reading the formula:**
- Every predicted record costs 1 unit in the denominator.
- Every true record costs 0.25 of a unit.
- The numerator only grows with correct predictions.
- The score is macro-averaged, so **every S1 counts equally** whether it has 1 true match or 11.

### 5.2 What each mistake costs [MATH]

| True matches | All correct | Miss 1 | 1 extra false positive | Miss 1 and 1 false positive | Predict empty |
|---|---|---|---|---|---|
| 0 (singleton) | 1.0 | n/a | **0.0** | n/a | 1.0 |
| 1 | 1.0 | **0.0** | 0.556 | 0.0 | 0.0 |
| 2 | 1.0 | 0.833 | 0.714 | 0.5 | 0.0 |
| 4 | 1.0 | 0.938 | 0.833 | 0.75 | 0.0 |
| 8 | 1.0 | 0.972 | 0.909 | 0.875 | 0.0 |

**Conclusions:**
1. **Precision is *not* uniformly 2× as important.**
   - At 4 true matches, a false positive costs 0.167 and a miss costs 0.062: **2.7×** (not 2×).
   - At 1 true match, **a miss costs 1.0 and a false positive 0.444**, so recall matters more.
   - For a singleton, a false positive costs everything.
2. **Predicting empty for a non-singleton scores 0.** The empty prediction is the most extreme bet you can make.
3. **The most expensive mistakes are:**
   - any match on a true singleton (5.6% of S1);
   - an empty prediction on a 1- or 2-match S1 (22% of S1);
   - a false positive on a small-set S1.

   False positives on large-set S1s are relatively cheap.

### 5.3 The metric-optimal marginal rule [MATH]

Suppose we have already predicted k correct matches for an S1 with n true matches. Adding one more candidate that is correct with probability p raises the expected score **if and only if**

> **p > k / (k + 0.25·n)**

| True matches | Required p for the 1st, 2nd, 3rd … prediction |
|---|---|
| n = 2 | 0, **0.67** |
| n = 4 | 0, 0.50, 0.67, **0.75** |
| n = 6 | 0, 0.40, 0.57, 0.67, 0.73, **0.77** |

**Interpretation:**
- **The first match is nearly free,** so long as the S1 is not a singleton.
- Each later match needs increasing confidence.
- **One global threshold is metric-suboptimal:** the right bar depends on the candidate's *rank within its S1* and on the expected set size n.

**Singleton or first match.** Take an S1 whose best candidate is correct with probability p and whose chance of being a singleton is q. Predicting {c} is worth about p·1.25/(1 + 0.25n), which is about 0.67p for n ≈ 3.5. Predicting empty is worth q. So **predict unless 0.67p < q**. With a 5.6% singleton prior, that means predict unless p < about 0.08.

This matches vishwajit's empirical finding [COMP] that the best global threshold, after one-to-one assignment, is **t = 0.02**: 0.9896 at t = 0.02, 0.9746 at t = 0.5 and 0.9644 at t = 0.75. Teams sweeping t over 0.3–0.9 are leaving points on the table.

### 5.4 Contrary evidence: expected-F0.5 decoding

**Caveat** [COMP]. Two strong teams implemented full **expected-F0.5 subset selection**: choose each S1's set to maximize E[F0.5] given calibrated probabilities. It did *not* beat a tuned global threshold:

| Team | Expected-F0.5 | Tuned threshold |
|---|---|---|
| vishwajit | 0.970 | 0.9896 |
| SanthoshReddy | 0.9705 | 0.9710 |

[INF] The likely causes:
- **Probabilities are not calibrated at the S1 level.** Pair probabilities are correlated within an S1.
- **n is unknown,** and misses from blocking inflate it.
- **Test n is larger than train n** (about 4.1 vs 3.46).

So metric-aware decoding is a *hypothesis worth testing with better calibration* (E7), not a guaranteed win.

### 5.5 Candidate generation and the recall ceiling

- A true match missing from the candidates can never be predicted.
- The "oracle F0.5" (a perfect matcher on your candidates) is the ceiling. Measured values:

  | Team | Oracle F0.5 | Blocking recall | Candidates per S1 |
  |---|---|---|---|
  | Abhigyanv23 | 0.980 | 0.948 | not reported |
  | SanthoshReddy | 0.9855 | 0.958 | 54 |
  | vishwajit | not reported | 0.988 | 69 per record |

- **Small development indexes lie.** Blocking recall was 0.997 against 8.6k S1 but **0.967 against the full 883k Indian S1** [COMP:vishwajit]. Any validation that does not block against the *full* S1 pool overstates both recall and precision.

---

## 6. What Competitors Are Doing

### 6.1 Landscape

About 230 repos were created in the last 24 hours; we cloned 91. The table counts repos that *mention* each technique in code or docs. A mention is an upper bound on use, not proof of it.

| Technique | Repos (of 91) | Notes |
|---|---|---|
| LightGBM | 57 | The default matcher |
| XGBoost / CatBoost | 33 / 21 | Often ensembled with LightGBM |
| rapidfuzz string similarities | 37 | ratio, token_set, Jaro-Winkler |
| TF-IDF (char n-gram) | 37 | 12 use `sparse_dot_topn` for top-k search |
| House or street number features | 37 | |
| Embeddings (e5 / bge / MiniLM) | 28 | Mostly for blocking; see 6.3 |
| faiss | 22 | |
| Calibration (isotonic) | 29 | |
| Margin or second-best features | 27 | |
| Transliteration (anyascii / unidecode) | 26 | About 4 learn an Indic→Latin dictionary from training pairs |
| Phonetic (soundex / metaphone) | 22 | |
| **One-to-one / exclusive assignment** | 18 | The key structural trick |
| Cross-encoder | 17 | Mostly planned, not shipped |
| Expected-F0.5 decoding | 14 | |
| LLMs (Ollama / Qwen / GPT) | 10 | Mostly in plans |
| splink / dedupe / recordlinkage | 10 | |
| Leave-one-country-out validation | about 8 | |
| **Pseudo-labels / test-time adaptation** | **2** | |
| **Graph or cluster methods** (networkx, connected components, union-find) | **0** | |
| **S2↔S3 twin linking** | **1** (Epic021 lists it as a stretch goal) | |

### 6.2 Serious implementations

| Repo | Architecture | Validation (honest?) | Public LB | Notable |
|---|---|---|---|---|
| **vishwajitsarnobat** | Per-country TF-IDF: char name + word address + combined view (k = 50) + exact keys; 41 features; LightGBM; record → best S1 if p ≥ 0.02 | **0.9896 at full density** (honest); dev sample 0.991 | Not shown | Best documentation. Found the density effect. Margin over second-best carries 50% of the gain. Embeddings rejected (LSA recall 0.70). Needs a 48-core cluster, about 5.5 h per run |
| **SanthoshReddy352** | Blocking over all train S1; LightGBM in stages (stage 2 "group consistency", stage 3 refiner); one-owner; house-number safeguard for unseen countries | 0.9706 → 0.9821 across versions | **0.680 → 0.962 → 0.951 → 0.957** | Only team with a full validation-vs-leaderboard history. Uses France-only leaderboard probes. Drift bug from token frequency. Fits in 16 GB with hashed keys |
| **Abhigyanv23** | Blocking per (country, state), IDF cosine, k = 20; LightGBM; decision tuned out-of-fold | Holdout 0.948 (441k S1) | **0.931** | Most careful EDA. Blocking-miss taxonomy. Implied France about 0.87. Blocking takes 815 s for 441k S1 |
| **Epic021** | Playbook only: two-direction TF-IDF, 60–100 features, log-odds of added words, number relationships, exclusive assignment plus expected-F0.5 | Not reported | Not shown | Best data-generation analysis (neighbour negatives). Plans France log-odds without labels |
| **Siva402-ai** | "Record-to-entity assignment" (reversed direction), learned Indic dictionary | 0.978 | Not shown | An earlier version reported 0.85 on a *sampled* corpus (admitted inflation) |
| VeerbhadraMahant | 1:1 assignment | 0.936 | Not shown | |
| Aamod007 | XGBoost + LightGBM + CatBoost blend | "0.980", but on only 1,498 S1 | Not shown | **Inflated by sample size** |
| tingirikar | Country-partitioned, 4 channels, 22 features, LightGBM | "0.99934", **pair-level** | Not shown | **Wrong metric** |
| vishalkumar-ai25 / Outliers-ML / bhargavgajare | Assorted | 0.66 / 0.27 / 0.77 | Not shown | Blocking or decoding failures |

### 6.3 Negative results others have already paid for [COMP]

- **Dense embeddings for blocking:**
  - A 256-d LSA embedding reached 0.70 recall at k = 50, against 0.97 for sparse TF-IDF.
  - vishwajit's reason: "the signal is in exact and near-exact token overlap".
  - Encoding 12M records would take 10+ hours on a laptop.
- **The learning curve saturates early:** 874 S1 → 0.9887, 17k S1 → 0.9912. About +0.0005 per doubling. *The blocking index must be full-size; the training set need not be.*
- **"Crowding" / density features overfit to train density** and broke France (−0.011 on the leaderboard).
- **Raw token-frequency features drift between train and test** and flip tree splits. SanthoshReddy's v4b would have dropped 46k correct India pairs.
- **Hard-blocking on state loses about 0.6% of true pairs** to state-swap noise (the Telangana/AP split). Partitioning must allow for it.

---

## 7. The Standard / Obvious Solution

If 1,000 teams used AI assistants, this is what most would build [INF, from the landscape above]:

```
normalize (lowercase, NFKC, anyascii, legal-form & street-abbrev maps, strip Door No/H.NO)
  → per-country sparse TF-IDF top-k retrieval (char n-gram name, word/char address, combined)
    ∪ exact keys (space-free name, house#+street, postcode+name)
  → ~40 pairwise features (rapidfuzz name/addr, number match/conflict, legal-form, blocking cosines,
    rank / gap / margin of this pair among the record's candidates)
  → LightGBM (often + XGB/CatBoost blend)
  → each S2/S3 → argmax S1; keep if p ≥ t (global t swept on a holdout)
  → write TSVs, validator
```

**The tiers are about 0.10 apart:**

| Tier | Score | What it does |
|---|---|---|
| Naive | 0.66–0.94 | Samples the index, uses per-pair thresholds, no exclusivity |
| Competent | 0.95–0.96 on the leaderboard | Exclusive assignment, full-density blocking |
| Refined | 0.96–0.97 on the leaderboard | + margin features, low threshold, France safeguards |

**Where many AI-assisted teams will converge** [INF]:
- The refined tier. The leading repos' docs share the same design vocabulary: "one-owner", "margin over second-best", "house-number conflict".
- This is the attractor that LLM assistants plus a careful EDA produce.
- **The leaderboard top will be crowded around 0.96–0.975.**

---

## 8. Failure Modes of the Standard Solution

These are found in data and measurements, not imagined.

| # | Failure mode | Evidence | Size |
|---|---|---|---|
| F1 | **Neighbour or sibling negatives in the high-similarity band** | 14% of records whose best S1 is ≥92 similar are negatives [Epic021]. France sibling merges cost −0.011 on the leaderboard [SanthoshReddy] | The main precision loss |
| F2 | **Unseen-country collapse** | Implied France F0.5 about 0.85–0.91 vs about 0.97 [Abhigyanv23, SanthoshReddy]. Leave-one-country-out US→India: −0.061 [SanthoshReddy] | Up to about 0.015 of total leaderboard score (15% × 0.1) |
| F3 | **Non-Latin names miss blocking** | 21k of 57.6k misses at k = 50 [Abhigyanv23]. Recall 0.954 vs 0.988 [vishwajit] | About 1% of India pairs |
| F4 | **Empty-address records** | Recall 0.920 [vishwajit]. Examples: `Delacruz Conehctcilut` with a blank address; the name is the only evidence and is typo'd | About 3% of records |
| F5 | **Domain-style names** (`konkandata.com`) and address-only overlap | 13.3k misses [Abhigyanv23] | Fixed by space-free keys |
| F6 | **State swaps break partitioned blocking** | Telangana→AP; France department↔region; 9.6k misses | About 0.6% |
| F7 | **Small-sample validation inflation** | Recall 0.997 on a sample vs 0.967 at full density [vishwajit]. Aamod's "0.98" was measured on 1.5k S1 | Wrong decisions and a wrong threshold |
| F8 | **Density-dependent features and threshold transfer** | Crowd features broke France. Test has 5.75 records per S1 vs 4.68, and train S1 2.2M vs test 1.73M | Silent miscalibration |
| F9 | **Global threshold vs per-S1 marginal value** | Section 5.3. Teams sweeping 0.3–0.9 miss the t ≈ 0.02 optimum | Several thousandths |
| F10 | **Train prior on set size used for test** | Test has about 4.1 true matches per S1 vs 3.46 [Epic021 estimate] | Under-prediction if decoding uses a prior on n |
| F11 | **Chains:** same name at different branches | 686k duplicate S1 names; "primary care group" ×253 | Handled only by address and number |

---

## 9. Hidden Structure & Opportunities

This section asks what information exists that people are not exploiting.

### 9.1 Entity-level structure: copies agree with each other [HYP, strongest lead]

- An S1 has on average 3.5 copies (4.1 in test). They are spread over **two sources with independent templates**, and S2 alone can hold up to 5 copies.
- Today's pipelines score each record against S1 **in isolation**. Nobody (0 of 91 repos) uses **record-to-record** evidence within S2 ∪ S3.

**What copy agreement can add:**
- **Recall on hard records.**
  - A Devanagari-name record with an address identical to a Latin-name twin can inherit the twin's S1.
  - An empty-address record with a name identical to a twin that has an address can do the same.
  - These are exactly failure modes F3 and F4, the residual blocking misses.
- **Precision against phantoms.**
  - Suppose neighbour negatives also come as *several* copies: "UD International Holdings" at number 94, appearing in both S2 and S3 with the same changed number. Then **a group of records that agree with each other and disagree with S1 is evidence of a separate entity.**
  - A real typo in the number is idiosyncratic: the other copies carry the correct number.
  - This gives a feature the pairwise model cannot compute: *"how many other records share my deviation from S1?"*
- **The key unknown:** do distractors come in clusters? That is experiment E1 (about 30 minutes with ground truth). If phantom clusters exist, this is probably the single biggest precision lever left. If they do not, twin linking is still a recall lever (E2).

### 9.2 Learn the generator's operations, not its tokens [INF]

The catalogue in section 4 is finite. A pair can be described by *which operations turn S1 into the record*:
- number relation: equal / dropped / added / typo-distance / substituted;
- words added (by category) and words dropped (legal forms);
- case change, shuffle, domain form, transliteration.

**Positives and neighbour negatives are produced by different operation mixes** ("add descriptor + change number" = neighbour).

Features written as *operation indicators* rather than raw similarity or token identity:
- (a) should transfer to France, because the operations are country-agnostic;
- (b) are the probabilistic-linkage (Fellegi–Sunter) view: each comparison outcome has an m-probability (how likely under "match") and a u-probability (how likely under "non-match");
- (c) are learnable per source (S2 ops ≠ S3 ops).

Epic021 plans part of this with log-odds of added words. The token-identity version will *not* transfer to France ("Développement" ≠ "holdings"). **The category or IDF-bucket version might.**

### 9.3 Sources as independent noisy channels [INF]

- S2 and S3 apply different templates: uppercase/registry vs directory/full-state.
- "Source" should be a feature, and cheaper still, used as **interaction context**: a missing state means something different in S3 (full-state style) than in S2.
- Two records from *different* sources that agree on a detail both of them changed from S1 are strong evidence of a phantom (9.1). Agreement between them on the S1 value is strong positive evidence.

### 9.4 Exclusivity is a constraint on both sides [INF]

- Records → S1 is at most one. Everyone uses this.
- Less used: the *S1 side* has a known set-size distribution with explicit caps (S2 ≤ 5, S3 ≤ 6 per S1 in train).
- Joint decoding means maximizing expected macro F0.5 subject to each record having at most one owner. That is a **constrained assignment problem**. It can be solved greedily or with Lagrangian relaxation (price each record's exclusivity, then optimize each S1 separately).
- It fuses sections 5.3 and 5.4. Treat it as an experiment (E7), since two teams found naive expected-F0.5 worse.

### 9.5 Test-set statistics as unlabeled supervision (for France) [INF]

France has 1.43M unlabeled records. Nothing stops us estimating France's m/u-probabilities for operations **by EM (expectation-maximization) on the France pairs themselves**. EM is the classic unsupervised Fellegi–Sunter fit: guess match probabilities, re-estimate parameters, repeat. Alternatively, use self-training from very-high-confidence pairs.

- Epic021 plans pseudo-labels of this kind: "address+number equal" as stand-in positives, "number differs" as stand-in negatives.
- Only 2 of 91 repos mention pseudo-labels.
- **Rules:** fitting on unlabeled test *inputs* is standard, and blocking already does it with IDF. Disclose it in the documentation.

### 9.6 Uncertainty is informative [INF]

- vishwajit's test smoke run: France has 53% of best-probabilities ≥ 0.99, against 35% for India.
- Low-confidence bands look similar across countries: same name and street, different house number.
- That band is small and **could justify an expensive second-stage model only there**, such as a cross-encoder or a phantom-cluster check, without paying for 700M pairs.

### 9.7 Duplicate records within S2 [HYP]

- Train S2 has 25.9k exact duplicate (name, address) pairs and 68k light-normalized duplicates.
- If duplicates always share fate (the same S1 or both unmatched), collapse them before scoring. That saves compute and adds a consistency constraint. Cheap to verify (E4).

---

## 10. Unseen-Country Analysis

**What changes in test:**
- France is 15% of test S1 and 14.4% of S2/S3.
- The India share rises from 40% to 47%, and the US share falls from 60% to 38%.
- Test S1 names contain 2.4% non-ASCII (accents).
- Short or acronym names in S2/S3 jump from 704 to 10k (S2) [COMP].

**Leaderboard evidence** [COMP]:

| Team | Local | Public LB | Implied France |
|---|---|---|---|
| Abhigyanv23 v001 | 0.948 (US 0.965, IN 0.923) | 0.931 | about 0.87 |
| SanthoshReddy v1 | 0.9706 | 0.962 | about 0.91 |
| SanthoshReddy v2 (+crowd features) | 0.9710 (+0.0007) | 0.951 (**−0.011**) | about 0.85 |
| SanthoshReddy v3 (France house-conflict safeguard, crowd features removed) | 0.9748 | 0.957 | Not reported |

**Leave-one-country-out proxies:**

| Team | Direction | Blind score | In-domain score | Gap |
|---|---|---|---|---|
| SanthoshReddy | US → India | 0.902 | 0.962 | −0.061 |
| SanthoshReddy | India → US | 0.962 | 0.976 | −0.013 |
| vishwajit | US → India | 0.982 | 0.988 | −0.006 |

[INF] The gap depends heavily on how country-agnostic the features are.

**What Amazon is testing** [INF]:
- Whether the solution learned **transferable matching logic** (operations, number relationships, rank structure) or **memorized lexicons and densities** of US/India.
- France keeps the generator but changes:
  1. the lexicon: legal forms, street abbreviations, region/department names, a new descriptor vocabulary for neighbours;
  2. the density: 15 cities, so streets are about 3× more crowded;
  3. the name distribution: templated association names with heavy collisions.

**What breaks** [COMP + INF]:
- Features that depend on token identity: log-odds of added words, learned abbreviation maps.
- Features that depend on density: crowding, raw candidate counts, token frequency.
- Thresholds tuned on the US/India mix.
- State partitioning when only 66% of French records carry a region.

**What transfers:**
- Features that normalize themselves: rank, margin over second-best, gap to best.
- Number-relationship features.
- Case, shuffle and typo robustness.
- Operation indicators bucketed by IDF, not by token.

**Generalization opportunities, cheapest first:**
1. A hand-written French lexicon: legal forms; R/AV/BD/IMP/ALL/RTE/PL/FBG; BIS/TER; department↔region and hdf/naq/pdl. This is allowed as domain knowledge [COMM:Epic021, consistent with the rules]. Document it.
2. Strip country-dependent features. Gate every feature with leave-one-country-out *in both directions*.
3. **Label-free checks per country on test predictions:**
   - predicted singleton rate (expect about 5–7%);
   - matches per S1 (expect about 3.5–4);
   - share of records assigned (expect about 70%).

   France deviating from US/India is a red flag.
4. **Leaderboard probes that change only France rows.** The public subset is fixed, so ΔLB = (France share of public) × ΔF_France. One submission then isolates France. SanthoshReddy already does this. Spend at most 2–3 of our submissions on it.
5. Unsupervised estimation of France operation probabilities (section 9.5).

---

## 11. Alternative Problem Formulations

Which of these actually fit *this* data:

| Formulation | Fits? | Why |
|---|---|---|
| Pairwise classification | Yes, as the core | Nearly every repo. Necessary, not sufficient |
| **Retrieval + ranking (record → S1)** | **Yes, strongly** | Each record has at most one owner, so this is multi-class ranking over its candidates. Margin/rank features are this formulation leaking into the classifier. A LambdaRank objective per record is a cheap variant [INF] |
| **Constrained assignment / structured prediction** | **Yes** | Exclusivity plus set-level F0.5 is a constrained optimization (sections 5.3 and 9.4) |
| **Graph / collective ER (clustering S2 ∪ S3)** | **Plausible, untested, 0 repos** | Copies of one S1 form clusters across two independent channels; phantom clusters are a possibility (section 9.1) |
| Probabilistic linkage (Fellegi–Sunter + EM) | Partly | Best suited to France, where there are no labels. Operation-level m/u-probabilities |
| Metric learning / dense embeddings | Weak for blocking | Recall 0.70 vs 0.97 sparse [vishwajit]. Maybe as an extra retrieval channel for non-Latin names only |
| Cross-encoder (≤8B, MIT/Apache) | Narrow | Only on the uncertain band. Remaining errors are "information-limited" [vishwajit], e.g. same-name chains with empty addresses |
| LLM judging | Poor fit | 10M records. The signal is numeric and character-level; latency and cost are prohibitive on your hardware |
| Anomaly detection | Weak | Distractors are not anomalies; they are constructed to look normal |
| Weak supervision | For France | Pseudo-labels from rule-based stand-ins (section 9.5) |

---

## 12. Novel Solution Directions

These are not ranked. Each is a genuinely different conceptual approach.

### A. Refined pairwise + exclusive assignment (the converged baseline)

| | |
|---|---|
| **Why it might work** | It is already proven. 0.99 at full density on validation |
| **What it addresses** | The bulk of the score |
| **Risk** | You land exactly where everyone else does |
| **Complexity** | Dominated by blocking: O(records × shared-term frequency). The test run is about 28 h on 8 laptop threads unpartitioned, about 15–40 min when partitioned by (country, state) [COMP] |
| **Difficulty** | Medium |
| **Robustness** | High on US/India, medium on France |

**It is necessary as the foundation for every other direction.**

### B. Entity-level collective resolution (twin and phantom graph)

| | |
|---|---|
| **Idea** | Link S2/S3 records to *each other* with a cheap high-precision rule or model. Build connected components or a similarity graph. Add cluster-level features (copies agreeing with me, sources in my cluster, share of my cluster that agrees with S1 on the number). Let records inherit candidate S1s from their twins |
| **What it addresses** | F1 (if phantom clusters exist), F3, F4 |
| **Evidence** | Copies average 3.5 per S1 across two independent channels. Zero repos exploit this |
| **What could go wrong** | Distractors may be single copies, which kills the precision half. Merging errors in the twin graph would propagate: transitive closure is dangerous, so use features, not hard merges. Adds an S2/S3 × S2/S3 blocking pass |
| **Complexity** | One more retrieval pass within country; small |
| **Difficulty** | Medium |
| **Robustness** | Should transfer to France, because it is structural |

### C. Generator-inversion / noisy-channel likelihood

| | |
|---|---|
| **Idea** | Represent each pair as the most likely *edit script* (operations from section 4). Learn P(script \| copy) and P(script \| neighbour) per source. Use the log-likelihood ratio as features, or as the whole scorer. Equivalently: Fellegi–Sunter with operation-level comparison vectors |
| **What it addresses** | F1 and F2 |
| **Evidence** | The catalogue is small and consistent across countries. Neighbours use "add descriptor + change number" |
| **What could go wrong** | Hand-built script extraction is fiddly. Gains over rapidfuzz features plus trees may be small because trees already approximate it. Operations we fail to enumerate become noise |
| **Complexity** | Linear in pairs |
| **Difficulty** | Medium–high |
| **Robustness** | High *if* the operations really are country-agnostic |

### D. Metric-aware constrained decoding

| | |
|---|---|
| **Idea** | Replace "global t" with: calibrate probabilities per country and source; estimate each S1's set-size distribution; choose sets by the marginal rule of section 5.3 under record exclusivity (Lagrangian or greedy) |
| **What it addresses** | F9 and F10 |
| **Evidence** | [MATH]. The t = 0.02 optimum shows the metric rewards recall-first decisions per S1 |
| **What could go wrong** | Already failed twice when done naively (section 5.4). Needs isotonic calibration grouped by S1 and a test-shifted prior on n |
| **Complexity** | Trivial compute |
| **Difficulty** | Low–medium |
| **Robustness** | Medium: sensitive to calibration drift on France |

**Cheapest direction to try,** because it only re-decodes existing scores.

### E. France-specific unsupervised adaptation

| | |
|---|---|
| **Idea** | A French lexicon, plus EM or self-training of operation weights and thresholds on France test pairs, plus France-only leaderboard probes to choose between variants |
| **What it addresses** | F2 |
| **Evidence** | The biggest measured gap. France is 15% of test |
| **What could go wrong** | Pseudo-label confirmation bias. Probing overfits the public subset. The grey area on pseudo-labels (disclose them) |
| **Complexity** | Small |
| **Difficulty** | Low–medium |
| **Robustness** | Medium |

### F. Two-speed pipeline: deterministic easy lane + learned hard lane

| | |
|---|---|
| **Idea** | Accept, with rules, pairs whose normalized name and address match exactly, with an equal number and no rival. That covers about 20% of pairs, and the rule is likely more than 99.9% precise [HYP]. Spend all model and compute budget on the rest |
| **What it addresses** | Compute on 8 GB RAM, and focus |
| **What could go wrong** | Rules silently fail on France. Validate the rule's precision per country, using leave-one-country-out |
| **Difficulty** | Low |

### G. Uncertainty-gated second stage (cross-encoder or small transformer)

| | |
|---|---|
| **Idea** | Only pairs with p between 0.05 and 0.9 are re-scored by a fine-tuned MIT/Apache model (for example xlm-roberta-base or mdeberta-v3-base), fed the name, address and extracted numbers |
| **What it addresses** | Residual semantic cases: aka/dba, trade names, transliteration |
| **Evidence** | Weak. vishwajit argues remaining errors are information-limited |
| **What could go wrong** | GPU need; small gain; package complexity |
| **Difficulty** | High on your hardware |

---

## 13. Potential Competitive Advantages

The question: what could plausibly outperform thousands of AI-assisted teams? Ordered by (evidence × size ÷ cost):

1. **France done deliberately** (directions E and C).
   - The largest *measured* gap: 0.85–0.91 vs 0.97 on 15% of the test, worth up to about 0.01–0.015 of total leaderboard score.
   - Most teams will only add a lexicon.
   - Operation-level features, unsupervised adaptation and France-only probes are rarer.
2. **Entity-level evidence** (direction B).
   - Nobody uses it, and it targets the remaining precision band (F1) and the two largest recall holes (F3, F4).
   - The size is unknown until E1 is run. **This is the highest-variance, highest-ceiling idea.**
3. **Metric-correct decoding** (direction D).
   - Cheap. The theory is exact, and one team's empirical threshold behaviour supports it.
   - Most teams still sweep t over 0.3–0.9 and use train priors.
4. **Honest full-density validation,** reweighted to the *test* country mix.
   - Not novel, but a large share of repos get it wrong. It prevents wasting submissions.
5. **Compute efficiency as an enabler.**
   - On 8 GB you *cannot* copy the 160 GB cluster designs.
   - Blocking partitioned by region, with a fallback channel, gives about 15–40 min test runs [COMP:Abhigyanv23, SanthoshReddy]. That means more iteration cycles than cluster teams who get two runs.
   - In a 72-hour contest, **iterations are the currency.**

**Not an edge:**
- Bigger boosting ensembles. XGBoost, LightGBM and CatBoost land within noise [COMP:vishwajit].
- Dense-embedding blocking (measured recall 0.70).
- More training rows (the learning curve is flat).
- LLMs.

---

## 14. Risks & Unknowns

- **All dataset facts are second-hand** until E0 runs. They agree across four teams, but we have not verified them.
- **Composition of the public/private split is unknown.** Is it stratified by country? The France share of the public subset is unknown, and so is the size of France-only probe effects.
- **The test set-size prior** (about 4.1 true matches per S1) is a *model-based estimate* by one team.
- **Phantom clusters might not exist** (E1). Then direction B reduces to a recall-only lever.
- **Rules on pseudo-labels.** Fitting on unlabeled test inputs is not explicitly addressed. IDF over test is unavoidable and normal; EM or self-training on test is greyer. Disclose it, or ask via the organisers' Google Form.
- **"External data".** Hand-written lexicons (US states, Indian states, French departments, street abbreviations) are generic domain knowledge. anyascii is a rule-based library. libpostal is trained on OpenStreetMap, so it is risky; Epic021 avoids it too.
- **Compute:** 8 GB RAM and 17 GB free disk. The prep parquet alone is 2.5 GB, and candidate pairs for test run to tens of GB unless streamed. A cloud VM or Kaggle notebook is needed for full runs [INF]. Check the claimed $200 AWS credit on the portal.
- **Submission budget:** at most 15 in total, and today's may already be partly spent.
- **Leaderboard ties go to the earlier submission** [COMM]. That argues for an early baseline.
- **Reproducibility review of the top 100:** the pipeline must rerun from raw data. Fragile notebooks are a disqualification risk.

---

## 15. Experiments Worth Running

Ordered by cost-adjusted value. "GT" means train ground truth. All experiments are analyses, not solution code.

| ID | Question | Method | Cost | Decision it drives |
|---|---|---|---|---|
| **E0** | Re-verify the section 3 facts | Row counts; singleton rate; at-most-one-owner; country mix; per-source formatting; ID correlation between S1 and its matches | 20 min | Trust the landscape numbers |
| **E1** | **Do distractors form phantom clusters?** | For unmatched records, find other unmatched records within the same country with the same normalized core name, street and house number. Compare how often a positive record's deviation from S1 (number or added word) is shared by another record | 30–60 min | Go or no-go on the precision half of direction B |
| **E2** | Twin recall potential | Among blocking misses (non-Latin names, empty addresses), what share has a twin in S2 ∪ S3 that *is* retrieved correctly (same S1 in GT) and is linked to the miss by an exact address or name key? | 30 min | Go or no-go on the recall half of direction B |
| E3 | Cross-source agreement | For each S1, how often do copies from S2 and S3 agree with each other on fields that differ from S1? | 20 min | Cluster-feature design |
| E4 | Duplicates within S2 share fate? | Group exact and light-normalized duplicates; check GT owner consistency | 10 min | A free constraint and compute saving |
| E5 | Rule-lane precision (direction F) | Exact normalized name + address + number, unique best: precision per country | 20 min | Cheap easy lane |
| E6 | Partitioned-blocking ceiling on 8 GB | Recall and oracle F0.5 by (country, state) partition + fallback, **against full S1**, at k = 20/50 | 1–2 h | Feasibility of the whole pipeline locally |
| E7 | Metric-aware decoding | Group-calibrated p; marginal rule (section 5.3) vs global t; test-shifted prior on n | 30 min once any model exists | Direction D |
| E8 | Leave-one-country-out both ways, per feature group | Drop token-identity and density features; measure the gap | 1 h | France robustness |
| E9 | Transfer of operation indicators | Train operation-bucket features on the US only and score India, vs raw rapidfuzz features | 1–2 h | Direction C |
| E10 | France label-free diagnostics | Predicted singleton rate, matches per S1, share of records assigned, number-conflict share, by country | 10 min | Detect France breakage before spending a submission |
| E11 | France-only leaderboard probes | Two submissions that differ only in France rows | 2 submissions | Measure France directly |
| E12 | S2/S3 source interaction | Do per-source models or source-interacted features help? | 30 min | Cheap gain? |

**Discipline:**
- Change one component per version.
- Always report: blocking ceiling, full-density macro F0.5 reweighted to the test country mix, splits by country × source × singleton, and the leave-one-country-out gap.
- Log seeds and configs.

---

## 16. Questions We Still Need to Answer

1. Do phantom (multi-copy) distractors exist? (E1)
2. What share of blocking misses have a correctly retrievable twin? (E2)
3. Is the public leaderboard stratified by country? What is France's share of it?
4. What is the real test singleton rate and set-size distribution? Can it be estimated without labels (for example with EM on best-candidate score mixtures)?
5. Why do short or acronym names jump from 704 to 10k in test S2? Is it France-specific? How do they match?
6. Are there operations in France not seen in US/India? (Inspect 200 random French best-pairs by hand.)
7. How much does the Telangana↔AP swap (and similar) cost under state partitioning, and does a "neighbouring states" fallback fix it?
8. Does the claimed $200 AWS credit exist for us? Is Kaggle's free CPU (about 30 GB RAM) enough for full-test blocking when partitioned?
9. Are pseudo-labels on test inputs acceptable to the organisers? Ask via the Google Form now, since the answer takes time.
10. How many submissions have we already used today?

---

## 17. Recommended Research Priorities

This is time-boxed: about 51 hours remain. Nothing here is solution code yet; it is the order in which to *learn*.

1. **Now:** put the dataset in `student_resource/dataset/`. Run E0 and E4 (30 min). Decide on compute. With 8 GB locally, either (a) partition everything by (country, region) and stream candidates to disk per partition, or (b) move full runs to a Kaggle notebook or cloud VM. Decide before writing pipeline code.
2. **Next 2–3 h:** run E1, E2 and E3, the entity-structure questions. This is where the unconventional edge lives or dies, and it only needs ground truth plus simple keys, not a model.
3. **Then (day 1 night to day 2 morning):** build the standard foundation (direction A) with honest full-density validation and partitioned blocking. **Get a first leaderboard submission early:** it calibrates validation against the leaderboard, and ties go to earlier submissions.
4. **Day 2:**
   - Direction D (re-decoding only, cheap).
   - Direction B *if* E1 or E2 is positive.
   - France work (the E lexicon, E8, E10).
   - Use 2 submissions for France probes (E11).
5. **Day 3:** freeze features. Only recalibrate or re-decode. Regenerate from raw data on a clean environment. Build the package and write the documentation. Upload the final version at least 3 hours before the deadline.

**What to skip:**
- Embedding-based blocking.
- Large boosting ensembles.
- LLM matching.
- Hyperparameter searches.
- 5-fold CV at full scale.
- Any score measured on a sampled S1 index.

**A final reminder:** a validation score is a hypothesis about the private leaderboard, not proof. The private board is a different subset, and 15% of it is a country we have no labels for.

---

### Appendix: key sources

- Official: [linked repo](https://github.com/Sugandh-vI/Amazon-ML-challenge) (problem statement PDF, guidelines PDF, `student_resource/`)
- [vishwajitsarnobat/amazon-ml-challenge-2026](https://github.com/vishwajitsarnobat/amazon-ml-challenge-2026): `docs/pipeline_report.md` (full-density measurements)
- [Abhigyanv23/Amazon-ML-Challenge-2026](https://github.com/Abhigyanv23/Amazon-ML-Challenge-2026): `experiments/` (EDA, blocking misses, leaderboard 0.931)
- [SanthoshReddy352/Amazon-ML-Challenge-2026](https://github.com/SanthoshReddy352/Amazon-ML-Challenge-2026): `STATUS.md` (leaderboard history, France probes)
- [Epic021/amazon-ml-challenge-2026](https://github.com/Epic021/amazon-ml-challenge-2026): `STRATEGY.md` (neighbour negatives, France, test shift)
- [Siva402-ai/amazon-ml-challenge-2026](https://github.com/Siva402-ai/amazon-ml-challenge-2026), [vitthalg17/amazon-ml-challenge-2026-entity-resolution](https://github.com/vitthalg17/amazon-ml-challenge-2026-entity-resolution) (learned transliteration)
- Kaggle mirror of the data (not inspected): [satwiksps/amazon-ml-challenge-2026](https://www.kaggle.com/datasets/satwiksps/amazon-ml-challenge-2026)
