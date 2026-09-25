# Entity Resolution Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the business entity-resolution pipeline described in the PRD in six phases. This document has two kinds of phase:
- **Phases 0–1** are fully specified here. They end with a valid leaderboard submission before 25 Sep, 23:59 IST.
- **Phases 2–6** are a roadmap. Each one has deliverables and gates, and gets its own detailed plan once the measurements it depends on exist.

**Architecture:** Resumable stages (`prep → block → features → train → decide`) under a single `src/run.py` command-line tool.
- Phase 1 ships an **exact-key rule baseline**. The key is (country, sorted core-name tokens, house number), and a key must be unique in S1.
- The baseline has two jobs: calibrate our scorer against the leaderboard, and produce Submission 1.

**Tech Stack:**
- Python 3.12 (to match Colab), with `uv` for the environment.
- polars for data frames.
- anyascii for transliteration.
- pytest for tests.
- Added later: scikit-learn, sparse_dot_topn, rapidfuzz, lightgbm.

**Spec:** `docs/superpowers/specs/2026-09-25-entity-resolution-prd.md`. Background: `AMAZON_ML_DEEP_RESEARCH.md`.

## Global Constraints

- Read every TSV with a tab separator, **no quoting**, all columns as strings, and empty fields as `""` (never null or NaN).
- Every test S1 gets exactly one output row, France included. Rows are written by iterating over the `test_source1` IDs.
- No duplicate IDs inside a list, and only `S2-`/`S3-` IDs in lists.
- `candidate_pairs.tsv` ⊇ `matching_results.tsv` for every S1.
- `country` is never a model feature. It is used only for partitioning and keys.
- No external data. The model is MIT or Apache-2.0 licensed with ≤8B parameters.
- `requirements.txt` pins exact versions, produced with `uv pip freeze`.
- The seed is 42 everywhere, and it is logged.
- Large artifacts go under `work/` and `output/`. Both are gitignored, as is `student_resource/dataset/`.

## Review Focus

These are the five failure modes most likely to bite, with the test that pins each one:

1. **Empty address or empty name field.** It must not crash, and it must not produce a key that matches everything. Pinned by `test_add_keys_drops_empty_keys` in Task 3.
2. **Chain businesses:** two S1 rows with the same name and house number. Neither may be matched by the rule, since that would be ambiguous. Pinned by `test_rule_skips_ambiguous_s1_keys` in Task 3.
3. **A France S1 that has no match** must still get a row. Pinned by `test_writer_emits_every_s1_once` in Task 4, which includes an S1 with no matches.
4. **Non-ASCII input** (accents, Devanagari). anyascii handles it without raising. Pinned by `test_clean_transliterates` in Task 2.
5. **Leading zeros and letter suffixes on house numbers** (`02A/3`, `3906a`). They must normalize to the digits only. Pinned by `test_house_no_variants` in Task 2.

---

## Phase map

| Phase | Delivers | Gate or exit criterion | Target time (IST) |
|---|---|---|---|
| **0 Setup** | Environment, config, data loader | `pytest` runs | 25 Sep, 21:30 |
| **1 Scorer + rule baseline** | `metrics.py`, `normalize.py` v1, `baseline_rule.py`, `decide.write_id_lists`, `run.py --stage rule` | Official validator PASS → **Submission 1** | 25 Sep, 23:30 |
| 2 Data experiments | `experiments/e0_verify.py`, `e1_phantoms.py`, `e2_twins.py`, `e4_dups.py` + results in `experiments/log.md` | Each gate decided (PRD §5) | 26 Sep, 01:00 |
| 3 Blocking | `normalize.py` v2 (lexicons, `state_canon`, `partition_key`), `block.py`, Colab notebook, holdout split | **G-block: holdout recall ≥ 0.97 against all train S1** | 26 Sep, 09:00 |
| 4 Features + model | `features.py`, `train.py`, `decide.py` threshold sweep, `experiments/log.csv` | Holdout test-mix F0.5 ≥ 0.97 → **Submission 2** | 26 Sep, 13:00 |
| 5 Gated edges | E7 decoding; France lexicon + E8 + E11 probes; twin/phantom if E1/E2 passed | Each edge: +0.002 test-mix, or a positive probe | 27 Sep, 18:00 |
| 6 Freeze + package | Regenerate from raw on a fresh Colab, validator, documentation, zip | Byte-identical output; final upload by 21:00 | 27 Sep, 21:00 |

---

## File structure (Phases 0–1)

| File | Responsibility |
|---|---|
| `requirements.txt` | Pinned dependencies (generated) |
| `src/__init__.py` | Marks `src` as a package (empty) |
| `src/config.py` | Paths (overridable by environment variable) and seed |
| `src/data.py` | Readers for the source TSVs and the ground truth |
| `src/metrics.py` | Exact F0.5 for one entity and macro F0.5 |
| `src/normalize.py` | `clean`, `name_key`, `house_no` |
| `src/baseline_rule.py` | `add_keys`, `rule_matches` |
| `src/decide.py` | `write_id_lists` (TSV writer, shared by all later phases) |
| `src/run.py` | Command-line tool (`--stage rule --split train\|test`) and result logging |
| `tests/test_metrics.py`, `tests/test_normalize.py`, `tests/test_rule.py`, `tests/test_decide.py` | Unit tests |

---

## Phase 0: Setup

### Task 0: Environment, config, data loader

**Files:**
- Create: `requirements.txt`, `src/__init__.py`, `src/config.py`, `src/data.py`, `tests/test_data.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces:
  - `config.DATA_DIR: Path`, `config.WORK_DIR: Path`, `config.OUTPUT_DIR: Path`, `config.SEED: int`
  - `data.read_source(split: str, source: int) -> pl.DataFrame`, with columns `entity_id, business_name, business_address, country`, all strings and no nulls
  - `data.read_ground_truth() -> dict[str, frozenset[str]]`

- [ ] **Step 1: Create the environment**

```bash
cd /Users/sakshamgarg/Desktop/projects/Amazon_ML
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install polars anyascii pytest
uv pip freeze > requirements.txt
printf 'work/\noutput/\n.venv/\n__pycache__/\n.pytest_cache/\n' >> .gitignore
mkdir -p src tests && touch src/__init__.py
```

- [ ] **Step 2: Write the failing test** in `tests/test_data.py`

```python
import pytest
from src import config
from src.data import read_source, read_ground_truth

needs_data = pytest.mark.skipif(
    not (config.DATA_DIR / "train" / "train_source1.tsv").exists(), reason="dataset not present"
)

@needs_data
def test_read_source_all_strings_no_nulls():
    df = read_source("test", 1)
    assert df.columns == ["entity_id", "business_name", "business_address", "country"]
    assert df.height == 1_732_544
    assert df.null_count().sum_horizontal().item() == 0

@needs_data
def test_read_ground_truth_singletons_are_empty_sets():
    gt = read_ground_truth()
    assert len(gt) == 2_206_821
    assert sum(1 for v in gt.values() if not v) == 123_247
```

- [ ] **Step 3: Run it and check that it fails**

Run: `python -m pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 4: Implement `src/config.py` and `src/data.py`**

```python
# src/config.py
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("ER_DATA_DIR", ROOT / "student_resource" / "dataset"))
WORK_DIR = Path(os.environ.get("ER_WORK_DIR", ROOT / "work"))
OUTPUT_DIR = Path(os.environ.get("ER_OUTPUT_DIR", ROOT / "output"))
SEED = 42
```

```python
# src/data.py
"""Readers for the challenge TSVs. Every column is a string; empty fields are ""."""
import polars as pl

from src.config import DATA_DIR


def _read_tsv(path) -> pl.DataFrame:
    # quote_char=None: addresses contain quotes/commas; the official format has no quoting.
    return pl.read_csv(
        path, separator="\t", quote_char=None, infer_schema=False,
        missing_utf8_is_empty_string=True,
    )


def read_source(split: str, source: int) -> pl.DataFrame:
    return _read_tsv(DATA_DIR / split / f"{split}_source{source}.tsv")


def read_ground_truth() -> dict[str, frozenset[str]]:
    df = _read_tsv(DATA_DIR / "train" / "train_ground_truth.tsv")
    return {s1: frozenset(m.split(",")) if m else frozenset() for s1, m in df.iter_rows()}
```

- [ ] **Step 5: Run it and check that it passes**

Run: `python -m pytest tests/test_data.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add .gitignore requirements.txt src/__init__.py src/config.py src/data.py tests/test_data.py
git commit -m "feat: environment, config and TSV loaders"
```

---

## Phase 1: Scorer + rule baseline → Submission 1

### Task 1: Exact macro F0.5 scorer

**Files:**
- Create: `src/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `entity_f05(pred: AbstractSet[str], true: AbstractSet[str]) -> float`
  - `macro_f05(pred: Mapping[str, AbstractSet[str]], truth: Mapping[str, AbstractSet[str]]) -> float`. It averages over the **truth** keys; a missing prediction counts as empty.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py
import pytest
from src import config
from src.metrics import entity_f05, macro_f05


def test_pdf_worked_example():
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    true = {"S2-00047", "S3-00812"}
    assert entity_f05(pred, true) == pytest.approx(0.7142857, abs=1e-6)


def test_singleton_rules():
    assert entity_f05(set(), set()) == 1.0
    assert entity_f05({"S2-1"}, set()) == 0.0


def test_non_singleton_empty_prediction_scores_zero():
    assert entity_f05(set(), {"S2-1"}) == 0.0


def test_macro_averages_over_truth_and_treats_missing_as_empty():
    truth = {"a": frozenset(), "b": frozenset({"S2-1"})}
    assert macro_f05({}, truth) == 0.5          # a: 1.0 (empty ok), b: 0.0
    assert macro_f05({"b": {"S2-1"}}, truth) == 1.0


@pytest.mark.skipif(not (config.DATA_DIR / "train").exists(), reason="dataset not present")
def test_all_empty_on_train_equals_singleton_rate():
    from src.data import read_ground_truth
    assert macro_f05({}, read_ground_truth()) == pytest.approx(0.0558, abs=5e-4)
```

- [ ] **Step 2: Run them and check that they fail**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.metrics'`

- [ ] **Step 3: Implement**

```python
# src/metrics.py
"""Competition metric: F0.5 per Source-1 entity, macro-averaged.

With P = TP/|pred| and R = TP/|true|, F0.5 = 1.25*P*R / (0.25*P + R), which simplifies to
1.25*TP / (|pred| + 0.25*|true|). A singleton (no true matches) scores 1 only for an empty
prediction.
"""
from collections.abc import Mapping, Set as AbstractSet

_EMPTY: frozenset[str] = frozenset()


def entity_f05(pred: AbstractSet[str], true: AbstractSet[str]) -> float:
    if not true:
        return 0.0 if pred else 1.0
    tp = len(pred & true)
    return 1.25 * tp / (len(pred) + 0.25 * len(true)) if tp else 0.0


def macro_f05(pred: Mapping[str, AbstractSet[str]], truth: Mapping[str, AbstractSet[str]]) -> float:
    return sum(entity_f05(pred.get(s1, _EMPTY), t) for s1, t in truth.items()) / len(truth)
```

- [ ] **Step 4: Run them and check that they pass**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: exact macro F0.5 scorer"
```

### Task 2: Normalizer v1 (`clean`, `name_key`, `house_no`)

**Files:**
- Create: `src/normalize.py`, `tests/test_normalize.py`

**Interfaces:**
- Produces:
  - `clean(text: str) -> str`: ASCII, lowercase, dots removed, other non-alphanumerics become spaces.
  - `name_key(name: str) -> str`: the sorted, de-duplicated core tokens (legal forms and stopwords removed), joined by one space. Returns `""` if nothing is left.
  - `house_no(address: str) -> str`: the first run of digits with leading zeros stripped, or `""` if there are none.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_normalize.py
from src.normalize import clean, name_key, house_no


def test_clean_transliterates():
    assert clean("Silver Fóundation Pvt. Ltd.") == "silver foundation pvt ltd"
    assert clean("Tucker [Metro-Transalta]") == "tucker metro transalta"
    assert clean("सिल्वर").isascii()


def test_name_key_ignores_order_case_legal_forms_and_stopwords():
    assert name_key("Yazzie-Empire Enterprises LLC") == name_key("ENTERPRISES  YAZZIE EMPIRE")
    assert name_key("Surgical Care Associates of Topeka Inc") == name_key(
        "surgical care associates of topeka (Inc)")
    assert name_key("Dukes Chadwick Có") == name_key("Dukes Chadwick Co") == "chadwick dukes"
    assert name_key("Meerut Jewellery Private Limited") == name_key("MEERUT JEWELLERY")


def test_name_key_empty_when_only_legal_form():
    assert name_key("LLC") == ""
    assert name_key("") == ""


def test_house_no_variants():
    assert house_no("24800 EUCLID AVE, EUCLID, OH") == "24800"
    assert house_no("02A/3, DELHI, MURADNAGAR") == "2"
    assert house_no("3906a 33rd Ter, Topeka, Kansas") == "3906"
    assert house_no("OH, EUCLID AVE, EUCLID") == ""
    assert house_no("") == ""
```

- [ ] **Step 2: Run them and check that they fail**

Run: `python -m pytest tests/test_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.normalize'`

- [ ] **Step 3: Implement**

```python
# src/normalize.py
"""Text normalisation shared by every stage. v1: just enough for the exact-key rule baseline."""
import re

from anyascii import anyascii

# Legal forms across US / India / France (hand-written domain knowledge, no external data).
LEGAL_FORMS = frozenset({
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
    "pvt", "private", "llp", "lp", "pc", "plc", "pllc", "sarl", "sas", "sasu", "eurl", "sci", "sa",
})
STOPWORDS = frozenset({"and", "the", "of"})

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DIGITS = re.compile(r"\d+")


def clean(text: str) -> str:
    # Dots are removed (not spaced) so "L.L.C." -> "llc" and "Pvt." -> "pvt".
    return _NON_ALNUM.sub(" ", anyascii(text).lower().replace(".", "")).strip()


def name_key(name: str) -> str:
    tokens = {t for t in clean(name).split() if t not in LEGAL_FORMS and t not in STOPWORDS}
    return " ".join(sorted(tokens))


def house_no(address: str) -> str:
    m = _DIGITS.search(anyascii(address))
    return (m.group().lstrip("0") or "0") if m else ""
```

- [ ] **Step 4: Run them and check that they pass**

Run: `python -m pytest tests/test_normalize.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/normalize.py tests/test_normalize.py
git commit -m "feat: normaliser v1 (clean, name_key, house_no)"
```

### Task 3: Exact-key rule matcher

**Files:**
- Create: `src/baseline_rule.py`, `tests/test_rule.py`

**Interfaces:**
- Consumes: `normalize.name_key`, `normalize.house_no`
- Produces:
  - `add_keys(df: pl.DataFrame) -> pl.DataFrame`. It keeps `entity_id` and adds `key` = `country|name_key|house_no`. It drops any row whose name_key or house_no is empty.
  - `rule_matches(s1: pl.DataFrame, recs: pl.DataFrame) -> dict[str, list[str]]`, mapping S1 ID → matched record IDs. Only keys that are **unique in S1** are used.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rule.py
import polars as pl
from src.baseline_rule import add_keys, rule_matches

COLS = ["entity_id", "business_name", "business_address", "country"]
S1 = pl.DataFrame([
    ["S1-1", "Yazzie Empire Enterprises LLC", "24800 Euclid Avenue, Euclid, OH", "US"],
    ["S1-2", "Primary Care Group", "10 Main St, Austin, TX", "US"],     # chain: same key as S1-3
    ["S1-3", "Primary Care Group", "10 Main Street, Austin, Texas", "US"],
    ["S1-4", "Lonely Shop", "5 Rue de Valmy, Lille", "France"],         # no records
], schema=COLS, orient="row")


def test_add_keys_drops_empty_keys():
    df = pl.DataFrame([["S2-9", "LLC", "1 X St", "US"], ["S2-8", "Acme", "", "US"]],
                      schema=COLS, orient="row")
    assert add_keys(df).height == 0


def test_rule_matches_exact_key_and_rejects_number_change():
    recs = pl.DataFrame([
        ["S2-1", "YAZZIE EMPIRE  ENTERPRISES", "24800 EUCLID AVE, EUCLID, OH", "US"],
        ["S3-1", "Yazzie Empire Enterprises", "24801 Euclid Ave, Euclid, Ohio", "US"],  # neighbour
    ], schema=COLS, orient="row")
    assert rule_matches(S1, recs) == {"S1-1": ["S2-1"]}


def test_rule_skips_ambiguous_s1_keys():
    recs = pl.DataFrame([["S2-5", "PRIMARY CARE GROUP", "10 MAIN ST, AUSTIN, TX", "US"]],
                        schema=COLS, orient="row")
    assert rule_matches(S1, recs) == {}


def test_rule_never_crosses_country():
    recs = pl.DataFrame([["S2-7", "Lonely Shop", "5 Main St", "US"]], schema=COLS, orient="row")
    assert rule_matches(S1, recs) == {}
```

- [ ] **Step 2: Run them and check that they fail**

Run: `python -m pytest tests/test_rule.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.baseline_rule'`

- [ ] **Step 3: Implement**

```python
# src/baseline_rule.py
"""Submission-1 baseline: a record matches an S1 iff they share (country, name_key, house_no)
and that key is unique in S1. High precision, modest recall; no learning."""
import polars as pl

from src.normalize import house_no, name_key


def add_keys(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col("business_name").map_elements(name_key, return_dtype=pl.String).alias("nk"),
            pl.col("business_address").map_elements(house_no, return_dtype=pl.String).alias("hn"),
        )
        .filter((pl.col("nk") != "") & (pl.col("hn") != ""))
        .select("entity_id", pl.concat_str(["country", "nk", "hn"], separator="|").alias("key"))
    )


def rule_matches(s1: pl.DataFrame, recs: pl.DataFrame) -> dict[str, list[str]]:
    s1k = add_keys(s1).filter(pl.col("key").is_unique())
    rk = add_keys(recs).rename({"entity_id": "rec_id"})
    pairs = rk.join(s1k, on="key", how="inner").select("entity_id", "rec_id")
    out: dict[str, list[str]] = {}
    for s1_id, rec_id in pairs.iter_rows():
        out.setdefault(s1_id, []).append(rec_id)
    return out
```

- [ ] **Step 4: Run them and check that they pass**

Run: `python -m pytest tests/test_rule.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/baseline_rule.py tests/test_rule.py
git commit -m "feat: exact-key rule baseline matcher"
```

### Task 4: TSV writer

**Files:**
- Create: `src/decide.py`, `tests/test_decide.py`

**Interfaces:**
- Produces: `write_id_lists(path: Path, list_col: str, s1_ids: Sequence[str], mapping: Mapping[str, Iterable[str]]) -> None`. It writes a header of `source1_entity_id\t{list_col}` and then **one row per S1 in `s1_ids`, in order**, with each list de-duplicated and sorted. Later phases reuse it for both output files.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_decide.py
from src.decide import write_id_lists


def test_writer_emits_every_s1_once(tmp_path):
    p = tmp_path / "m.tsv"
    write_id_lists(p, "matched_entity_ids", ["S1-1", "S1-2", "S1-3"],
                   {"S1-1": ["S3-9", "S2-4", "S2-4"], "S1-9": ["S2-1"]})
    assert p.read_text(encoding="utf-8").splitlines() == [
        "source1_entity_id\tmatched_entity_ids",
        "S1-1\tS2-4,S3-9",
        "S1-2\t",
        "S1-3\t",
    ]
```

- [ ] **Step 2: Run it and check that it fails**

Run: `python -m pytest tests/test_decide.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.decide'`

- [ ] **Step 3: Implement**

```python
# src/decide.py
"""Final decisions -> submission files."""
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


def write_id_lists(path: Path, list_col: str, s1_ids: Sequence[str],
                   mapping: Mapping[str, Iterable[str]]) -> None:
    """One row per S1 in s1_ids (so no entity is ever missing), deduped + sorted ID lists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{list_col}\n")
        for s1 in s1_ids:
            f.write(f"{s1}\t{','.join(sorted(set(mapping.get(s1, ()))))}\n")
```

- [ ] **Step 4: Run it and check that it passes**

Run: `python -m pytest tests/test_decide.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/decide.py tests/test_decide.py
git commit -m "feat: submission TSV writer"
```

### Task 5: Command-line tool, train evaluation, test run, validator → Submission 1

**Files:**
- Create: `src/run.py`, `experiments/log.csv` (created by the first run)

**Interfaces:**
- Consumes: `data.read_source`, `data.read_ground_truth`, `baseline_rule.rule_matches`, `metrics.macro_f05`, `decide.write_id_lists`, `config.OUTPUT_DIR`
- Produces:
  - `python -m src.run --stage rule --split train` → prints and logs overall and per-country macro F0.5.
  - `python -m src.run --stage rule --split test` → writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

- [ ] **Step 1: Implement `src/run.py`**

The pieces are unit-tested in Tasks 1–4. This step is glue code, verified end to end on real data in Steps 2–4.

```python
# src/run.py
"""Pipeline entry point. Phase 1: --stage rule."""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

from src import config
from src.baseline_rule import rule_matches
from src.data import read_ground_truth, read_source
from src.decide import write_id_lists
from src.metrics import macro_f05

LOG = config.ROOT / "experiments" / "log.csv"


def log_result(version: str, metrics: dict[str, float]) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "version", "seed", "metric", "value"])
        for k, v in metrics.items():
            w.writerow([datetime.now().isoformat(timespec="seconds"), version, config.SEED, k, f"{v:.5f}"])


def run_rule(split: str) -> None:
    t0 = time.time()
    s1 = read_source(split, 1)
    matches: dict[str, list[str]] = {}
    for src in (2, 3):  # one source at a time keeps peak memory low on an 8 GB machine
        for k, v in rule_matches(s1, read_source(split, src)).items():
            matches.setdefault(k, []).extend(v)
    print(f"rule matched {len(matches):,} S1 in {time.time() - t0:.0f}s")

    if split == "train":
        truth = read_ground_truth()
        pred = {k: set(v) for k, v in matches.items()}
        metrics = {"f05_all": macro_f05(pred, truth)}
        for country, ids in s1.group_by("country").agg("entity_id").iter_rows():
            metrics[f"f05_{country}"] = macro_f05(pred, {i: truth[i] for i in ids})
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        log_result("v0_rule", metrics)
    else:
        ids = s1["entity_id"].to_list()
        write_id_lists(config.OUTPUT_DIR / "matching_results.tsv", "matched_entity_ids", ids, matches)
        # For the rule baseline the "model" is the rule, so candidates == matches.
        write_id_lists(config.OUTPUT_DIR / "candidate_pairs.tsv", "candidate_entity_ids", ids, matches)
        print(f"wrote {config.OUTPUT_DIR}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["rule"])
    p.add_argument("--split", required=True, choices=["train", "test"])
    a = p.parse_args(argv)
    if a.stage == "rule":
        run_rule(a.split)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Score the rule on all of train** (the rule has no fitted parameters, so all of train is a fair evaluation)

Run: `python -m src.run --stage rule --split train`

Expected:
- It prints `f05_all`, `f05_US` and `f05_India`.
- A score in the **0.55–0.75** range is expected: SanthoshReddy's similar rule scored 0.680 on validation and 0.680 on the public leaderboard.
- A score below 0.3 means there is a bug. Debug with `superpowers:systematic-debugging` before continuing.

- [ ] **Step 3: Produce the test outputs**

Run: `python -m src.run --stage rule --split test`
Expected: `output/matching_results.tsv` and `output/candidate_pairs.tsv`, each with 1,732,545 lines including the header.

- [ ] **Step 4: Run the official validator**

Run:
```bash
python student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
```
Expected: `PASS`. Then re-run with `--check-ids` added. If it runs out of memory on the 8 GB machine, drop `--candidate`: the two files are identical here.

- [ ] **Step 5: Commit, and upload Submission 1**

```bash
git add src/run.py experiments/log.csv
git commit -m "feat: rule-baseline CLI; train score logged; test outputs validated"
```

Upload `output/matching_results.tsv` in the Unstop portal before **23:59 IST**. Record the public score in `experiments/log.csv` as the metric `public_lb` for version `v0_rule`, then commit.

**Why this is still useful:** the gap between the public score and the train score is our first check that the scorer and the leaderboard agree. The France share of that gap is visible because France has no training counterpart.

---

## Phases 2–6: Roadmap

Each phase gets its own detailed plan when it starts. Their designs depend on measurements that only exist once Phase 1 has shipped.

### Phase 2: Data experiments (target 26 Sep, 01:00; runs while Colab is set up)

| Script | Question | Gate (from PRD §5) |
|---|---|---|
| `experiments/e0_verify.py` | Re-check research-report §3: at most one owner per record, match counts, per-source formatting, ID correlation | Informational |
| `e1_phantoms.py` | Unmatched records sharing (name_key, street, house_no) with other records, vs how often a true copy's deviation from S1 is shared | ≥3× gap → build edge 3a |
| `e2_twins.py` | Share of rule-missed or non-Latin / empty-address positives that have an exact-key twin which is matched | ≥20% → build edge 3b |
| `e4_dups.py` | Do exact duplicates within a source always share an owner? | ≥99.9% → deduplicate before scoring |

Results are written to `experiments/log.md`.

### Phase 3: Blocking (target 26 Sep, 09:00)

- `normalize.py` v2:
  - lexicons for US / India / France (street abbreviations, R/AV/BD/IMP/ALL/RTE/PL/QUAI/FBG, BIS/TER);
  - `name_core`, `name_nospace`, `nums`, `street_tokens`;
  - `state_canon` with equivalence groups (TG≈AP, department → region, hdf/naq/pdl, French city → region);
  - `partition_key`.
- A `prep` stage that writes Parquet to `WORK_DIR`.
- `block.py`:
  - per (country, partition): char 3–4-gram TF-IDF on the name + word TF-IDF on the address + a combined view;
  - top-k search with `sparse_dot_topn` (k_name = 20, k_comb = 50);
  - ∪ exact keys (`name_nospace`, house_no + first street token, name_core + locality);
  - ∪ a country-level fallback (k = 20) for records with no region.
- A holdout split: 10% of train S1, hashed and stratified by country × match count.
- `metrics.blocking_recall` / `oracle_f05`.
- A Colab notebook `notebooks/colab_run.ipynb` that mounts Drive, installs pinned requirements and calls `run.py` stages.
- **Gate G-block:** holdout recall ≥ 0.97 against all 2.2M train S1.

### Phase 4: Features + model (target 26 Sep, 13:00)

- `features.py`: the ~40 country-agnostic features in PRD §3.4.
- `train.py`: LightGBM on about 250k non-holdout train S1, blocked against all train S1.
- `decide.py`: argmax per record + threshold sweep over [0.01, 0.95]; candidate file = all scored pairs.
- Run log with the full PRD §4.2 metrics and the label-free test checks.
- **Exit:** test-mix F0.5 ≥ 0.97 → Submission 2.

### Phase 5: Gated edges (26 Sep 13:00 → 27 Sep 18:00)

| Edge | Contents | Gate |
|---|---|---|
| E7 | Isotonic calibration + rank-dependent marginal rule | +0.002 test-mix |
| France | Lexicon (always on); E8 leave-one-country-out per feature group; E11 probes (≤2 submissions) | Probe raises the public score |
| 3a / 3b | Twin and phantom features or pass, only if E1/E2 passed | +0.002 test-mix |
| Extras | E4 deduplication, E5 rule lane | +0.002 test-mix |

Each edge is its own version and its own submission.

### Phase 6: Freeze + package (27 Sep, 18:00–21:00)

- Fresh Colab → `run.py` all stages from the raw TSVs → `diff` against the submitted file (must be identical).
- Validator with `--check-ids`.
- Assemble `code/business_entity_resolution/{src,README.md,requirements.txt}`, `output/`, and the filled-in `Documentation_template.md` into `<team>_submission.zip`.
- Final upload by 21:00.
