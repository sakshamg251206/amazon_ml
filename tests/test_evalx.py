import polars as pl
import pytest

from competition.evalx import candidate_metrics, macro_f05_pairs
from resolver.metrics import macro_f05

TRUTH = pl.DataFrame({"s1": ["a", "b", "b", "c"], "rec": [None, "x", "y", "z"]})
IDS = pl.Series(["a", "b", "c"])


def test_vectorised_f05_matches_reference_scorer():
    pred = pl.DataFrame({"s1": ["a", "b", "b", "c"], "rec": ["q", "x", "w", "z"]})
    ref = macro_f05({"a": {"q"}, "b": {"x", "w"}, "c": {"z"}},
                    {"a": frozenset(), "b": frozenset({"x", "y"}), "c": frozenset({"z"})})
    m = macro_f05_pairs(pred, TRUTH, IDS)
    assert m["f05"] == pytest.approx(ref)
    assert m["precision"] == pytest.approx(2 / 4)
    assert m["recall"] == pytest.approx(2 / 3)


def test_restricting_to_subset_ignores_other_s1():
    pred = pl.DataFrame({"s1": ["b", "zzz"], "rec": ["x", "x"]})
    m = macro_f05_pairs(pred, TRUTH, pl.Series(["b"]))
    assert m["precision"] == 1.0 and m["n_s1"] == 1


def test_candidate_recall_and_oracle():
    cand = pl.DataFrame({"s1": ["b", "b", "a"], "rec": ["x", "w", "q"]})
    m = candidate_metrics(cand, TRUTH, IDS)
    assert m["cand_recall"] == pytest.approx(1 / 3)
    # oracle keeps only true candidates: a -> {} (1.0), b -> {x} (1.25/(1+0.5)), c -> {} (0.0)
    assert m["oracle_f05"] == pytest.approx((1.0 + 1.25 / 1.5 + 0.0) / 3)
