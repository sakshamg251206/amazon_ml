import pytest
from competition import config
from resolver.metrics import entity_f05, macro_f05


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
    from competition.data import read_ground_truth
    assert macro_f05({}, read_ground_truth()) == pytest.approx(0.0558, abs=5e-4)
