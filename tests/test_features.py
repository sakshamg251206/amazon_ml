import polars as pl
import pytest

from src.features import FEATURES, pair_features


def _recs(rows):
    return pl.DataFrame(rows, schema=["entity_id", "name_tok", "addr", "nums", "hn", "name_ns", "name_key",
                                      "nonlatin", "empty_addr"], orient="row")


S1 = _recs([["S1-1", "yazzie empire", "24800 euclid ave", ["24800"], "24800", "yazzieempire", "empire yazzie", False, False],
            ["S1-2", "yazzie empire", "24801 euclid ave", ["24801"], "24801", "yazzieempire", "empire yazzie", False, False]])
RECS = _recs([["S2-1", "yazzie empire", "24800 euclid ave", ["24800"], "24800", "yazzieempire", "empire yazzie", False, False]])
PAIRS = pl.DataFrame({"s1": ["S1-1", "S1-2"], "rec": ["S2-1", "S2-1"], "cos_name": [1.0, 1.0],
                      "cos_addr": [1.0, 0.8], "cos_comb": [1.0, 0.92], "in_keys": [True, False]})


def test_number_relation_separates_neighbour():
    f = pair_features(PAIRS, S1, RECS).sort("s1")
    assert f["hn_eq"].to_list() == [True, False]
    assert f["hn_lev"].to_list() == [0.0, 1.0]
    assert f["nums_jac"].to_list() == [1.0, 0.0]


def test_rank_and_margin_context():
    f = pair_features(PAIRS, S1, RECS).sort("s1")
    assert f["rank_fz"].to_list() == [1, 2]
    assert f["n_cand_rec"].to_list() == [2, 2]
    assert f["margin_fz"][0] == pytest.approx(f["fz"][0] - f["fz"][1])


def test_all_declared_features_exist():
    f = pair_features(PAIRS, S1, RECS)
    assert set(FEATURES) <= set(f.columns)
