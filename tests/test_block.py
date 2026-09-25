import polars as pl
import pytest

from src.block import tfidf_candidates

S1 = pl.DataFrame({"name_tok": ["yazzie empire enterprises", "surgical care associates topeka", "blue program"],
                   "addr": ["24800 euclid avenue euclid oh", "3906 33rd terrace topeka ks", "261 jo mar road ardmore al"]})


def test_typo_and_reordered_records_retrieve_their_s1_first():
    recs = pl.DataFrame({"name_tok": ["yazzie empier enterprises", "surgical caare associates topeka"],
                         "addr": ["oh euclid ave euclid", "3906a 33rd ter topeka kansas"]})
    c = tfidf_candidates(S1, recs, k_name=1, k_comb=1)
    best = c.sort("cos_comb", descending=True).group_by("r").first().sort("r")
    assert best["s"].to_list() == [0, 1]


def test_comb_cosine_is_weighted_sum():
    c = tfidf_candidates(S1, S1, k_name=3, k_comb=3)
    row = c.filter((pl.col("r") == 0) & (pl.col("s") == 0)).row(0, named=True)
    assert row["cos_name"] == pytest.approx(1.0, abs=1e-5)
    assert row["cos_comb"] == pytest.approx(0.6 * row["cos_name"] + 0.4 * row["cos_addr"], abs=1e-5)


def test_empty_addresses_do_not_crash():
    recs = pl.DataFrame({"name_tok": ["blue program"], "addr": [""]})
    c = tfidf_candidates(S1, recs, k_name=2, k_comb=2)
    assert 2 in c["s"].to_list()


def test_no_records_returns_empty_frame():
    recs = pl.DataFrame({"name_tok": [], "addr": []}, schema={"name_tok": pl.String, "addr": pl.String})
    c = tfidf_candidates(S1, recs, k_name=2, k_comb=2)
    assert c.height == 0 and "cos_comb" in c.columns
