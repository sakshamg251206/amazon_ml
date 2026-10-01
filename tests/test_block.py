import polars as pl
import pytest

from resolver.block import tfidf_candidates

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


def test_address_view_retrieves_renamed_business():
    recs = pl.DataFrame({"name_tok": ["zzzz qqqq"], "addr": ["3906 33rd terrace topeka ks"]})
    without = tfidf_candidates(S1, recs, k_name=1, k_comb=1)
    with_addr = tfidf_candidates(S1, recs, k_name=1, k_comb=1, k_addr=1)
    assert 1 in with_addr["s"].to_list() and with_addr.height >= without.height


def test_chunked_topk_equals_unchunked(monkeypatch):
    import resolver.block as B
    recs = pl.DataFrame({"name_tok": ["yazzie empier", "surgical caare topeka", "blue progrm", "blue"],
                         "addr": ["euclid oh", "topeka ks", "ardmore al", ""]})
    full = tfidf_candidates(S1, recs, k_name=2, k_comb=2).sort("r", "s")
    monkeypatch.setattr(B, "MAX_CHUNK_CELLS", 1)
    chunked = tfidf_candidates(S1, recs, k_name=2, k_comb=2).sort("r", "s")
    assert full.select("r", "s").equals(chunked.select("r", "s"))


def test_topk_matches_bruteforce_dense_and_skips_zero_scores(monkeypatch):
    import numpy as np
    import scipy.sparse as sp
    import resolver.block as B
    q = sp.random(37, 50, density=0.1, format="csr", random_state=1, dtype=np.float32)
    idx = sp.random(23, 50, density=0.1, format="csr", random_state=2, dtype=np.float32)
    dense = (q @ idx.T).toarray()
    for chunk_cells in (10**9, 50):  # one chunk vs many tiny chunks
        monkeypatch.setattr(B, "MAX_CHUNK_CELLS", chunk_cells)
        got = B._topk(q, idx, 3, "x")
        for r in range(q.shape[0]):
            row = dense[r]
            want = {int(s) for s in np.argsort(-row)[:3] if row[s] > 1e-6}
            have = set(got.filter(pl.col("r") == r)["s"].to_list())
            # ties at the k-th score may be broken either way: compare scores, not ids
            assert sorted(row[list(have)].round(6)) == sorted(row[list(want)].round(6))


def test_ubiquitous_features_are_pruned_only_in_large_partitions():
    big = pl.DataFrame({"name_tok": [f"club w{i:05d}" for i in range(1500)], "addr": [f"{i} rue x" for i in range(1500)]})
    only_common = pl.DataFrame({"name_tok": ["club"], "addr": ["rue x"]})
    assert tfidf_candidates(big, only_common, k_name=3, k_comb=3).height == 0  # 'club'/'rue' pruned
    specific = pl.DataFrame({"name_tok": ["club w00042"], "addr": ["42 rue x"]})
    best = tfidf_candidates(big, specific, k_name=1, k_comb=1).sort("cos_comb", descending=True)
    assert best["s"][0] == 42
    # a tiny partition keeps every feature (fractional max_df would otherwise delete them all)
    assert tfidf_candidates(big.head(3), only_common, k_name=3, k_comb=3).height > 0


def test_cosine_features_do_not_depend_on_pruning(monkeypatch):
    import resolver.block as B
    big = pl.DataFrame({"name_tok": [f"club w{i:05d}" for i in range(1500)], "addr": [f"{i} rue x" for i in range(1500)]})
    q = pl.DataFrame({"name_tok": ["club w00042", "club w00007"], "addr": ["42 rue x", "7 rue x"]})
    monkeypatch.setattr(B, "MAX_DF", 1.0)
    full = tfidf_candidates(big, q, k_name=2, k_comb=2)
    monkeypatch.setattr(B, "MAX_DF", 0.05)
    pruned = tfidf_candidates(big, q, k_name=2, k_comb=2)
    both = full.join(pruned, on=["r", "s"], suffix="_p")
    assert both.height > 0
    for c in ("cos_name", "cos_addr", "cos_comb"):
        assert (both[c] - both[f"{c}_p"]).abs().max() < 1e-5
