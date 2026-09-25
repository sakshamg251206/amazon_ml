"""Candidate generation: TF-IDF top-k retrieval S2/S3 -> S1 inside one (country, state) partition.

Views (cosine similarity on L2-normalised TF-IDF vectors):
  name : char 3-grams of the core name (typos, spacing, transliteration noise)
  addr : word 1-2-grams of the cleaned address
  comb : sqrt(0.6)*name ++ sqrt(0.4)*addr, so comb cosine = 0.6*cos_name + 0.4*cos_addr
Each record keeps its top-k S1 under the name view, the combined view and (optionally)
the address-only view.
"""
import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

W_NAME, W_ADDR = 0.6, 0.4


def _fit(s1_text: list[str], rec_text: list[str], **kw) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    vec = TfidfVectorizer(dtype=np.float32, **kw)
    try:
        a = vec.fit_transform(s1_text)
    except ValueError:  # empty vocabulary (e.g. all-empty addresses)
        return sp.csr_matrix((len(s1_text), 1), dtype=np.float32), sp.csr_matrix((len(rec_text), 1), dtype=np.float32)
    return a, vec.transform(rec_text)


def _topk(q: sp.csr_matrix, idx: sp.csr_matrix, k: int, view: str) -> pl.DataFrame:
    m = sp_matmul_topn(q, idx.T.tocsr(), top_n=k, threshold=1e-6, n_threads=8).tocoo()
    return pl.DataFrame({"r": m.row.astype(np.int32), "s": m.col.astype(np.int32), view: m.data})


def tfidf_candidates(s1: pl.DataFrame, recs: pl.DataFrame, k_name: int, k_comb: int, k_addr: int = 0) -> pl.DataFrame:
    """s1/recs need columns name_tok, addr. Returns (r, s, cos_name, cos_comb) with r/s row indices
    into recs/s1. Cosines are filled for every returned pair, whichever view retrieved it."""
    if recs.height == 0 or s1.height == 0:
        return pl.DataFrame(schema={"r": pl.Int32, "s": pl.Int32, "cos_name": pl.Float32,
                                    "cos_addr": pl.Float32, "cos_comb": pl.Float32})
    n1, nr = _fit(s1["name_tok"].to_list(), recs["name_tok"].to_list(),
                  analyzer="char_wb", ngram_range=(3, 3), min_df=1, sublinear_tf=True)
    a1, ar = _fit(s1["addr"].to_list(), recs["addr"].to_list(),
                  analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    c1 = sp.hstack([np.sqrt(W_NAME) * n1, np.sqrt(W_ADDR) * a1]).tocsr()
    cr = sp.hstack([np.sqrt(W_NAME) * nr, np.sqrt(W_ADDR) * ar]).tocsr()
    views = [_topk(nr, n1, k_name, "x"), _topk(cr, c1, k_comb, "x")]
    if k_addr:  # address-only view: catches renamed businesses (name changed, address kept)
        views.append(_topk(ar, a1, k_addr, "x"))
    pairs = pl.concat([v.select("r", "s") for v in views]).unique()
    r, s = pairs["r"].to_numpy(), pairs["s"].to_numpy()
    cos_n = np.asarray(nr[r].multiply(n1[s]).sum(axis=1)).ravel()
    cos_a = np.asarray(ar[r].multiply(a1[s]).sum(axis=1)).ravel()
    return pairs.with_columns(pl.Series("cos_name", cos_n), pl.Series("cos_addr", cos_a),
                              pl.Series("cos_comb", W_NAME * cos_n + W_ADDR * cos_a))
