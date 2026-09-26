"""Candidate generation: TF-IDF top-k retrieval S2/S3 -> S1 inside one (country, state) partition.

Views (cosine similarity on L2-normalised TF-IDF vectors):
  name : char 3-grams of the core name (typos, spacing, transliteration noise)
  addr : word 1-2-grams of the cleaned address
  comb : sqrt(0.6)*name ++ sqrt(0.4)*addr, so comb cosine = 0.6*cos_name + 0.4*cos_addr
Each record keeps its top-k S1 under the name view, the combined view and (optionally)
the address-only view.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

W_NAME, W_ADDR = 0.6, 0.4
# Features present in more than this share of a partition's S1 are dropped: they carry almost
# no IDF weight but make the query x S1 product nearly dense (the cost driver). Chosen by E-B4.
MAX_DF = 0.05
MIN_S1_FOR_PRUNING = 1_000  # a fractional max_df on tiny partitions would delete every feature


def _fit(s1_text: list[str], rec_text: list[str], **kw) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    vec = TfidfVectorizer(dtype=np.float32, **kw)
    try:
        a = vec.fit_transform(s1_text)
    except ValueError:  # empty vocabulary (e.g. all-empty addresses)
        return sp.csr_matrix((len(s1_text), 1), dtype=np.float32), sp.csr_matrix((len(rec_text), 1), dtype=np.float32)
    return a, vec.transform(rec_text)


MAX_CHUNK_CELLS = 5_000_000  # bounds each chunk's product to <= this many (query x S1) cells
N_THREADS = 6  # scipy sparse matmul and numpy sort release the GIL, so threads run in parallel


def _chunk_topk(q: sp.csr_matrix, idx_t: sp.csr_matrix, k: int, start: int):
    prod = (q @ idx_t).tocsr()
    if prod.nnz == 0:
        return None
    row_of = np.repeat(np.arange(prod.shape[0], dtype=np.int64), np.diff(prod.indptr))
    order = np.lexsort((-prod.data, row_of))
    row_sorted = row_of[order]
    rank = np.arange(order.size) - prod.indptr[row_sorted]  # position within its row
    keep = (rank < k) & (prod.data[order] > 1e-6)
    return row_sorted[keep] + start, prod.indices[order][keep], prod.data[order][keep]


def _topk(q: sp.csr_matrix, idx: sp.csr_matrix, k: int, view: str) -> pl.DataFrame:
    """Top-k S1 per query row by cosine (rows are L2-normalised), scores > 1e-6 only.

    Pure scipy/numpy (replaces sparse_dot_topn, which crashed with EXC_BAD_ACCESS on Apple
    Silicon for dense French partitions). Queries are processed in chunks so no product exceeds
    MAX_CHUNK_CELLS cells; within a chunk each row's top-k comes from one vectorised sort on
    (row, -score). Chunks run on a small thread pool.
    """
    idx_t = idx.T.tocsr()
    step = max(1, MAX_CHUNK_CELLS // max(idx.shape[0], 1))
    starts = range(0, q.shape[0], step)
    with ThreadPoolExecutor(N_THREADS) as ex:
        parts = [p for p in ex.map(lambda st: _chunk_topk(q[st:st + step], idx_t, k, st), starts) if p]
    if not parts:
        return pl.DataFrame(schema={"r": pl.Int32, "s": pl.Int32, view: pl.Float32})
    return pl.DataFrame({"r": np.concatenate([p[0] for p in parts]).astype(np.int32),
                         "s": np.concatenate([p[1] for p in parts]).astype(np.int32),
                         view: np.concatenate([p[2] for p in parts]).astype(np.float32)})


def tfidf_candidates(s1: pl.DataFrame, recs: pl.DataFrame, k_name: int, k_comb: int, k_addr: int = 0) -> pl.DataFrame:
    """s1/recs need columns name_tok, addr. Returns (r, s, cos_name, cos_comb) with r/s row indices
    into recs/s1. Cosines are filled for every returned pair, whichever view retrieved it."""
    if recs.height == 0 or s1.height == 0:
        return pl.DataFrame(schema={"r": pl.Int32, "s": pl.Int32, "cos_name": pl.Float32,
                                    "cos_addr": pl.Float32, "cos_comb": pl.Float32})
    names = (s1["name_tok"].to_list(), recs["name_tok"].to_list())
    addrs = (s1["addr"].to_list(), recs["addr"].to_list())
    max_df = MAX_DF if s1.height >= MIN_S1_FOR_PRUNING else 1.0
    # Retrieval uses pruned vectors (fast: common features dropped); cosine FEATURES always use
    # the full vectors, so their values match what the matcher was trained on.
    n1, nr = _fit(*names, analyzer="char_wb", ngram_range=(3, 3), min_df=1, max_df=max_df, sublinear_tf=True)
    a1, ar = _fit(*addrs, analyzer="word", ngram_range=(1, 2), min_df=1, max_df=max_df, sublinear_tf=True)
    c1 = sp.hstack([np.sqrt(W_NAME) * n1, np.sqrt(W_ADDR) * a1]).tocsr()
    cr = sp.hstack([np.sqrt(W_NAME) * nr, np.sqrt(W_ADDR) * ar]).tocsr()
    views = [_topk(nr, n1, k_name, "x"), _topk(cr, c1, k_comb, "x")]
    if k_addr:  # address-only view: catches renamed businesses (name changed, address kept)
        views.append(_topk(ar, a1, k_addr, "x"))
    pairs = pl.concat([v.select("r", "s") for v in views]).unique()
    if max_df < 1.0:
        n1, nr = _fit(*names, analyzer="char_wb", ngram_range=(3, 3), min_df=1, sublinear_tf=True)
        a1, ar = _fit(*addrs, analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    r, s = pairs["r"].to_numpy(), pairs["s"].to_numpy()
    cos_n, cos_a = _pair_cos(nr, n1, r, s), _pair_cos(ar, a1, r, s)
    return pairs.with_columns(pl.Series("cos_name", cos_n), pl.Series("cos_addr", cos_a),
                              pl.Series("cos_comb", W_NAME * cos_n + W_ADDR * cos_a))


def _pair_cos(q: sp.csr_matrix, idx: sp.csr_matrix, r: np.ndarray, s: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    """Row-wise dot products q[r_i] . idx[s_i], in chunks to bound memory."""
    out = [np.asarray(q[r[i:i + chunk]].multiply(idx[s[i:i + chunk]]).sum(axis=1)).ravel()
           for i in range(0, len(r), chunk)]
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


# ---- exact-key channel (country-wide) ----
CAP = 20
ADDR_STOP = ["door", "no", "unit", "flat", "plot", "suite", "apt", "block", "floor", "house",
             "hno", "shop", "office", "room", "building", "bldg", "near", "opp"]


def with_keys(df: pl.DataFrame) -> pl.DataFrame:
    street = (pl.col("addr").str.extract_all(r"[a-z]{3,}")
              .list.eval(pl.element().filter(~pl.element().is_in(ADDR_STOP))).list.first())
    c = pl.col("country")
    return df.with_columns(
        pl.when(pl.col("hn") != "").then(pl.concat_str([c, pl.col("name_key"), pl.col("hn")], separator="|")).alias("k_v0"),
        pl.when(pl.col("name_ns").str.len_chars() >= 4).then(pl.concat_str([c, pl.col("name_ns")], separator="|")).alias("k_ns"),
        pl.when(pl.col("hn") != "").then(pl.concat_str([c, pl.col("hn"), street], separator="|")).alias("k_hn_street"),
        pl.when(pl.col("hn") != "").then(pl.concat_str([c, pl.col("name_tok").str.split(" ").list.first(), pl.col("hn")], separator="|")).alias("k_name1_hn"),
        pl.when(pl.col("name_key") != "").then(pl.concat_str([c, pl.col("name_key")], separator="|")).alias("k_namekey"),
    )


KEYS = ["k_v0", "k_ns", "k_hn_street", "k_name1_hn", "k_namekey"]


def key_candidates(s1: pl.DataFrame, recs: pl.DataFrame) -> pl.DataFrame:
    """Union of exact-key matches (s1, rec). Key groups with more than CAP S1 (chains) are skipped."""
    s1k, rk = with_keys(s1), with_keys(recs)
    out = []
    for k in KEYS:
        b = s1k.filter(pl.col(k).is_not_null()).select(pl.col("entity_id").alias("s1"), k).filter(pl.len().over(k) <= CAP)
        out.append(rk.filter(pl.col(k).is_not_null()).select(pl.col("entity_id").alias("rec"), k).join(b, on=k).select("s1", "rec"))
    return pl.concat(out).unique()
