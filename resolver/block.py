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


def _topk(q: sp.csr_matrix, idx: sp.csr_matrix, k: int, view: str, idx_t: sp.csr_matrix | None = None) -> pl.DataFrame:
    """Top-k S1 per query row by cosine (rows are L2-normalised), scores > 1e-6 only.

    Pure scipy/numpy (replaces sparse_dot_topn, which crashed with EXC_BAD_ACCESS on Apple
    Silicon for dense French partitions). Queries are processed in chunks so no product exceeds
    MAX_CHUNK_CELLS cells; within a chunk each row's top-k comes from one vectorised sort on
    (row, -score). Chunks run on a small thread pool.
    """
    idx_t = idx.T.tocsr() if idx_t is None else idx_t
    step = max(1, MAX_CHUNK_CELLS // max(idx.shape[0], 1))
    starts = range(0, q.shape[0], step)
    with ThreadPoolExecutor(N_THREADS) as ex:
        parts = [p for p in ex.map(lambda st: _chunk_topk(q[st:st + step], idx_t, k, st), starts) if p]
    if not parts:
        return pl.DataFrame(schema={"r": pl.Int32, "s": pl.Int32, view: pl.Float32})
    return pl.DataFrame({"r": np.concatenate([p[0] for p in parts]).astype(np.int32),
                         "s": np.concatenate([p[1] for p in parts]).astype(np.int32),
                         view: np.concatenate([p[2] for p in parts]).astype(np.float32)})


class PartitionIndex:
    """TF-IDF index over one partition's S1 (vectorisers are fitted on S1 only, so the index is
    independent of the queries and can be reused: batch blocking and single-record lookups get
    identical cosines). Retrieval uses vectors pruned to features in <= MAX_DF of S1 (fast); the
    cosine FEATURES always use the full vectors, so their values match what the matcher learned."""

    def __init__(self, s1: pl.DataFrame):
        names, addrs = s1["name_tok"].to_list(), s1["addr"].to_list()
        self.n = len(names)
        max_df = MAX_DF if self.n >= MIN_S1_FOR_PRUNING else 1.0
        kw_n = dict(analyzer="char_wb", ngram_range=(3, 3), min_df=1, sublinear_tf=True)
        kw_a = dict(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.vn, self.n1 = _fit_one(names, **kw_n)
        self.va, self.a1 = _fit_one(addrs, **kw_a)
        if max_df < 1.0:
            self.vn_r, self.n1_r = _fit_one(names, max_df=max_df, **kw_n)
            self.va_r, self.a1_r = _fit_one(addrs, max_df=max_df, **kw_a)
        else:
            self.vn_r, self.n1_r, self.va_r, self.a1_r = self.vn, self.n1, self.va, self.a1
        self.c1_r = sp.hstack([np.sqrt(W_NAME) * self.n1_r, np.sqrt(W_ADDR) * self.a1_r]).tocsr()
        # transposes cached once: single-record lookups then cost one sparse product per view
        self._t = {"n": self.n1_r.T.tocsr(), "c": self.c1_r.T.tocsr(), "a": self.a1_r.T.tocsr()}

    def query(self, recs: pl.DataFrame, k_name: int, k_comb: int, k_addr: int = 0) -> pl.DataFrame:
        """recs: (name_tok, addr) -> (r, s, cos_name, cos_addr, cos_comb), r/s row indices into recs/S1."""
        if recs.height == 0 or self.n == 0:
            return pl.DataFrame(schema={"r": pl.Int32, "s": pl.Int32, "cos_name": pl.Float32,
                                        "cos_addr": pl.Float32, "cos_comb": pl.Float32})
        names, addrs = recs["name_tok"].to_list(), recs["addr"].to_list()
        nr_r, ar_r = _transform(self.vn_r, names, self.n1_r), _transform(self.va_r, addrs, self.a1_r)
        cr_r = sp.hstack([np.sqrt(W_NAME) * nr_r, np.sqrt(W_ADDR) * ar_r]).tocsr()
        views = [_topk(nr_r, self.n1_r, k_name, "x", self._t["n"]), _topk(cr_r, self.c1_r, k_comb, "x", self._t["c"])]
        if k_addr:  # address-only view: catches renamed businesses (name changed, address kept)
            views.append(_topk(ar_r, self.a1_r, k_addr, "x", self._t["a"]))
        pairs = pl.concat([v.select("r", "s") for v in views]).unique().sort("r", "s")
        nr = nr_r if self.vn is self.vn_r else _transform(self.vn, names, self.n1)
        ar = ar_r if self.va is self.va_r else _transform(self.va, addrs, self.a1)
        r, s = pairs["r"].to_numpy(), pairs["s"].to_numpy()
        cos_n, cos_a = _pair_cos(nr, self.n1, r, s), _pair_cos(ar, self.a1, r, s)
        return pairs.with_columns(pl.Series("cos_name", cos_n), pl.Series("cos_addr", cos_a),
                                  pl.Series("cos_comb", W_NAME * cos_n + W_ADDR * cos_a))


def _fit_one(texts: list[str], **kw):
    vec = TfidfVectorizer(dtype=np.float32, **kw)
    try:
        return vec, vec.fit_transform(texts).tocsr()
    except ValueError:  # empty vocabulary (e.g. all-empty addresses)
        return None, sp.csr_matrix((len(texts), 1), dtype=np.float32)


def _transform(vec, texts: list[str], fitted: sp.csr_matrix) -> sp.csr_matrix:
    if vec is None:
        return sp.csr_matrix((len(texts), fitted.shape[1]), dtype=np.float32)
    return vec.transform(texts).tocsr()


def _fit(s1_text: list[str], rec_text: list[str], **kw) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """(S1 matrix, record matrix) with vectorisers fitted on S1 (kept for experiments/e_b5_nostate)."""
    vec, a = _fit_one(s1_text, **kw)
    return a, _transform(vec, rec_text, a)


def tfidf_candidates(s1: pl.DataFrame, recs: pl.DataFrame, k_name: int, k_comb: int, k_addr: int = 0) -> pl.DataFrame:
    """s1/recs need columns name_tok, addr. Returns (r, s, cos_name, cos_addr, cos_comb) with r/s row indices
    into recs/s1. Cosines are filled for every returned pair, whichever view retrieved it."""
    if recs.height == 0 or s1.height == 0:
        return pl.DataFrame(schema={"r": pl.Int32, "s": pl.Int32, "cos_name": pl.Float32,
                                    "cos_addr": pl.Float32, "cos_comb": pl.Float32})
    return PartitionIndex(s1).query(recs, k_name, k_comb, k_addr)


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
