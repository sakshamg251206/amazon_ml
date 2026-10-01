"""Pair features for (S1, record) candidates. All features are country-agnostic by design:
`country` is never a feature, so France is scored by the same logic as US/India.

Groups:
  name     : fuzzy similarities of core names, exact core/space-free equality
  address  : fuzzy address similarities
  numbers  : house-number relation (equal / missing / differs) and number-set overlap
  context  : rank, gap to best and margin over second-best of this S1 among the record's
             candidates (each record has at most one true S1), and the reverse rank
  flags    : source, non-Latin name, empty address, blocking channel
"""
import polars as pl
from rapidfuzz import distance, fuzz
from rapidfuzz.process import cpdist

ATTRS = ["entity_id", "name_tok", "addr", "nums", "hn", "name_ns", "name_key", "nonlatin", "empty_addr"]


def _fuzz(a: pl.Series, b: pl.Series, scorer) -> pl.Series:
    return pl.Series(cpdist(a.to_list(), b.to_list(), scorer=scorer, workers=-1, dtype="float32"))


def pair_features(pairs: pl.DataFrame, s1: pl.DataFrame, recs: pl.DataFrame) -> pl.DataFrame:
    """pairs: (s1, rec, [cos_name, cos_addr, cos_comb, in_keys]). Returns pairs + feature columns."""
    a = s1.select(ATTRS).rename({c: f"{c}_1" for c in ATTRS}).rename({"entity_id_1": "s1"})
    b = recs.select(ATTRS).rename({c: f"{c}_2" for c in ATTRS}).rename({"entity_id_2": "rec"})
    d = pairs.join(a, on="s1").join(b, on="rec").sort("s1", "rec")  # canonical order: ordinal ranks break ties by it
    f = d.with_columns(
        _fuzz(d["name_tok_1"], d["name_tok_2"], fuzz.ratio).alias("name_ratio"),
        _fuzz(d["name_tok_1"], d["name_tok_2"], fuzz.token_set_ratio).alias("name_tset"),
        _fuzz(d["name_tok_1"], d["name_tok_2"], fuzz.partial_ratio).alias("name_partial"),
        _fuzz(d["name_tok_1"], d["name_tok_2"], distance.JaroWinkler.normalized_similarity).alias("name_jw"),
        _fuzz(d["name_ns_1"], d["name_ns_2"], fuzz.ratio).alias("ns_ratio"),
        _fuzz(d["addr_1"], d["addr_2"], fuzz.token_set_ratio).alias("addr_tset"),
        _fuzz(d["addr_1"], d["addr_2"], fuzz.partial_ratio).alias("addr_partial"),
        _fuzz(d["hn_1"], d["hn_2"], distance.Levenshtein.distance).alias("hn_lev"),
    ).with_columns(
        (pl.col("name_key_1") == pl.col("name_key_2")).alias("namekey_eq"),
        (pl.col("name_ns_1") == pl.col("name_ns_2")).alias("ns_eq"),
        (pl.col("hn_1") == pl.col("hn_2")).alias("hn_eq"),
        (pl.col("hn_2") == "").alias("hn_missing_rec"),
        (pl.col("hn_1") == "").alias("hn_missing_s1"),
        (pl.col("hn_1").str.starts_with(pl.col("hn_2")) | pl.col("hn_2").str.starts_with(pl.col("hn_1")))
        .alias("hn_prefix"),
        pl.col("nums_1").list.set_intersection("nums_2").list.len().alias("nums_shared"),
        (pl.col("nums_1").list.set_intersection("nums_2").list.len()
         / pl.col("nums_1").list.set_union("nums_2").list.len()).alias("nums_jac"),
        pl.col("nums_2").list.set_difference("nums_1").list.len().alias("nums_extra_rec"),
        pl.col("nums_1").list.set_difference("nums_2").list.len().alias("nums_missing_rec"),
        pl.col("rec").str.starts_with("S3").alias("src3"),
        pl.col("nonlatin_2").alias("nonlatin"),
        pl.col("empty_addr_2").alias("empty_addr"),
        pl.col("name_tok_2").str.len_chars().alias("len_name_rec"),
        pl.col("name_tok_1").str.len_chars().alias("len_name_s1"),
    )
    # A combined fuzzy score used for ranking context (independent of the blocking channel).
    f = f.with_columns((0.6 * pl.col("name_tset") + 0.4 * pl.col("addr_tset")).alias("fz"))
    rec_w, s1_w = ["rec"], ["s1"]
    f = f.with_columns(
        pl.col("fz").rank("ordinal", descending=True).over(rec_w).alias("rank_fz"),
        (pl.col("fz").max().over(rec_w) - pl.col("fz")).alias("gap_fz"),
        pl.col("fz").rank("ordinal", descending=True).over(s1_w).alias("rev_rank_fz"),
        pl.len().over(rec_w).alias("n_cand_rec"),
        pl.len().over(s1_w).alias("n_cand_s1"),
    ).with_columns(
        # margin of the best candidate over the runner-up (0 for non-best candidates' reference)
        (pl.col("fz") - pl.col("fz").filter(pl.col("rank_fz") == 2).first().over(rec_w)).fill_null(100.0)
        .alias("margin_fz"),
    )
    return f.drop([c for c in f.columns if c.endswith("_1") or c.endswith("_2")])


FEATURES = ["name_ratio", "name_tset", "name_partial", "name_jw", "ns_ratio", "addr_tset", "addr_partial",
            "hn_lev", "namekey_eq", "ns_eq", "hn_eq", "hn_missing_rec", "hn_missing_s1", "hn_prefix",
            "nums_shared", "nums_jac", "nums_extra_rec", "nums_missing_rec", "src3", "nonlatin", "empty_addr",
            "len_name_rec", "len_name_s1", "fz", "rank_fz", "gap_fz", "rev_rank_fz", "n_cand_rec",
            "n_cand_s1", "margin_fz", "cos_name", "cos_addr", "cos_comb", "in_keys"]
