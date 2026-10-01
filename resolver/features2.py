"""Competition + difference features (E-F2), added on top of src.features.FEATURES.

Motivation (research/f05-gap): the best documented team's LightGBM put 0.66 + 0.10 of its gain on
record-side margin / is-best features computed on the blocker's TF-IDF score, not on token-set
fuzz (which scores "Acme" vs "Acme Holdings" as 100). On our OOF: the record's cosine-best S1 is a
different S1 for 18.5% of confident false positives vs 0.7% of confident true matches.

competition_features needs every candidate of a record AND of an S1 in the same frame (it groups
both ways); diff_features is per pair and can run chunk-wise.
"""
import polars as pl

COMP = ["rev_is_best_cos", "rev_margin_cos", "rev_rel_cos", "rev_rel_name",
        "ctx_rel_cos", "ctx_gap_cos", "ctx_rel_name", "ctx_rel_addr", "ctx_is_best_cos"]
DIFF = ["n_tok_extra", "n_tok_miss", "hn_delta", "hn_absdelta", "hn_substr"]
NEW = COMP + DIFF
REV = ["rev_is_best_cos", "rev_margin_cos", "rev_rel_cos", "rev_rel_name"]


def mask_keyonly(df: pl.DataFrame) -> pl.DataFrame:
    """Blank record-side features on key-only pairs (no cosine). Validation reaches records outside
    the sampled states only through keys, so there all their candidates have cosine 0 and look
    'best'; on test those records also have TF-IDF candidates in their own state. Blanking makes
    the feature mean the same thing in both (E-F2b)."""
    return df.with_columns(pl.when(pl.col("cos_comb").is_null()).then(None).otherwise(pl.col(c)).alias(c) for c in REV)


def _cna(pairs: pl.DataFrame | pl.LazyFrame):
    c, n, a = (pl.col(x).fill_null(0.0) for x in ("cos_comb", "cos_name", "cos_addr"))
    return pairs.select("s1", "rec", c.alias("c"), n.alias("n"), a.alias("a"))


def s1_maxima(pairs: pl.DataFrame | pl.LazyFrame):
    """Per S1: best combined / name / address cosine over ALL its candidates (can stream)."""
    return _cna(pairs).group_by("s1").agg(pl.col("c").max().alias("s_best"), pl.col("n").max().alias("s_best_n"),
                                           pl.col("a").max().alias("s_best_a"))


def competition_features(pairs: pl.DataFrame, s1_best: pl.DataFrame | None = None) -> pl.DataFrame:
    """pairs: (s1, rec, cos_name, cos_addr, cos_comb) -> (s1, rec, *COMP). Missing cosines (pairs
    found only by exact keys) count as 0 for ranking, like a score the blocker never gave.
    Record-side features need all of a record's candidates in `pairs`; S1-side maxima come from
    `s1_best` (see s1_maxima) when `pairs` holds only part of each S1's candidates."""
    if s1_best is None:
        s1_best = s1_maxima(pairs)
    d = _cna(pairs).join(s1_best, on="s1", how="left").with_columns(
        pl.col("c").max().over("rec").alias("r_best"),
        pl.col("c").sort(descending=True).slice(1, 1).first().over("rec").fill_null(0.0).alias("r_second"),
        pl.col("n").max().over("rec").alias("r_best_n"),
    )
    return d.select(
        "s1", "rec",
        (pl.col("c") >= pl.col("r_best")).cast(pl.Float32).alias("rev_is_best_cos"),
        pl.when(pl.col("c") >= pl.col("r_best")).then(pl.col("c") - pl.col("r_second"))
          .otherwise(pl.col("c") - pl.col("r_best")).alias("rev_margin_cos"),
        (pl.col("c") / pl.col("r_best").clip(1e-6)).alias("rev_rel_cos"),
        (pl.col("n") / pl.col("r_best_n").clip(1e-6)).alias("rev_rel_name"),
        (pl.col("c") / pl.col("s_best").clip(1e-6)).alias("ctx_rel_cos"),
        (pl.col("s_best") - pl.col("c")).alias("ctx_gap_cos"),
        (pl.col("n") / pl.col("s_best_n").clip(1e-6)).alias("ctx_rel_name"),
        (pl.col("a") / pl.col("s_best_a").clip(1e-6)).alias("ctx_rel_addr"),
        (pl.col("c") >= pl.col("s_best")).cast(pl.Float32).alias("ctx_is_best_cos"),
    ).with_columns(pl.col(x).cast(pl.Float32) for x in COMP)


def diff_features(pairs: pl.DataFrame, s1: pl.DataFrame, recs: pl.DataFrame) -> pl.DataFrame:
    """pairs: (s1, rec); s1/recs: (entity_id, name_tok, hn) -> (s1, rec, *DIFF).
    Word counts use exact tokens; typos show up as one extra + one missing word, which the fuzzy
    name features already disambiguate."""
    a = s1.select(pl.col("entity_id").alias("s1"), pl.col("name_tok").str.split(" ").alias("t1"), pl.col("hn").alias("h1"))
    b = recs.select(pl.col("entity_id").alias("rec"), pl.col("name_tok").str.split(" ").alias("t2"), pl.col("hn").alias("h2"))
    d = pairs.select("s1", "rec").join(a, on="s1", how="left").join(b, on="rec", how="left")
    h1 = pl.col("h1").str.slice(0, 9).cast(pl.Int64, strict=False)
    h2 = pl.col("h2").str.slice(0, 9).cast(pl.Int64, strict=False)
    return d.select(
        "s1", "rec",
        pl.col("t2").list.set_difference("t1").list.len().alias("n_tok_extra"),
        pl.col("t1").list.set_difference("t2").list.len().alias("n_tok_miss"),
        (h2 - h1).clip(-10_000, 10_000).alias("hn_delta"),
        (h2 - h1).abs().clip(0, 10_000).alias("hn_absdelta"),
        pl.when(pl.col("h1").is_null() | pl.col("h2").is_null() | (pl.col("h1") == "") | (pl.col("h2") == ""))
          .then(None).otherwise((pl.col("h1").str.contains(pl.col("h2"), literal=True)
                                 | pl.col("h2").str.contains(pl.col("h1"), literal=True)).cast(pl.Int8)).alias("hn_substr"),
    ).with_columns(pl.col(x).cast(pl.Float32) for x in DIFF)


if __name__ == "__main__":
    p = pl.DataFrame({"s1": ["A", "A", "B"], "rec": ["r1", "r2", "r1"],
                      "cos_name": [0.9, 0.5, 0.8], "cos_addr": [0.8, 0.9, 0.2], "cos_comb": [0.86, 0.66, 0.56]})
    c = competition_features(p).sort("s1", "rec")
    assert c.filter((pl.col("s1") == "A") & (pl.col("rec") == "r1"))["rev_is_best_cos"].item() == 1.0
    assert abs(c.filter((pl.col("s1") == "A") & (pl.col("rec") == "r1"))["rev_margin_cos"].item() - 0.30) < 1e-6
    assert abs(c.filter((pl.col("s1") == "B"))["rev_margin_cos"].item() + 0.30) < 1e-6        # r1 prefers A
    assert c.filter((pl.col("s1") == "A") & (pl.col("rec") == "r2"))["ctx_is_best_cos"].item() == 0.0
    s1 = pl.DataFrame({"entity_id": ["A"], "name_tok": ["yazzie empire"], "hn": ["338"]})
    rc = pl.DataFrame({"entity_id": ["r1", "r2"], "name_tok": ["yazzie empire holdings", "yazzie empire"], "hn": ["352", "33"]})
    d = diff_features(pl.DataFrame({"s1": ["A", "A"], "rec": ["r1", "r2"]}), s1, rc).sort("rec")
    assert d["n_tok_extra"].to_list() == [1.0, 0.0] and d["hn_delta"].to_list() == [14.0, -305.0]
    assert d["hn_substr"].to_list() == [0.0, 1.0]                                                # 338 contains 33
    print("features2 self-check ok")
