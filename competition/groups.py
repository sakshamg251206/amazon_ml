"""Entity-group gains (E-G2): judge a record together with its near-copy siblings among the same S1's candidates.

Copy noise is per record (one copy has a typo, its siblings don't); a decoy differs per entity. Group of a
pair (s1, r) = the S1's other candidates r' that are near-copies of r (name ratio >= NAME, address token-set
>= ADDR or either address empty, same house number when both have one, so a shifted-number decoy never joins
the true copies). Features are GAINS over the pair itself (best sibling minus self, floored at 0), never counts:
duplicating a decoy adds a sibling with gain 0, so the features do not move with decoy density (E-P2 lesson).
"""
import polars as pl
from rapidfuzz import fuzz

from resolver.features import _fuzz

NAME, ADDR = 85.0, 80.0
GRP = ["g_name_gain", "g_addr_gain", "g_p1_gain"]   # no presence flag: a duplicated decoy would flip it


def group_features(pairs: pl.DataFrame, recs: pl.DataFrame) -> pl.DataFrame:
    """pairs: (s1, rec, p1, name_tset, addr_tset) = one split's stage-2 pairs; recs: (entity_id, name_tok, addr, hn)
    -> (s1, rec, *GRP)."""
    r = recs.select(pl.col("entity_id").alias("rec"), "name_tok", "addr", "hn")
    a = pairs.select("s1", "rec", "p1", "name_tset", "addr_tset").join(r, on="rec", how="left")
    b = a.rename({c: c + "_b" for c in a.columns if c != "s1"})
    x = a.select("s1", "rec", "name_tok", "addr", "hn").join(b, on="s1").filter(pl.col("rec") != pl.col("rec_b"))
    x = x.with_columns(_fuzz(x["name_tok"].fill_null(""), x["name_tok_b"].fill_null(""), fuzz.ratio).alias("rr_name"),
                       _fuzz(x["addr"].fill_null(""), x["addr_b"].fill_null(""), fuzz.token_set_ratio).alias("rr_addr"))
    empty = lambda c: pl.col(c).is_null() | (pl.col(c) == "")
    close = x.filter((pl.col("rr_name") >= NAME) & ((pl.col("rr_addr") >= ADDR) | empty("addr") | empty("addr_b"))
                     & (empty("hn") | empty("hn_b") | (pl.col("hn") == pl.col("hn_b"))))
    g = close.group_by("s1", "rec").agg(pl.col("name_tset_b").max().alias("g_name"), pl.col("addr_tset_b").max().alias("g_addr"),
                                        pl.col("p1_b").max().alias("g_p1"))
    out = pairs.select("s1", "rec", "p1", "name_tset", "addr_tset").join(g, on=["s1", "rec"], how="left")
    return out.select("s1", "rec",
                      (pl.col("g_name") - pl.col("name_tset")).clip(0).fill_null(0).alias("g_name_gain"),
                      (pl.col("g_addr") - pl.col("addr_tset")).clip(0).fill_null(0).alias("g_addr_gain"),
                      (pl.col("g_p1") - pl.col("p1")).clip(0).fill_null(0).alias("g_p1_gain"),
                      ).with_columns(pl.col(c).cast(pl.Float32) for c in GRP)


if __name__ == "__main__":
    pairs = pl.DataFrame({"s1": ["A"] * 4, "rec": ["clean", "typo", "decoy", "shift"], "p1": [0.99, 0.4, 0.7, 0.6],
                          "name_tset": [100.0, 80.0, 75.0, 100.0], "addr_tset": [100.0, 100.0, 100.0, 90.0]})
    recs = pl.DataFrame({"entity_id": ["clean", "typo", "decoy", "shift"],
                         "name_tok": ["daniels hawk", "daniels hrawk", "hanners hawk", "daniels hawk"],
                         "addr": ["1500 polaris pkwy", "1500 polaris pkwy", "1500 polaris pkwy", "1502 polaris pkwy"],
                         "hn": ["1500", "1500", "1500", "1502"]})
    f = {r["rec"]: r for r in group_features(pairs, recs).iter_rows(named=True)}
    assert f["typo"]["g_p1_gain"] > 0.5 and f["typo"]["g_name_gain"] == 20, f["typo"]      # rescued by its clean sibling
    assert f["decoy"]["g_p1_gain"] == 0 and f["decoy"]["g_addr_gain"] == 0, f["decoy"]     # name swap: not a sibling
    assert f["shift"]["g_p1_gain"] == 0 and f["shift"]["g_addr_gain"] == 0, f["shift"]     # shifted number: not a sibling
    assert f["clean"]["g_p1_gain"] == 0, f["clean"]
    dup = group_features(pl.concat([pairs, pairs.filter(pl.col("rec") == "decoy").with_columns(pl.lit("decoy2").alias("rec"))]),
                         pl.concat([recs, recs.filter(pl.col("entity_id") == "decoy").with_columns(pl.lit("decoy2").alias("entity_id"))]))
    d = {r["rec"]: r for r in dup.iter_rows(named=True)}
    assert d["decoy"]["g_p1_gain"] == 0 and d["decoy"]["g_name_gain"] == 0, d["decoy"]     # duplicate decoy: gains stay 0
    print("groups self-check ok")
