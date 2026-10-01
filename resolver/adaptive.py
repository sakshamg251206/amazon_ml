"""Adaptive name vocabulary (E-T2): learn, per dataset and country, which name tokens are generic.

Our stop words / legal forms are a fixed list. Fake "neighbour" records add GENERIC descriptor
words (holdings, group, enterprises; in France e.g. developpement, participations) while true
copies differ in RARE tokens (typos, transliteration). Generic = token in >= SHARE of that
country's S1 names in the same split. Label-free, so it is fitted on test S1 for test (France gets
its own list) and on train S1 for validation.
"""
import polars as pl
from rapidfuzz import fuzz

from resolver.features import _fuzz

SHARE = 0.001
GEN = ["n_extra_generic", "n_extra_rare", "n_miss_generic", "n_miss_rare", "core_tset", "core_eq"]


def generic_tokens(s1: pl.DataFrame, share: float = SHARE) -> pl.DataFrame:
    """s1: (country, name_tok) -> (country, tok) of generic tokens."""
    t = s1.select("country", pl.col("name_tok").str.split(" ").list.unique().alias("tok")).explode("tok") \
          .filter(pl.col("tok").is_not_null() & (pl.col("tok") != ""))
    n = s1.group_by("country").len("n_s1")
    return t.group_by("country", "tok").len("df").join(n, on="country") \
            .filter(pl.col("df") >= share * pl.col("n_s1")).select("country", "tok")


def _split(ents: pl.DataFrame, gen: pl.DataFrame, country: pl.DataFrame) -> pl.DataFrame:
    """ents: (id, name_tok); country: (id, country) -> (id, gen tokens list, rare tokens list)."""
    t = ents.join(country, on="id").select("id", "country", pl.col("name_tok").str.split(" ").alias("tok")).explode("tok") \
            .filter(pl.col("tok").is_not_null() & (pl.col("tok") != ""))
    t = t.join(gen.with_columns(pl.lit(True).alias("g")), on=["country", "tok"], how="left").with_columns(pl.col("g").fill_null(False))
    return t.group_by("id").agg(pl.col("tok").filter(pl.col("g")).alias("gen"), pl.col("tok").filter(~pl.col("g")).alias("rare"))


def generic_features(pairs: pl.DataFrame, s1: pl.DataFrame, recs: pl.DataFrame, gen: pl.DataFrame) -> pl.DataFrame:
    """pairs: (s1, rec); s1: (entity_id, country, name_tok); recs: (entity_id, name_tok).
    A record is judged with its S1's country vocabulary (pairs never cross countries)."""
    sc = s1.select(pl.col("entity_id").alias("id"), "country")
    a = _split(s1.select(pl.col("entity_id").alias("id"), "name_tok"), gen, sc).rename({"id": "s1", "gen": "g1", "rare": "r1"})
    rc = pairs.join(sc.rename({"id": "s1"}), on="s1").select(pl.col("rec").alias("id"), "country").unique("id")
    b = _split(recs.select(pl.col("entity_id").alias("id"), "name_tok"), gen, rc).rename({"id": "rec", "gen": "g2", "rare": "r2"})
    empty = pl.lit([], dtype=pl.List(pl.String))
    d = pairs.select("s1", "rec").join(a, on="s1", how="left").join(b, on="rec", how="left") \
             .with_columns(pl.col("g1", "r1", "g2", "r2").fill_null(empty))
    c1, c2 = d["r1"].list.join(" "), d["r2"].list.join(" ")
    return d.select(
        "s1", "rec",
        pl.col("g2").list.set_difference("g1").list.len().alias("n_extra_generic"),
        pl.col("r2").list.set_difference("r1").list.len().alias("n_extra_rare"),
        pl.col("g1").list.set_difference("g2").list.len().alias("n_miss_generic"),
        pl.col("r1").list.set_difference("r2").list.len().alias("n_miss_rare"),
        _fuzz(c1, c2, fuzz.token_set_ratio).alias("core_tset"),
        (pl.col("r1").list.sort() == pl.col("r2").list.sort()).alias("core_eq"),
    ).with_columns(pl.col(x).cast(pl.Float32) for x in GEN)


if __name__ == "__main__":
    s1 = pl.DataFrame({"entity_id": [f"s{i}" for i in range(1000)] + ["A"], "country": "US",
                       "name_tok": [f"w{i} holdings" if i % 2 else f"w{i} group" for i in range(1000)] + ["yazzie empire"]})
    gen = generic_tokens(s1)
    assert set(gen["tok"]) == {"holdings", "group"}, gen
    recs = pl.DataFrame({"entity_id": ["r1", "r2"], "name_tok": ["yazzie empire holdings", "yazzie empyre"]})
    f = generic_features(pl.DataFrame({"s1": ["A", "A"], "rec": ["r1", "r2"]}), s1, recs, gen).sort("rec")
    assert f["n_extra_generic"].to_list() == [1.0, 0.0] and f["n_extra_rare"].to_list() == [0.0, 1.0]   # fake vs typo copy
    assert f["core_eq"].to_list() == [1.0, 0.0]
    print("adaptive self-check ok")
