"""Raw-name similarity (E-T5): what normalisation throws away.

Half of all train S1 (1.11M of 2.2M) share their normalised name key with another S1: different businesses that
differ only in stripped details ('Mandsaur Brothers Limited' vs 'Mandsaur & Brothers Ltd'; legal forms, E-T4).
These features compare names that are only transliterated, lower-cased and whitespace-collapsed, keeping '&',
stopwords, legal forms and punctuation tokens.
"""
import re

import polars as pl
from anyascii import anyascii
from rapidfuzz import fuzz

from resolver.features import _fuzz

RAW = ["raw_ratio", "raw_tsort", "raw_eq", "raw_small_diff"]
_WS = re.compile(r"\s+")
_SMALL = {"&", "and", "the", "of", "+", "-", "'s"}


def raw_norm(name: str | None) -> str:
    return _WS.sub(" ", anyascii(name or "").lower()).strip()


def raw_names(raw: pl.DataFrame) -> pl.DataFrame:
    """raw: (entity_id, business_name) -> (entity_id, rn); computed once per distinct name."""
    u = raw.select("business_name").unique().with_columns(
        pl.col("business_name").map_elements(raw_norm, return_dtype=pl.String, skip_nulls=False).alias("rn"))
    return raw.join(u, on="business_name", how="left", nulls_equal=True).select("entity_id", pl.col("rn").fill_null(""))


def raw_features(pairs: pl.DataFrame, rn1: pl.DataFrame, rn2: pl.DataFrame) -> pl.DataFrame:
    """pairs: (s1, rec); rn1 / rn2: raw_names of S1 / records -> (s1, rec, *RAW)."""
    d = pairs.select("s1", "rec").join(rn1.rename({"entity_id": "s1", "rn": "a"}), on="s1", how="left") \
             .join(rn2.rename({"entity_id": "rec", "rn": "b"}), on="rec", how="left").with_columns(pl.col("a", "b").fill_null(""))
    small = lambda c: pl.col(c).str.replace_all(r"([&+])", " $1 ").str.split(" ").list.eval(pl.element().filter(pl.element().is_in(list(_SMALL))))
    return d.select("s1", "rec",
                    _fuzz(d["a"], d["b"], fuzz.ratio).alias("raw_ratio"),
                    _fuzz(d["a"], d["b"], fuzz.token_sort_ratio).alias("raw_tsort"),
                    (pl.col("a") == pl.col("b")).alias("raw_eq"),
                    small("a").list.set_symmetric_difference(small("b")).list.len().alias("raw_small_diff"),
                    ).with_columns(pl.col(c).cast(pl.Float32) for c in RAW)


if __name__ == "__main__":
    raw = pl.DataFrame({"entity_id": ["A", "B", "r"], "business_name": ["Mandsaur Brothers Limited", "Mandsaur & Brothers Ltd", "Mandsaur &  Brothers Ltd"]})
    rn = raw_names(raw)
    f = raw_features(pl.DataFrame({"s1": ["A", "B"], "rec": ["r", "r"]}), rn, rn).sort("s1")
    assert f["raw_eq"].to_list() == [0.0, 1.0] and f["raw_small_diff"].to_list() == [1.0, 0.0], f
    assert f["raw_ratio"][1] == 100 and f["raw_ratio"][0] < 90, f
    print("rawname self-check ok")
