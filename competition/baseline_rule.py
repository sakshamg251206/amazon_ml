"""Submission-1 baseline: a record matches an S1 iff they share (country, name_key, house_no)
and that key is unique in S1. High precision, modest recall; no learning."""
import polars as pl

from resolver.normalize import house_no, name_key


def add_keys(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col("business_name").map_elements(name_key, return_dtype=pl.String).alias("nk"),
            pl.col("business_address").map_elements(house_no, return_dtype=pl.String).alias("hn"),
        )
        .filter((pl.col("nk") != "") & (pl.col("hn") != ""))
        .select("entity_id", pl.concat_str(["country", "nk", "hn"], separator="|").alias("key"))
    )


def rule_matches(s1: pl.DataFrame, recs: pl.DataFrame) -> dict[str, list[str]]:
    s1k = add_keys(s1).filter(pl.col("key").is_unique())
    rk = add_keys(recs).rename({"entity_id": "rec_id"})
    pairs = rk.join(s1k, on="key", how="inner").select("entity_id", "rec_id")
    out: dict[str, list[str]] = {}
    for s1_id, rec_id in pairs.iter_rows():
        out.setdefault(s1_id, []).append(rec_id)
    return out
