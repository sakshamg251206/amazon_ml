"""E-C1: density-checked model consensus (label-free, test inputs only).

Vote pattern per test pair over 4 submissions [v1 ens, v2 ens, v3 ens, v2 M1]. Test has ~2x the decoy
records per S1 of train; stage-2 collective features are raw counts of competing records, so they shift.
Cell '111.' (all three stage-2 ensembles accept, stage-1 M1 rejects) occurs 5.2x more per S1 on US test
than on validation (0.092 vs 0.018) and its house numbers mismatch 90% vs 63%: the decoy signature
(docs: research/f05-gap). India's cell rate is 1.1x (kept). France '111.': 1.3-3.4x the US/India rates and
34% hn mismatch vs 1% in agreed France pairs (dropped). Output = v3 ensemble (sub4d) minus those cells.

Usage: python -m experiments.e_c1_consensus  -> output/sub6_consensus/
"""
import shutil

import polars as pl

from src.config import OUTPUT_DIR, WORK_DIR

SUBS = ["sub2d_ens_ef", "sub3d_ens_ef", "sub4d_ens_ef", "sub3b_m1_ef"]
BASE = "sub4d_ens_ef"
DROP = {("US", "111."), ("France", "111.")}
OUT = OUTPUT_DIR / "sub6_consensus"


def pairs(sub: str) -> pl.DataFrame:
    return (pl.scan_csv(OUTPUT_DIR / sub / "matching_results.tsv", separator="\t", quote_char=None,
                        schema_overrides={"matched_entity_ids": pl.String})
            .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").str.split(",").alias("rec"))
            .explode("rec", empty_as_null=True).filter(pl.col("rec").is_not_null() & (pl.col("rec") != "")).collect())


def patterns() -> pl.DataFrame:
    """(s1, rec, pat): pat[i] = '1' if SUBS[i] predicts the pair."""
    v = pl.concat([pairs(s).with_columns(pl.lit(1 << i, pl.UInt8).alias("m")) for i, s in enumerate(SUBS)])
    v = v.group_by("s1", "rec").agg(pl.col("m").sum())
    return v.with_columns(pl.concat_str([pl.when((pl.col("m") & (1 << i)) != 0).then(pl.lit("1")).otherwise(pl.lit("."))
                                         for i in range(len(SUBS))]).alias("pat")).drop("m")


def main() -> None:
    v = patterns()
    s1 = pl.read_parquet(WORK_DIR / "norm/test_s1.parquet", columns=["entity_id", "country"])
    v = v.join(s1.rename({"entity_id": "s1"}), on="s1")
    drop = pl.DataFrame(list(DROP), schema=["country", "pat"], orient="row")
    base = pairs(BASE)
    keep = base.join(v.join(drop, on=["country", "pat"]).select("s1", "rec"), on=["s1", "rec"], how="anti")
    print("dropped per country:", v.join(drop, on=["country", "pat"]).group_by("country").len().rows(), f"kept {keep.height:,} of {base.height:,}")
    OUT.mkdir(parents=True, exist_ok=True)
    lists = keep.group_by("s1").agg(pl.col("rec").unique().sort().str.join(",").alias("matched_entity_ids"))
    s1.select(pl.col("entity_id").alias("source1_entity_id")) \
      .join(lists.rename({"s1": "source1_entity_id"}), on="source1_entity_id", how="left", maintain_order="left").fill_null("") \
      .write_csv(OUT / "matching_results.tsv", separator="\t", quote_style="never")
    shutil.copy(OUTPUT_DIR / BASE / "candidate_pairs.tsv", OUT / "candidate_pairs.tsv")  # matches only shrink
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
