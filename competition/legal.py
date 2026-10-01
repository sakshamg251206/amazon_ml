"""Legal-form agreement (E-T4): the one name token normalisation throws away.

S1 holds different businesses that differ only by legal form ('Crandall Enterprises P.C.' vs 'Crandall
Enterprises Inc'; 'Surgical Advanced Medicine LLC' vs '... Inc'), but name_tok / name_ns / name_key strip legal
forms, so both S1 score the same. Validation candidate pairs: legal forms equal -> 0.90 true, different -> 0.54.
Computed from RAW names (norm parquet has none); canonical groups merge spelling variants only.
"""
import polars as pl

from resolver.normalize import clean

CANON = {"inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "co": "co", "company": "co",
         "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt", "llc": "llc", "llp": "llp", "lp": "lp",
         "pc": "pc", "plc": "plc", "pllc": "pllc", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl",
         "sci": "sci", "sa": "sa"}
LF = ["lf_equal", "lf_diff", "lf_one_side"]


def legal_form(name: str | None) -> str:
    return " ".join(sorted({CANON[t] for t in clean(name or "").split() if t in CANON}))


def legal_forms(raw: pl.DataFrame) -> pl.DataFrame:
    """raw: (entity_id, business_name) -> (entity_id, lf); computed once per distinct name."""
    u = raw.select("business_name").unique().with_columns(
        pl.col("business_name").map_elements(legal_form, return_dtype=pl.String, skip_nulls=False).alias("lf"))
    return raw.join(u, on="business_name", how="left", nulls_equal=True).select("entity_id", pl.col("lf").fill_null(""))


def legal_features(pairs: pl.DataFrame, lf1: pl.DataFrame, lf2: pl.DataFrame) -> pl.DataFrame:
    """pairs: (s1, rec); lf1 / lf2: legal_forms of S1 / records -> (s1, rec, *LF)."""
    d = pairs.select("s1", "rec").join(lf1.rename({"entity_id": "s1", "lf": "a"}), on="s1", how="left") \
             .join(lf2.rename({"entity_id": "rec", "lf": "b"}), on="rec", how="left").with_columns(pl.col("a", "b").fill_null(""))
    one = (pl.col("a") == "") != (pl.col("b") == "")
    both = (pl.col("a") != "") & (pl.col("b") != "")
    return d.select("s1", "rec", (both & (pl.col("a") == pl.col("b"))).alias("lf_equal"),
                    (both & (pl.col("a") != pl.col("b"))).alias("lf_diff"), one.alias("lf_one_side")) \
            .with_columns(pl.col(c).cast(pl.Float32) for c in LF)


if __name__ == "__main__":
    assert legal_form("Crandall Enterprises P.C.") == "pc" and legal_form("Crandall  Enterprises Inc") == "inc"
    assert legal_form("Acme Incorporated") == legal_form("ACME INC.") == "inc" and legal_form(None) == ""
    raw = pl.DataFrame({"entity_id": ["A", "B", "r"], "business_name": ["Crandall Enterprises P.C.", "Crandall Enterprises Inc", "Crandall  Enterprises Inc"]})
    lfs = legal_forms(raw)
    f = legal_features(pl.DataFrame({"s1": ["A", "B"], "rec": ["r", "r"]}), lfs, lfs).sort("s1")
    assert f["lf_diff"].to_list() == [1.0, 0.0] and f["lf_equal"].to_list() == [0.0, 1.0], f
    print("legal self-check ok")
