"""Vocabulary class of unmatched name tokens (E-T3): copy noise vs real-word swap.

True copies differ from their S1 by noise that creates tokens seen nowhere else (typos, transliteration:
'daniels hawk' -> 'daniels hrawk', 'apex food' -> 'ayapeks phud'). Decoys swap in another REAL word of the name
vocabulary at the same address ('daniels hawk' -> 'hanners hawk', 'sanchez trinity' -> 'meyers trinity').
DF = number of this split's S1 names (per country) containing the token: label-free, so test is fitted on
test S1 (France gets its own vocabulary). Generic tokens (src/adaptive.py) are their own class.
"""
import polars as pl
from rapidfuzz import fuzz

from src.features import _fuzz

VOCAB_MIN_DF = 2   # an extra token used by >= 2 other S1 names is a real word, not copy noise
MAX_TOK = 4        # align at most 4 extra x 4 missing tokens per pair
VOC = ["n_extra_vocab", "n_extra_unseen", "tok_align", "vocab_align"]


def token_df(s1: pl.DataFrame) -> pl.DataFrame:
    """s1: (country, name_tok) -> (country, tok, df)."""
    t = s1.select("country", pl.col("name_tok").str.split(" ").list.unique().alias("tok")).explode("tok", empty_as_null=True) \
          .filter(pl.col("tok").is_not_null() & (pl.col("tok") != ""))
    return t.group_by("country", "tok").len("df")


def _tokens(ids: pl.DataFrame, names: pl.DataFrame, key: str) -> pl.DataFrame:
    return ids.select(key).unique().join(names.rename({"entity_id": key}), on=key) \
              .select(key, pl.col("name_tok").str.split(" ").list.unique().list.drop_nulls().alias(f"t_{key}"))


def vocab_features(pairs: pl.DataFrame, s1: pl.DataFrame, recs: pl.DataFrame, df: pl.DataFrame, gen: pl.DataFrame) -> pl.DataFrame:
    """pairs: (s1, rec); s1: (entity_id, country, name_tok); recs: (entity_id, name_tok);
    df: token_df of this split; gen: generic_tokens of this split -> (s1, rec, *VOC)."""
    d = pairs.select("s1", "rec").with_row_index("i") \
             .join(s1.select(pl.col("entity_id").alias("s1"), "country"), on="s1", how="left") \
             .join(_tokens(pairs, s1, "s1"), on="s1", how="left").join(_tokens(pairs, recs, "rec"), on="rec", how="left")
    empty = pl.lit([], dtype=pl.List(pl.String))
    d = d.with_columns(pl.col("t_s1", "t_rec").fill_null(empty)).select(
        "i", "country", pl.col("t_rec").list.set_difference("t_s1").list.filter(pl.element() != "").alias("extra"),
        pl.col("t_s1").list.set_difference("t_rec").list.filter(pl.element() != "").alias("miss"))
    ex = d.select("i", "country", pl.col("extra").list.head(MAX_TOK)).explode("extra", empty_as_null=True).drop_nulls("extra") \
          .join(df.rename({"tok": "extra"}), on=["country", "extra"], how="left") \
          .join(gen.rename({"tok": "extra"}).with_columns(pl.lit(True).alias("g")), on=["country", "extra"], how="left") \
          .with_columns(pl.col("df").fill_null(0), pl.col("g").fill_null(False))
    ex = ex.with_columns((~pl.col("g") & (pl.col("df") >= VOCAB_MIN_DF)).alias("voc"), (pl.col("df") < VOCAB_MIN_DF).alias("unseen"))
    cross = ex.join(d.select("i", pl.col("miss").list.head(MAX_TOK)).explode("miss", empty_as_null=True).drop_nulls("miss"), on="i")
    cross = cross.with_columns(_fuzz(cross["extra"], cross["miss"], fuzz.ratio).alias("r"))
    best = cross.group_by("i", "extra").agg(pl.col("r").max(), pl.col("voc").first())       # best match per extra token
    agg = best.group_by("i").agg(pl.col("r").max().alias("tok_align"),
                                 pl.col("r").filter(pl.col("voc")).min().alias("vocab_align"))
    cnt = ex.group_by("i").agg(pl.col("voc").sum().alias("n_extra_vocab"), pl.col("unseen").sum().alias("n_extra_unseen"))
    out = pairs.select("s1", "rec").with_row_index("i").join(cnt, on="i", how="left").join(agg, on="i", how="left")
    out = out.join(ex.filter(pl.col("voc")).select("i").unique().with_columns(pl.lit(True).alias("has_voc")), on="i", how="left")
    return out.select(
        "s1", "rec",
        pl.col("n_extra_vocab").fill_null(0), pl.col("n_extra_unseen").fill_null(0),
        pl.col("tok_align").fill_null(-1.0),                                               # -1: nothing to align
        pl.when(pl.col("has_voc") & pl.col("vocab_align").is_null()).then(0.0)              # real word added, nothing removed
          .otherwise(pl.col("vocab_align")).fill_null(-1.0).alias("vocab_align"),
    ).with_columns(pl.col(x).cast(pl.Float32) for x in VOC)


if __name__ == "__main__":
    s1 = pl.DataFrame({"entity_id": [f"s{i}" for i in range(6)] + ["A"], "country": "US",
                       "name_tok": ["hanners law", "hanners co", "x hanners", "hawk y", "daniels z", "q r", "daniels hawk"]})
    recs = pl.DataFrame({"entity_id": ["decoy", "typo", "same"], "name_tok": ["hanners hawk", "daniels hrawk", "daniels hawk"]})
    df = token_df(s1)
    gen = pl.DataFrame({"country": [], "tok": []}, schema={"country": pl.String, "tok": pl.String})
    f = vocab_features(pl.DataFrame({"s1": ["A"] * 3, "rec": ["decoy", "typo", "same"]}), s1, recs, df, gen).sort("rec")
    r = {x["rec"]: x for x in f.iter_rows(named=True)}
    assert r["decoy"]["n_extra_vocab"] == 1 and r["decoy"]["n_extra_unseen"] == 0 and r["decoy"]["vocab_align"] < 60, r["decoy"]
    assert r["typo"]["n_extra_vocab"] == 0 and r["typo"]["n_extra_unseen"] == 1 and r["typo"]["tok_align"] > 85, r["typo"]
    assert r["same"]["tok_align"] == -1 and r["same"]["vocab_align"] == -1, r["same"]
    print("vocab self-check ok")
