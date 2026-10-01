"""Stage-2 collective (entity-level) and graph-twin features (E-M2, E-G1).

Evidence from OTHER records improves distractor discrimination: how strongly other records claim
the same S1, where this record ranks among all claimants, and whether this record's exact-key
"twins" (records sharing a strict key) point to the same S1. Graph evidence is used softly, as
features, never as hard merges (D8: propagating along twin edges adds pairs that are 0.15% true).

All features are computed from stage-1 probabilities `p` that must be out-of-fold during training,
so no pair's features come from a model that saw its own label.
"""
import polars as pl

COLLECTIVE = ["p1", "p1_rank_rec", "p1_margin_rec", "p1_sum_rec", "claims_s1", "best_other_p1_s1",
              "rank_among_claimants", "n_twins", "twins_agree", "twins_best_p1_same"]
TWIN_COLS = ["entity_id", "country", "name_key", "name_ns", "hn", "addr"]
MAX_GROUP = 12  # key groups bigger than this are chains / generic names, not twins


def twin_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Two strict keys per record: (name_key, house number) and (space-free name, full address)."""
    c = pl.col("country")
    return df.select("entity_id", pl.concat_str([c, pl.lit("A"), pl.col("name_key"), pl.col("hn")], separator="|").alias("k1"),
                     pl.concat_str([c, pl.lit("B"), pl.col("name_ns"), pl.col("addr")], separator="|").alias("k2")).with_columns(
        pl.when(pl.col("k1").str.ends_with("|")).then(None).otherwise(pl.col("k1")).alias("k1"),
        pl.when(pl.col("k2").str.ends_with("|")).then(None).otherwise(pl.col("k2")).alias("k2"))


def twin_edges(recs: pl.DataFrame) -> pl.DataFrame:
    """(entity_id, key) for every key shared by 2..MAX_GROUP records."""
    k = twin_keys(recs)
    edges = pl.concat([k.select("entity_id", pl.col(c).alias("key")).drop_nulls() for c in ("k1", "k2")])
    return edges.filter(pl.len().over("key").is_between(2, MAX_GROUP))


def components(recs: pl.DataFrame, max_iter: int = 30) -> pl.DataFrame:
    """Connected components over twin-key groups by min-label propagation -> (entity_id, comp).
    Records without any twin are absent."""
    edges = twin_edges(recs)
    lab = edges.select("entity_id").unique().with_columns(pl.col("entity_id").alias("comp"))
    for _ in range(max_iter):
        m = edges.join(lab, on="entity_id").group_by("key").agg(pl.col("comp").min().alias("kmin"))
        new = edges.join(m, on="key").group_by("entity_id").agg(pl.col("kmin").min().alias("comp"))
        changed = new.join(lab, on="entity_id").filter(pl.col("comp") != pl.col("comp_right")).height
        lab = new
        if changed == 0:
            break
    return lab


def collective(scored: pl.DataFrame, comp: pl.DataFrame) -> pl.DataFrame:
    """scored: (s1, rec, p) stage-1 probabilities; comp: (entity_id, comp) -> (s1, rec, *COLLECTIVE)."""
    s = scored.select("s1", "rec", pl.col("p").alias("p1"))
    s = s.with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("rec").alias("p1_rank_rec"),
        pl.col("p1").sum().over("rec").alias("p1_sum_rec"),
    ).with_columns(
        (pl.col("p1") - pl.col("p1").filter(pl.col("p1_rank_rec") == 2).first().over("rec")).fill_null(pl.col("p1"))
        .alias("p1_margin_rec"),
        pl.col("p1").rank("ordinal", descending=True).over("s1").alias("rank_among_claimants"),
    )
    # claims: records whose BEST candidate is this S1 with p1 >= 0.5 (excluding this record)
    best = s.filter(pl.col("p1_rank_rec") == 1)
    claims = best.filter(pl.col("p1") >= 0.5).group_by("s1").len("claims_all")
    s = s.join(claims, on="s1", how="left").with_columns(
        (pl.col("claims_all").fill_null(0) - ((pl.col("p1_rank_rec") == 1) & (pl.col("p1") >= 0.5)).cast(pl.UInt32))
        .alias("claims_s1"))
    # best p1 of any OTHER record on this S1: top-2 per S1, take the one that is not this record
    top2 = s.sort("p1", descending=True).group_by("s1", maintain_order=True).head(2) \
            .group_by("s1").agg(pl.col("rec"), pl.col("p1").alias("tp"))
    s = s.join(top2, on="s1").with_columns(
        pl.when(pl.col("rec") == pl.col("rec_right").list.get(0))
          .then(pl.col("tp").list.get(1, null_on_oob=True)).otherwise(pl.col("tp").list.get(0))
          .fill_null(0.0).alias("best_other_p1_s1")).drop("rec_right", "tp")
    # graph twins: do this record's twins pick the same S1 as their best candidate?
    tw = comp.select(pl.col("entity_id").alias("rec"), "comp")
    best_s1 = best.select("rec", pl.col("s1").alias("twin_best"), pl.col("p1").alias("twin_p1")).join(tw, on="rec")
    pairs_tw = s.select("s1", "rec").join(tw, on="rec").join(best_s1.rename({"rec": "twin"}), on="comp") \
                .filter(pl.col("twin") != pl.col("rec"))
    agg = pairs_tw.group_by("s1", "rec").agg(
        pl.len().alias("n_twins"),
        (pl.col("twin_best") == pl.col("s1")).sum().alias("twins_agree"),
        pl.when(pl.col("twin_best") == pl.col("s1")).then(pl.col("twin_p1")).max().alias("twins_best_p1_same"))
    s = s.join(agg, on=["s1", "rec"], how="left").with_columns(
        pl.col("n_twins", "twins_agree").fill_null(0), pl.col("twins_best_p1_same").fill_null(0.0))
    return s.drop("claims_all").with_columns(pl.col(c).cast(pl.Float32) for c in COLLECTIVE)
