"""E-G1: graph / transitive evidence among S2+S3 records (ground-truth analysis, no model).

Twin edges: two records (any source) sharing an exact key. Components = connected groups
(union of key groups, via iterative min-label propagation = BFS on the key graph).

Questions:
  1. Purity: do records in one component share one owner (or are all unowned)?
  2. Phantoms: do unmatched records cluster together (distractor entities with several copies)?
  3. Transitive recall: of true pairs missed by E-B3 candidates, how many have a twin that IS a
     candidate of the right S1 (recoverable by propagating along twin edges)?
  4. Transitive precision: propagate every candidate along twin edges; share of new pairs true.
"""
import time

import polars as pl

from src.config import WORK_DIR
from src.evalx import candidate_metrics, log_experiment, truth_pairs

COLS = ["entity_id", "country", "name_key", "name_ns", "hn", "addr"]
MAX_GROUP = 12  # key groups bigger than this are chains/generic names, not twins


def twin_keys(df: pl.DataFrame) -> pl.DataFrame:
    c = pl.col("country")
    return df.select("entity_id", pl.concat_str([c, pl.lit("A"), pl.col("name_key"), pl.col("hn")], separator="|").alias("k1"),
                     pl.concat_str([c, pl.lit("B"), pl.col("name_ns"), pl.col("addr")], separator="|").alias("k2")).with_columns(
        pl.when(pl.col("k1").str.ends_with("|")).then(None).otherwise(pl.col("k1")).alias("k1"),
        pl.when(pl.col("k2").str.ends_with("|")).then(None).otherwise(pl.col("k2")).alias("k2"))


def components(recs: pl.DataFrame) -> pl.DataFrame:
    """Connected components over key groups by min-label propagation. Returns (entity_id, comp)."""
    k = twin_keys(recs)
    edges = pl.concat([k.select("entity_id", pl.col(c).alias("key")).drop_nulls() for c in ("k1", "k2")])
    edges = edges.filter(pl.len().over("key").is_between(2, MAX_GROUP))
    lab = edges.select("entity_id").unique().with_columns(pl.col("entity_id").alias("comp"))
    for it in range(30):
        m = edges.join(lab, on="entity_id").group_by("key").agg(pl.col("comp").min().alias("kmin"))
        new = edges.join(m, on="key").group_by("entity_id").agg(pl.col("kmin").min().alias("comp"))
        changed = new.join(lab, on="entity_id").filter(pl.col("comp") != pl.col("comp_right")).height
        lab = new
        if changed == 0:
            break
    print(f"components converged after {it + 1} iterations", flush=True)
    return lab


def main() -> None:
    t0 = time.time()
    recs = pl.concat([pl.read_parquet(WORK_DIR / f"norm/train_s{s}.parquet", columns=COLS) for s in (2, 3)])
    truth = truth_pairs()
    owner = truth.drop_nulls().select(pl.col("rec").alias("entity_id"), pl.col("s1").alias("owner"))
    comp = components(recs).join(owner, on="entity_id", how="left")
    del recs
    g = comp.group_by("comp").agg(pl.len().alias("n"), pl.col("owner").n_unique().alias("n_owner_vals"),
                                  pl.col("owner").null_count().alias("n_unowned"))
    pure = (g["n_owner_vals"] == 1).mean()
    phantom = g.filter(pl.col("n_unowned") == pl.col("n")).height
    matched_mixed = g.filter((pl.col("n_unowned") > 0) & (pl.col("n_unowned") < pl.col("n"))).height
    log_experiment("E-G1:components", {
        "n_components": g.height, "records_in_components": comp.height, "purity": pure,
        "phantom_components": phantom, "phantom_share": phantom / g.height,
        "mixed_owned_unowned": matched_mixed / g.height, "mean_size": g["n"].mean()}, t0)

    # Transitive propagation on the E-B3 sample.
    sample = pl.read_parquet(WORK_DIR / "e_b3_sample_s1.parquet")["entity_id"]
    cand = pl.read_parquet(WORK_DIR / "e_b3_union_cands.parquet")
    base = candidate_metrics(cand, truth, sample)
    c2 = cand.join(comp.select(pl.col("entity_id").alias("rec"), "comp"), on="rec")
    prop = c2.join(comp.select(pl.col("entity_id").alias("rec2"), "comp"), on="comp") \
             .select("s1", pl.col("rec2").alias("rec")).unique() \
             .join(cand, on=["s1", "rec"], how="anti")
    tp = prop.join(truth.drop_nulls(), on=["s1", "rec"])
    after = candidate_metrics(pl.concat([cand, prop]), truth, sample)
    log_experiment("E-G1:propagation", {
        "cand_recall_before": base["cand_recall"], "cand_recall_after": after["cand_recall"],
        "oracle_before": base["oracle_f05"], "oracle_after": after["oracle_f05"],
        "new_pairs": prop.height, "new_pairs_true_share": tp.height / max(prop.height, 1)}, t0)


if __name__ == "__main__":
    main()
