"""E-X1: where the remaining macro F0.5 is lost (current best system, OOF).

Predictions = E-E2 stage-2 mean (LightGBM + ExtraTrees + MLP) on pairs with p1 >= 0.01, then
exclusive + expected-F decoding, over the 414k validation S1. For every error category we report
its size and the F0.5 we would gain if exactly that category were fixed (errors removed / misses
added), holding everything else fixed. Categories are mutually exclusive, assigned in order.
"""
import numpy as np
import polars as pl

from experiments.e_d1_decode import f05, ntrue_table
from src.config import WORK_DIR
from src.decode import ef_decode, exclusive
from src.evalx import truth_pairs
from src.partition import infer_state

ENS2 = WORK_DIR / "ens2"


def main() -> None:
    nt = ntrue_table()                                         # s1(hash), country, n_true
    d = pl.read_parquet(ENS2 / "data.parquet", columns=["s1", "rec", "y", "fold", "name_tset", "name_ratio", "namekey_eq",
                                                        "addr_tset", "hn_eq", "hn_missing_rec", "hn_missing_s1",
                                                        "empty_addr", "nonlatin", "n_tok_extra", "cos_comb"])
    p = np.mean([pl.read_parquet(ENS2 / f"oof_{m}.parquet")["p"].to_numpy() for m in ("lgbm", "extratrees", "mlp")], axis=0)
    pred = ef_decode(exclusive(d.select("s1", "rec", "y").with_columns(pl.Series("p", p)))).select("s1", "rec", "y")
    base = f05(pred, nt)
    print(f"current: f05={base['f05']:.4f} precision={base['precision']:.4f} recall={base['recall']:.4f} predicted={pred.height:,}")

    # truth for the validation S1 (hashed), and which records are real copies of *some* S1
    t_all = truth_pairs().drop_nulls()
    owner = t_all.select(pl.col("rec").hash().alias("rec"), pl.col("s1").hash().alias("owner"))
    sample = pl.read_parquet(WORK_DIR / "e_b3_sample_s1.parquet").select(pl.col("entity_id").hash().alias("s1"))
    truth = t_all.select(pl.col("s1").hash(), pl.col("rec").hash()).join(sample, on="s1")
    cands = pl.scan_parquet(WORK_DIR / "e_f2b_feats.parquet").select(pl.col("s1").hash(), pl.col("rec").hash()).collect()

    feats = d.drop("y", "fold")
    fp = pred.filter(pl.col("y") == 0).join(feats, on=["s1", "rec"], how="left").join(owner, on="rec", how="left") \
             .join(nt.select("s1", "n_true"), on="s1", how="left")
    fp = fp.with_columns(pl.when(pl.col("owner").is_not_null()).then(pl.lit("FP swap: record belongs to another S1"))
        .when(pl.col("n_true") == 0).then(pl.lit("FP on a singleton S1 (distractor)"))
        .when(pl.col("namekey_eq") > 0.5).then(pl.lit("FP distractor: same name key"))
        .when((pl.col("name_tset") >= 90) & (pl.col("hn_eq") < 0.5) & (pl.col("hn_missing_rec") < 0.5) & (pl.col("hn_missing_s1") < 0.5))
          .then(pl.lit("FP distractor: similar name, different house no"))
        .when((pl.col("name_tset") < 70) & (pl.col("addr_tset") >= 85)).then(pl.lit("FP distractor: different name, same address"))
        .when(pl.col("name_tset") >= 90).then(pl.lit("FP distractor: similar name, other"))
        .otherwise(pl.lit("FP distractor: other")).alias("cat"))

    missed = truth.join(pred.select("s1", "rec"), on=["s1", "rec"], how="anti")
    in_c = missed.join(cands, on=["s1", "rec"], how="semi")
    blk = missed.join(cands, on=["s1", "rec"], how="anti")
    fn = in_c.join(feats, on=["s1", "rec"], how="left").with_columns(
        pl.when(pl.col("name_tset").is_null()).then(pl.lit("FN model: pruned at stage 1 (p1<0.01)"))
        .when(pl.col("empty_addr") > 0.5).then(pl.lit("FN model: record has empty address"))
        .when(pl.col("nonlatin") > 0.5).then(pl.lit("FN model: non-Latin record"))
        .when(pl.col("name_tset") < 50).then(pl.lit("FN model: renamed (name sim < 50)"))
        .when((pl.col("hn_eq") < 0.5) & (pl.col("hn_missing_rec") < 0.5) & (pl.col("hn_missing_s1") < 0.5)).then(pl.lit("FN model: house numbers differ"))
        .otherwise(pl.lit("FN model: other low-confidence")).alias("cat")).select("s1", "rec", "cat")
    # blocking misses: classify by the record's inferred state
    cols = ["entity_id", "country", "parts"]
    recs = pl.concat([pl.read_parquet(WORK_DIR / f"norm/train_s{s}.parquet", columns=cols) for s in (2, 3)]) \
             .with_columns(pl.col("entity_id").hash().alias("rec")).join(blk.select("rec").unique(), on="rec")
    recs = recs.with_columns(infer_state(recs, pl.read_parquet(WORK_DIR / "alias_train.parquet")).alias("rstate"))
    s1st = pl.read_parquet(WORK_DIR / "e_b3_sample_s1.parquet").select(pl.col("entity_id").hash().alias("s1"), "state")
    blk = blk.join(recs.select("rec", "rstate"), on="rec", how="left").join(s1st, on="s1", how="left").with_columns(
        pl.when(pl.col("rstate").is_null()).then(pl.lit("FN blocking: record has no state"))
        .when(pl.col("rstate") != pl.col("state")).then(pl.lit("FN blocking: record in another state"))
        .otherwise(pl.lit("FN blocking: same state, not retrieved")).alias("cat")).select("s1", "rec", "cat")

    rows = []
    for cat, g in fp.group_by("cat"):
        fixed = pred.join(g.select("s1", "rec"), on=["s1", "rec"], how="anti")
        rows.append((cat[0], g.height, f05(fixed, nt)["f05"] - base["f05"]))
    for cat, g in pl.concat([fn, blk]).group_by("cat"):
        fixed = pl.concat([pred, g.select("s1", "rec").with_columns(pl.lit(1, pl.Int8).alias("y"))], how="vertical_relaxed")
        rows.append((cat[0], g.height, f05(fixed, nt)["f05"] - base["f05"]))
    out = pl.DataFrame(rows, schema=["category", "pairs", "f05_gain_if_fixed"], orient="row").sort("f05_gain_if_fixed", descending=True)
    pl.Config.set_tbl_rows(30); pl.Config.set_fmt_str_lengths(60)
    print(out)
    out.write_csv(WORK_DIR / "e_x1_error_audit.csv")
    print(f"all FPs fixed: +{f05(pred.filter(pl.col('y') == 1), nt)['f05'] - base['f05']:.4f}   "
          f"all model FNs fixed: +{f05(pl.concat([pred, in_c.with_columns(pl.lit(1, pl.Int8).alias('y'))], how='vertical_relaxed'), nt)['f05'] - base['f05']:.4f}   "
          f"all blocking FNs fixed: +{f05(pl.concat([pred, blk.select('s1', 'rec').with_columns(pl.lit(1, pl.Int8).alias('y'))], how='vertical_relaxed'), nt)['f05'] - base['f05']:.4f}")


if __name__ == "__main__":
    main()
