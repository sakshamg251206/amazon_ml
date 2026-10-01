"""E-C2: vote-level stacked ensemble of every trained model, density-corrected for test.

Voters = the decoded outputs of 7 models (each validated out-of-fold, decoded exactly as its test file).
Stacker: q(country, vote pattern) = smoothed precision of validation pairs with that pattern
(shrunk towards pairs with the same number of votes). Scored cross-fitted: fit on one state fold,
decide on the other. Test: test has ~2x the decoys per S1, so q_test = q x min(1, val rate / test rate)
per (country, pattern): true copies per S1 are the same in train and test (3.46), so any excess of a
pattern on test is decoys (E-C1 / D24). France (no labels) uses the pooled US+India cells and rates.
Validation: a plain q >= T stacker scores 0.9630 cross-fitted, BELOW the best voter v5ens (0.9639): a per-cell
threshold loses the per-S1 context of each model's expected-F decoding. So the shipped rule keeps v5ens and uses
all 7 voters only to REMOVE its pairs whose cell is over-represented on test (density-corrected q < T). On
validation that is exactly v5ens (rates match by construction); rate_val uses an upper bound (n + 2 sqrt n + 3)
so small noisy cells are not removed.

Usage: python -m experiments.e_c2_stack [val|test]   (test needs every voter's submission in output/)
"""
import sys
import time

import numpy as np
import polars as pl

from experiments.e_d1_decode import f05, ntrue_table
from competition.config import OUTPUT_DIR, SEED, WORK_DIR
from resolver.decode import ef_decode, exclusive
from competition.evalx import log_experiment, truth_pairs

VOTERS = [  # (name, validation source, test submission)
    ("v1m1", "m1:e_m1_oof_k5", "sub2b_m1_ef"),
    ("v1ens", "ens:ens", "sub2d_ens_ef"),
    ("v2m1", "m1:e_f2b_m1_oof", "sub3b_m1_ef"),
    ("v2ens", "ens:ens2", "sub3d_ens_ef"),
    ("v3ens", "ens:ens3g", "sub4d_ens_ef"),
    ("v5m1", "m1:e_f2b_gen_ca_m1_oof", "sub5b_m1_ef"),
    ("v5ens", "ens:ens3gca", "sub5d_ens_ef"),
    ("v5nt", "ens:ens3gca_no_twins", "sub5notwinsd_ens_ef"),   # E-P2: stage 2 without twin features
    ("v5ntg", "ens:ens3gca_no_twins_grp", "sub5notwinsgrpd_ens_ef"),   # E-G2: + entity-group gains
]
BASE = "v5ntg"  # voter whose pairs are kept unless density-flagged
K = len(VOTERS)
A = 10.0                      # shrinkage pseudo-count towards the same-vote-count precision
TS = np.round(np.arange(0.30, 0.91, 0.025), 3)
CELLS = WORK_DIR / "e_c2_cells.parquet"
OUT = OUTPUT_DIR / "sub8_stack"   # sub7 = 7 voters, base v5ens; sub8 = 9 voters, base v5ntg


def _val_decoded(src: str) -> pl.DataFrame:
    kind, name = src.split(":")
    if kind == "ens":
        x = pl.read_parquet(WORK_DIR / name / "data.parquet", columns=["s1", "rec", "y"])
        p = np.mean([pl.read_parquet(WORK_DIR / name / f"oof_{m}.parquet")["p"].to_numpy() for m in ("lgbm", "extratrees", "mlp")], axis=0)
        x = x.with_columns(pl.Series("p", p))
    else:
        x = pl.read_parquet(WORK_DIR / f"{name}.parquet", columns=["s1", "rec", "y", "p"]).select(pl.col("s1").hash(), pl.col("rec").hash(), "y", "p")
    return ef_decode(exclusive(x)).select("s1", "rec")


def _pattern(v: pl.DataFrame) -> pl.DataFrame:
    """v: (s1, rec, m bit) rows -> (s1, rec, pat, votes)."""
    v = v.group_by("s1", "rec").agg(pl.col("m").sum())
    return v.with_columns(pl.concat_str([pl.when((pl.col("m") & (1 << i)) != 0).then(pl.lit("1")).otherwise(pl.lit("."))
                                         for i in range(K)]).alias("pat"),
                          pl.col("m").cast(pl.UInt32).map_elements(lambda m: bin(m).count("1"), return_dtype=pl.Int32).alias("votes")).drop("m")


def _folds() -> pl.DataFrame:
    sample = pl.read_parquet(WORK_DIR / "e_b3_sample_s1.parquet")
    parts = sorted(set(sample.select("country", "state").unique().rows()))
    fold_of = {p: int(f) for p, f in zip(parts, np.random.default_rng(SEED).permutation(len(parts)) % 2)}
    return sample.select(pl.col("entity_id").hash().alias("s1"),
                         pl.struct("country", "state").map_elements(lambda r: fold_of[(r["country"], r["state"])], return_dtype=pl.Int8).alias("fold"))


def val_pairs() -> pl.DataFrame:
    v = pl.concat([_val_decoded(src).with_columns(pl.lit(1 << i, pl.UInt8).alias("m")) for i, (_, src, _) in enumerate(VOTERS)])
    v = _pattern(v)
    t = truth_pairs().drop_nulls().select(pl.col("s1").hash(), pl.col("rec").hash()).with_columns(pl.lit(1, pl.Int8).alias("y"))
    nt = ntrue_table()
    return v.join(t, on=["s1", "rec"], how="left").with_columns(pl.col("y").fill_null(0)) \
            .join(nt.select("s1", "country"), on="s1").join(_folds(), on="s1")


def fit_cells(d: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Smoothed precision per (by + pat), shrunk towards the precision of pairs with the same vote count."""
    prior = d.group_by(by + ["votes"]).agg(pl.col("y").mean().alias("prior"))
    c = d.group_by(by + ["pat", "votes"]).agg(pl.len().alias("n"), pl.col("y").sum().alias("tp"))
    return c.join(prior, on=by + ["votes"]).with_columns(((pl.col("tp") + A * pl.col("prior")) / (pl.col("n") + A)).alias("q"))


def _decide(d: pl.DataFrame, t: float) -> pl.DataFrame:
    return exclusive(d.filter(pl.col("q") >= t).select("s1", "rec", "y", pl.col("q").alias("p")))


def _score(pred: pl.DataFrame, nt: pl.DataFrame, s1: pl.DataFrame) -> float:
    return f05(pred.join(s1, on="s1"), nt.join(s1, on="s1"))["f05"]


def validate() -> None:
    t0 = time.time()
    d, nt = val_pairs(), ntrue_table()
    sf = _folds()
    ids = {f: sf.filter(pl.col("fold") == f).select("s1") for f in (0, 1)}
    for i, (name, _, _) in enumerate(VOTERS):   # each voter alone, same S1 set
        pred = d.filter(pl.col("pat").str.slice(i, 1) == "1")
        log_experiment(f"E-C2:voter:{name}", {"f05": f05(pred, nt)["f05"], "fold0": _score(pred, nt, ids[0]), "fold1": _score(pred, nt, ids[1])}, t0)
    out, chosen = [], {}
    for f in (0, 1):                             # cross-fitted: cells + threshold from fold f, applied to 1-f
        fit, app = d.filter(pl.col("fold") == f), d.filter(pl.col("fold") != f)
        cells = fit_cells(fit, ["country"])
        fs = fit.join(cells.select("country", "pat", "q"), on=["country", "pat"])
        best = max(TS, key=lambda t: _score(_decide(fs, t), nt, ids[f]))
        chosen[f] = float(best)
        ap = app.join(cells.select("country", "pat", "q", "votes"), on=["country", "pat", "votes"], how="left") \
                .with_columns(pl.col("q").fill_null(0.0))
        out.append(_decide(ap, best))
    pred = pl.concat(out)
    log_experiment("E-C2:stack_xfit", {"f05": f05(pred, nt)["f05"], "fold0": _score(pred, nt, ids[0]),
                                       "fold1": _score(pred, nt, ids[1]), "T_fold0": chosen[0], "T_fold1": chosen[1]}, t0)
    cells = pl.concat([fit_cells(d, ["country"]),
                       fit_cells(d, []).with_columns(pl.lit("pooled").alias("country")).select(fit_cells(d, ["country"]).columns)])
    cells.with_columns(pl.lit(float(np.mean(list(chosen.values())))).alias("T")).write_parquet(CELLS)
    print(f"cells -> {CELLS}; T = {np.mean(list(chosen.values())):.3f}", flush=True)


def _test_pairs(sub: str) -> pl.LazyFrame:
    return (pl.scan_csv(OUTPUT_DIR / sub / "matching_results.tsv", separator="\t", quote_char=None,
                        schema_overrides={"matched_entity_ids": pl.String})
            .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").str.split(",").alias("rec"))
            .explode("rec", empty_as_null=True).filter(pl.col("rec").is_not_null() & (pl.col("rec") != "")))


def test() -> None:
    cells = pl.read_parquet(CELLS)
    T = cells["T"][0]
    n_val = {"US": 249883, "India": 163858}
    n_val["pooled"] = n_val["US"] + n_val["India"]
    s1 = pl.read_parquet(WORK_DIR / "norm/test_s1.parquet", columns=["entity_id", "country"])
    n_test = dict(s1.group_by("country").len().rows())
    v = pl.concat([_test_pairs(sub).select(pl.col("s1").hash(), pl.col("rec").hash(), pl.lit(1 << i, pl.UInt8).alias("m")).collect()
                   for i, (_, _, sub) in enumerate(VOTERS)])
    v = _pattern(v).join(s1.select(pl.col("entity_id").hash().alias("s1"), "country"), on="s1")
    rt = v.group_by("country", "pat").agg(pl.len().alias("n_test"))
    v = v.with_columns(pl.when(pl.col("country") == "France").then(pl.lit("pooled")).otherwise(pl.col("country")).alias("ref"))
    rt = rt.with_columns(pl.when(pl.col("country") == "France").then(pl.lit("pooled")).otherwise(pl.col("country")).alias("ref"))
    c = cells.rename({"country": "ref", "n": "n_val"}).select("ref", "pat", "votes", "q", "n_val")
    rt = rt.join(c, on=["ref", "pat"], how="left").with_columns(
        pl.col("n_val").fill_null(0),
        ((pl.col("n_val").fill_null(0) + 2 * pl.col("n_val").fill_null(0).sqrt() + 3) / pl.col("ref").replace_strict(n_val, return_dtype=pl.Float64)).alias("rate_val"),
        ((pl.col("n_test") + 3) / pl.col("country").replace_strict(n_test, return_dtype=pl.Float64)).alias("rate_test"))
    prior = cells.rename({"country": "ref"}).group_by("ref", "votes").agg(pl.col("prior").first())
    v = v.join(rt.select("country", "pat", "q", "rate_val", "rate_test"), on=["country", "pat"], how="left") \
         .join(prior, on=["ref", "votes"], how="left") \
         .with_columns(pl.min_horizontal(pl.lit(1.0), pl.col("rate_val") / pl.col("rate_test")).alias("c"),
                       pl.coalesce("q", "prior").fill_null(0.0).alias("q_val"))
    v = v.with_columns((pl.col("q_val") * pl.col("c")).alias("q"))
    v.write_parquet(WORK_DIR / "e_c2_test_pairs.parquet")
    base = pl.col("pat").str.slice([n for n, _, _ in VOTERS].index(BASE), 1) == "1"
    drop = base & (pl.col("c") < 1) & (pl.col("q") < T)                   # density-driven only
    keep = v.filter(base & ~drop).select("s1", "rec")
    rep = v.filter(base).group_by("country").agg(pl.len().alias(BASE), drop.sum().alias("dropped"))
    print(f"T = {T:.3f}", rep.sort("country"), flush=True)
    # hashed -> string ids: every kept pair appears in at least one voter file
    strs = pl.concat([_test_pairs(sub).with_columns(pl.col("s1").hash().alias("hs"), pl.col("rec").hash().alias("hr"))
                      .join(keep.lazy().select(pl.col("s1").alias("hs"), pl.col("rec").alias("hr")), on=["hs", "hr"], how="semi")
                      .select("s1", "rec").collect() for _, _, sub in VOTERS]).unique()
    OUT.mkdir(parents=True, exist_ok=True)
    lists = strs.group_by("s1").agg(pl.col("rec").unique().sort().str.join(",").alias("matched_entity_ids"))
    s1.select(pl.col("entity_id").alias("source1_entity_id")) \
      .join(lists.rename({"s1": "source1_entity_id"}), on="source1_entity_id", how="left", maintain_order="left").fill_null("") \
      .write_csv(OUT / "matching_results.tsv", separator="\t", quote_style="never")
    import shutil
    shutil.copy(OUTPUT_DIR / VOTERS[-1][2] / "candidate_pairs.tsv", OUT / "candidate_pairs.tsv")
    print(f"{strs.height:,} pairs -> {OUT}", flush=True)


if __name__ == "__main__":
    {"val": validate, "test": test}[sys.argv[1] if len(sys.argv) > 1 else "val"]()
