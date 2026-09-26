"""E-T1: learned transliteration map -> 4 name features on the romanised non-Latin records.

learn : map from train true pairs with a non-Latin record; 'val' map excludes validation-sample
        S1 (no leakage), 'all' map uses all of train (for test).
eval  : add tr_name_tset / tr_name_ratio / tr_name_jw / tr_n_tok_extra to the current best
        validation features and compare M1 (masked, same folds) with its baseline.
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from experiments.e_d1_decode import f05, ntrue_table
from experiments.e_m1_lgbm import PARAMS, ROUNDS
from src.config import DATA_DIR, SEED, WORK_DIR
from src.decode import ef_decode, exclusive
from src.evalx import log_experiment, truth_pairs
from src.features import FEATURES, _fuzz
from src.features2 import NEW
from src.translit import learn_map, translit_tokens

TR = ["tr_name_tset", "tr_name_ratio", "tr_name_jw", "tr_n_tok_extra"]


def raw_names(split: str, src: int, ids: pl.DataFrame) -> pl.DataFrame:
    return pl.scan_csv(DATA_DIR / split / f"{split}_source{src}.tsv", separator="\t", quote_char=None, infer_schema=False) \
             .select("entity_id", "business_name").join(ids.lazy(), on="entity_id").collect()


def learn() -> None:
    nl = pl.concat([pl.read_parquet(WORK_DIR / f"norm/train_s{s}.parquet", columns=["entity_id", "nonlatin"]) for s in (2, 3)]) \
           .filter(pl.col("nonlatin")).select("entity_id")
    t = truth_pairs().drop_nulls().join(nl.rename({"entity_id": "rec"}), on="rec")
    s1n = raw_names("train", 1, t.select(pl.col("s1").alias("entity_id")).unique()).rename({"entity_id": "s1", "business_name": "s1_name"})
    rn = pl.concat([raw_names("train", s, t.select(pl.col("rec").alias("entity_id")).unique()) for s in (2, 3)]) \
           .rename({"entity_id": "rec", "business_name": "rec_name"})
    pairs = t.join(s1n, on="s1").join(rn, on="rec")
    sample = pl.read_parquet(WORK_DIR / "e_b3_sample_s1.parquet").select(pl.col("entity_id").alias("s1"))
    for tag, p in (("val", pairs.join(sample, on="s1", how="anti")), ("all", pairs)):
        m = learn_map(p)
        (WORK_DIR / f"translit_map_{tag}.json").write_text(json.dumps(m, ensure_ascii=False))
        print(f"{tag}: {p.height:,} non-Latin true pairs -> {len(m):,} token mappings", flush=True)
    print("examples:", list(m.items())[:25], flush=True)


def tr_features(pairs: pl.DataFrame, s1: pl.DataFrame, recs: pl.DataFrame, mapping: dict) -> pl.DataFrame:
    """pairs: (s1, rec, name_tset, name_ratio, name_jw, n_tok_extra); s1: (entity_id, name_tok);
    recs: (entity_id, business_name, nonlatin). Latin records keep their original values."""
    r = recs.with_columns(pl.col("business_name").map_elements(lambda x: translit_tokens(x, mapping), return_dtype=pl.String).alias("tr"))
    d = pairs.join(s1.select(pl.col("entity_id").alias("s1"), pl.col("name_tok").alias("n1")), on="s1", how="left") \
             .join(r.select(pl.col("entity_id").alias("rec"), "tr", "nonlatin"), on="rec", how="left")
    n1, tr = d["n1"].fill_null(""), d["tr"].fill_null("")
    d = d.with_columns(_fuzz(n1, tr, fuzz.token_set_ratio).alias("a"), _fuzz(n1, tr, fuzz.ratio).alias("b"),
                       _fuzz(n1, tr, JaroWinkler.normalized_similarity).alias("c"),
                       pl.col("tr").str.split(" ").list.set_difference(pl.col("n1").str.split(" ")).list.len().alias("e"))
    nl = pl.col("nonlatin").fill_null(False)
    return d.select("s1", "rec",
                    pl.when(nl).then(pl.col("a")).otherwise(pl.col("name_tset")).cast(pl.Float32).alias("tr_name_tset"),
                    pl.when(nl).then(pl.col("b")).otherwise(pl.col("name_ratio")).cast(pl.Float32).alias("tr_name_ratio"),
                    pl.when(nl).then(pl.col("c")).otherwise(pl.col("name_jw")).cast(pl.Float32).alias("tr_name_jw"),
                    pl.when(nl).then(pl.col("e")).otherwise(pl.col("n_tok_extra")).cast(pl.Float32).alias("tr_n_tok_extra"))


def evaluate(src: str) -> None:
    t0 = time.time()
    mapping = json.loads((WORK_DIR / "translit_map_val.json").read_text())
    feats = pl.read_parquet(WORK_DIR / f"{src}_feats.parquet")
    ids = feats.select("s1", "rec")
    s1 = pl.read_parquet(WORK_DIR / "norm/train_s1.parquet", columns=["entity_id", "name_tok"]).join(ids.select(pl.col("s1").alias("entity_id")).unique(), on="entity_id")
    nlf = pl.concat([pl.read_parquet(WORK_DIR / f"norm/train_s{s}.parquet", columns=["entity_id", "nonlatin"]) for s in (2, 3)]) \
            .join(ids.select(pl.col("rec").alias("entity_id")).unique(), on="entity_id")
    raw = pl.concat([raw_names("train", s, nlf.filter(pl.col("nonlatin")).select("entity_id")) for s in (2, 3)])
    recs = nlf.join(raw, on="entity_id", how="left")
    feats = feats.join(tr_features(feats.select("s1", "rec", "name_tset", "name_ratio", "name_jw", "n_tok_extra"), s1, recs, mapping), on=["s1", "rec"])
    feats.write_parquet(WORK_DIR / f"{src}_tr_feats.parquet")
    parts = sorted(set(feats.select("country", "state").unique().rows()))
    fold_of = {p: int(f) for p, f in zip(parts, np.random.default_rng(SEED).permutation(len(parts)) % 2)}
    feats = feats.join(pl.DataFrame({"country": [p[0] for p in parts], "state": [p[1] for p in parts],
                                     "fold": [fold_of[p] for p in parts]}), on=["country", "state"])
    cols = FEATURES + NEW + TR
    oof = []
    for f in (0, 1):
        tr, va = feats.filter(pl.col("fold") != f), feats.filter(pl.col("fold") == f)
        model = lgb.train({**PARAMS, "num_threads": 8}, lgb.Dataset(tr.select(cols).to_numpy(), tr["y"].to_numpy()), ROUNDS)
        oof.append(va.select("s1", "rec", "country", "y").with_columns(pl.Series("p", model.predict(va.select(cols).to_numpy()))))
        del tr, va, model
    scored = pl.concat(oof)
    scored.write_parquet(WORK_DIR / f"{src}_tr_m1_oof.parquet")
    m = f05(ef_decode(exclusive(scored.select(pl.col("s1").hash(), pl.col("rec").hash(), "y", "p"))), ntrue_table(), by_country=True)
    log_experiment(f"E-T1:m1v2_masked+translit[{src}]+ef", m, t0)


if __name__ == "__main__":
    if sys.argv[1] == "learn":
        learn()
    else:
        evaluate(sys.argv[2] if len(sys.argv) > 2 else "e_f2b")
