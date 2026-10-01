"""E-T3: vocabulary class of unmatched name tokens (src/vocab.py) on top of a base feature set.

Usage: python -m experiments.e_t3_vocab feats <base>   -> work/<base>_voc_only.parquet (light, chunked)
       python -m experiments.e_t3_vocab m1 <base>      -> M1 OOF vs the base M1 OOF, per fold (keep only if both up)
DF / generic vocabularies are fitted label-free on all train S1 (per country), as on test.
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from experiments.e_a1_address import fold_scores
from experiments.e_d1_decode import ntrue_table
from experiments.e_m1_lgbm import PARAMS, ROUNDS
from resolver.adaptive import GEN, generic_tokens
from competition.address import CA
from competition.config import SEED, WORK_DIR
from competition.evalx import log_experiment
from resolver.features import FEATURES
from resolver.features2 import NEW
from competition.vocab import VOC, token_df, vocab_features

N_CHUNKS = 8


def feats(base: str) -> None:
    t0 = time.time()
    ids = pl.read_parquet(WORK_DIR / f"{base}_feats.parquet", columns=["s1", "rec"])
    s1 = pl.read_parquet(WORK_DIR / "norm/train_s1.parquet", columns=["entity_id", "country", "name_tok"])
    df, gen = token_df(s1), generic_tokens(s1)
    recs = pl.concat([pl.read_parquet(WORK_DIR / f"norm/train_s{s}.parquet", columns=["entity_id", "name_tok"]) for s in (2, 3)]) \
             .join(ids.select(pl.col("rec").alias("entity_id")).unique(), on="entity_id")
    out = []
    for c in range(N_CHUNKS):
        part = ids.filter(pl.col("rec").hash(SEED) % N_CHUNKS == c)
        out.append(vocab_features(part, s1, recs, df, gen))
        print(f"  chunk {c + 1}/{N_CHUNKS} ({time.time() - t0:.0f}s)", flush=True)
    pl.concat(out).write_parquet(WORK_DIR / f"{base}_voc_only.parquet")


def m1(base: str) -> None:
    t0 = time.time()
    f = pl.read_parquet(WORK_DIR / f"{base}_feats.parquet").join(pl.read_parquet(WORK_DIR / f"{base}_voc_only.parquet"), on=["s1", "rec"])
    out = f"{base}_voc"
    f.write_parquet(WORK_DIR / f"{out}_feats.parquet")
    parts = sorted(set(f.select("country", "state").unique().rows()))
    fold_of = {p: int(x) for p, x in zip(parts, np.random.default_rng(SEED).permutation(len(parts)) % 2)}
    f = f.join(pl.DataFrame({"country": [p[0] for p in parts], "state": [p[1] for p in parts],
                             "fold": [fold_of[p] for p in parts]}), on=["country", "state"])
    use = FEATURES + NEW + (GEN if "_gen" in base else []) + (CA if "_ca" in base else []) + VOC
    oof = []
    for k in (0, 1):
        tr, va = f.filter(pl.col("fold") != k), f.filter(pl.col("fold") == k)
        model = lgb.train({**PARAMS, "num_threads": 8}, lgb.Dataset(tr.select(use).to_numpy(), tr["y"].to_numpy()), ROUNDS)
        oof.append(va.select("s1", "rec", "country", "y").with_columns(pl.Series("p", model.predict(va.select(use).to_numpy()))))
        if k == 0:
            g = model.feature_importance("gain")
            print("VOC gain shares:", {n: round(float(v / g.sum()), 3) for n, v in zip(use, g) if n in VOC}, flush=True)
        del tr, va, model
    scored = pl.concat(oof)
    scored.write_parquet(WORK_DIR / f"{out}_m1_oof.parquet")
    nt = ntrue_table()
    log_experiment(f"E-T3:m1[{out}]+ef", fold_scores(scored, nt), t0)
    log_experiment(f"E-T3:baseline[{base}]", fold_scores(pl.read_parquet(WORK_DIR / f"{base}_m1_oof.parquet"), nt), t0)


if __name__ == "__main__":
    {"feats": feats, "m1": m1}[sys.argv[1]](sys.argv[2] if len(sys.argv) > 2 else "e_f2b_gen_ca")
