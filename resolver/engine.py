"""Resolution engine: raw records -> candidates -> features -> two-stage model -> matches.

This is the validated competition design (macro F0.5 0.9614 out-of-fold on the real data, see
DECISIONS.md D4-D18 and E-T2) as one in-memory flow:

  1. normalise   names/addresses (legal forms, scripts, house numbers)            normalize.py
  2. partition   label-free (country, state) inference                            partition.py
  3. block       TF-IDF top-5 per record inside its partition U country-wide keys  block.py
  4. featurise   34 pair + 14 competition/difference + 6 adaptive-vocabulary      features*.py, adaptive.py
  5. stage 1     LightGBM -> p1
  6. stage 2     10 collective/twin features on p1 >= 0.01, mean of LightGBM +
                 ExtraTrees + MLP -> p                                             collective.py
  7. decode      each record -> its best S1, then expected-F0.5 per S1             decode.py
"""
from __future__ import annotations

import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl

from resolver.adaptive import GEN, generic_features, generic_tokens
from resolver.block import PartitionIndex, key_candidates
from resolver.collective import COLLECTIVE, TWIN_COLS, collective, components
from resolver.decode import ef_decode, exclusive
from resolver.features import ATTRS, FEATURES, pair_features
from resolver.features2 import NEW, competition_features, diff_features, s1_maxima
from resolver.normalize import normalize_frame
from resolver.partition import infer_state, learn_aliases, s1_state

log = logging.getLogger("resolver")

K_NAME, K_COMB, K_ADDR, K_KEEP = 10, 20, 10, 5   # D4: per-view retrieval depth, keep top-5 by combined cosine
PRUNE = 0.01                                     # D14: stage 2 sees pairs with p1 >= 0.01
STAGE1 = FEATURES + NEW + GEN                    # 54 features
STAGE2 = STAGE1 + COLLECTIVE                     # 64 features
SEED = 42
THREADS = max(1, (os.cpu_count() or 2))
MAX_BUCKET_PAIRS = 3_000_000                     # featurise in record buckets above this many pairs


# ====================================================================== blocking
@dataclass
class Blocked:
    pairs: pl.DataFrame                    # s1, rec, cos_name, cos_addr, cos_comb, in_keys (True = key-only)
    s1_part: pl.DataFrame                  # entity_id, country, state
    rec_part: pl.DataFrame                 # entity_id, state
    aliases: dict[str, pl.DataFrame] = field(default_factory=dict)
    partitions: dict[tuple[str, str], tuple[PartitionIndex, np.ndarray]] = field(default_factory=dict)


def _top_keep(c: pl.DataFrame) -> pl.DataFrame:
    return c.with_columns(pl.col("cos_comb").rank("ordinal", descending=True).over("r").alias("_rk")) \
            .filter(pl.col("_rk") <= K_KEEP).drop("_rk")


def block(s1n: pl.DataFrame, recn: pl.DataFrame, keep_index: bool = False) -> Blocked:
    """One country at a time: aliases, keys and partitions never cross countries (France gets its
    own label-free aliases at test time). keep_index retains each partition's TF-IDF index for
    single-record lookups."""
    tf, keys, s1p, rp, aliases, parts = [], [], [], [], {}, {}
    for country in s1n["country"].unique().sort().to_list():
        s1 = s1n.filter(pl.col("country") == country)
        recs = recn.filter(pl.col("country") == country)
        keys.append(key_candidates(s1, recs))
        alias = learn_aliases(s1, recs)
        aliases[country] = alias
        s1 = s1.with_columns(s1_state(s1).alias("state"))
        recs = recs.with_columns(infer_state(recs, alias).alias("state"))
        s1p.append(s1.select("entity_id", "country", "state"))
        rp.append(recs.select("entity_id", "state"))
        for (state,), g in s1.filter(pl.col("state").is_not_null()).group_by("state"):
            idx, ids = PartitionIndex(g), g["entity_id"].to_numpy()
            if keep_index:
                parts[(country, state)] = (idx, ids)
            q = recs.filter(pl.col("state") == state)
            if q.height == 0:
                continue
            c = _top_keep(idx.query(q, K_NAME, K_COMB, K_ADDR))
            tf.append(pl.DataFrame({"s1": ids[c["s"].to_numpy()], "rec": q["entity_id"].to_numpy()[c["r"].to_numpy()]},
                                   schema={"s1": pl.String, "rec": pl.String})
                        .with_columns(c.select("cos_name", "cos_addr", "cos_comb")))
    schema = {"s1": pl.String, "rec": pl.String, "cos_name": pl.Float32, "cos_addr": pl.Float32, "cos_comb": pl.Float32}
    tfp = pl.concat(tf) if tf else pl.DataFrame(schema=schema)
    kp = pl.concat(keys).unique() if keys else pl.DataFrame(schema={"s1": pl.String, "rec": pl.String})
    key_only = kp.join(tfp.select("s1", "rec"), on=["s1", "rec"], how="anti") \
                 .with_columns(pl.lit(None, pl.Float32).alias(c) for c in ("cos_name", "cos_addr", "cos_comb"))
    pairs = pl.concat([tfp.with_columns(pl.lit(False).alias("in_keys")),
                       key_only.select(*schema).with_columns(pl.lit(True).alias("in_keys"))], how="vertical_relaxed")
    return Blocked(pairs=pairs.sort("s1", "rec"), s1_part=pl.concat(s1p) if s1p else pl.DataFrame(),
                   rec_part=pl.concat(rp) if rp else pl.DataFrame(), aliases=aliases, partitions=parts)


# ====================================================================== features
def featurize(pairs: pl.DataFrame, s1n: pl.DataFrame, recn: pl.DataFrame, gen: pl.DataFrame,
              s1_best: pl.DataFrame | None = None) -> pl.DataFrame:
    """pairs (from block) -> (s1, rec, *STAGE1) as Float32. Record-side features need all of a
    record's candidates in the same call; S1-side maxima come from s1_best (default: these pairs)."""
    if pairs.height == 0:
        return pl.DataFrame(schema={"s1": pl.String, "rec": pl.String, **{c: pl.Float32 for c in STAGE1}})
    best = s1_maxima(pairs) if s1_best is None else s1_best
    n_b = max(1, -(-pairs.height // MAX_BUCKET_PAIRS))
    s1a = s1n.select(list(dict.fromkeys(ATTRS + ["country"])))
    out = []
    for b in range(n_b):
        pb = pairs if n_b == 1 else pairs.filter((pl.col("rec").hash(SEED) % n_b) == b)
        f = pair_features(pb, s1a, recn.select(ATTRS)).select("s1", "rec", *[pl.col(x).cast(pl.Float32) for x in FEATURES])
        comp = competition_features(pb.select("s1", "rec", "cos_name", "cos_addr", "cos_comb"), best)
        diff = diff_features(pb.select("s1", "rec"), s1n, recn)
        genf = generic_features(pb.select("s1", "rec"), s1n.select("entity_id", "country", "name_tok"),
                                recn.select("entity_id", "name_tok"), gen)
        out.append(f.join(comp, on=["s1", "rec"], how="left").join(diff, on=["s1", "rec"], how="left")
                    .join(genf, on=["s1", "rec"], how="left"))
    return pl.concat(out).select("s1", "rec", *[pl.col(c).cast(pl.Float32) for c in STAGE1]).sort("s1", "rec")


def to_matrix(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return df.select(cols).to_numpy().astype(np.float32)


# ====================================================================== models
S1_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, seed=SEED, verbose=-1, num_threads=THREADS)
S1_ROUNDS = 400


def train_stage1(X: np.ndarray, y: np.ndarray, rounds: int = S1_ROUNDS) -> lgb.Booster:
    return lgb.train(S1_PARAMS, lgb.Dataset(X, y, free_raw_data=True), rounds)


class LGBModel:
    """sklearn-style wrapper so the stage-2 members share one interface (E-E1 config 'b')."""

    def __init__(self, rounds: int = 700, **params):
        self.params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                           feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, seed=SEED, verbose=-1,
                           num_threads=THREADS, **params)
        self.rounds = rounds

    def fit(self, X, y):
        self.booster = lgb.train(self.params, lgb.Dataset(X, y), self.rounds)
        return self

    def predict_proba(self, X):
        p = self.booster.predict(X)
        return np.c_[1 - p, p]


class DenseModel:
    """Models that need finite inputs: missing values become -1 (outside every feature's range)."""

    def __init__(self, model):
        self.m = model

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")    # MLP max_iter is deliberately short (E-E1)
            self.m.fit(np.nan_to_num(X, nan=-1.0), y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(np.nan_to_num(X, nan=-1.0))


def stage2_members(n_rows: int) -> dict:
    """The three E-E1 winners (LightGBM b, ExtraTrees a, MLP b); plain mean (D15)."""
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer, StandardScaler
    small = n_rows < 200_000      # fewer rounds on small data keeps the boosted member from overfitting
    return {
        "lightgbm": LGBModel(rounds=300 if small else 700),
        "extratrees": DenseModel(ExtraTreesClassifier(n_estimators=150, min_samples_leaf=40, max_features=0.5,
                                                      max_depth=24, n_jobs=THREADS, random_state=SEED)),
        "mlp": DenseModel(make_pipeline(
            QuantileTransformer(n_quantiles=200, output_distribution="normal", subsample=200_000, random_state=SEED),
            StandardScaler(), MLPClassifier((64, 32), alpha=1e-3, batch_size=2048,
                                            max_iter=30 if small else 12, random_state=SEED))),
    }


@dataclass
class Matcher:
    """The fitted two-stage scorer."""
    stage1: lgb.Booster
    stage2: dict
    meta: dict = field(default_factory=dict)

    def p1(self, feats: pl.DataFrame) -> np.ndarray:
        return self.stage1.predict(to_matrix(feats, STAGE1), num_threads=THREADS) if feats.height else np.zeros(0)

    def p2(self, X2: pl.DataFrame) -> np.ndarray:
        if X2.height == 0:
            return np.zeros(0)
        X = to_matrix(X2, STAGE2)
        return np.mean([m.predict_proba(X)[:, 1] for m in self.stage2.values()], axis=0)

    def contributions(self, feats: pl.DataFrame) -> np.ndarray:
        """Per-feature stage-1 SHAP values (log-odds), last column = bias."""
        return self.stage1.predict(to_matrix(feats, STAGE1), pred_contrib=True)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"stage1": self.stage1.model_to_string(), "stage2": self.stage2, "meta": self.meta},
                    path, compress=3)

    @classmethod
    def load(cls, path: str | Path) -> "Matcher":
        d = joblib.load(path)
        return cls(stage1=lgb.Booster(model_str=d["stage1"]), stage2=d["stage2"], meta=d["meta"])


# ====================================================================== scoring
def score(feats: pl.DataFrame, matcher: Matcher, comp: pl.DataFrame) -> pl.DataFrame:
    """feats -> (s1, rec, p1, p, + collective) for pairs with p1 >= PRUNE; p is the stage-2 mean."""
    p1 = matcher.p1(feats)
    kept = feats.with_columns(pl.Series("p1", p1, dtype=pl.Float32)).filter(pl.col("p1") >= PRUNE)
    col = collective(kept.select("s1", "rec", pl.col("p1").alias("p")), comp)
    x2 = kept.drop("p1").join(col, on=["s1", "rec"])
    return x2.with_columns(pl.Series("p", matcher.p2(x2), dtype=pl.Float32))


def decode(scored: pl.DataFrame) -> pl.DataFrame:
    """Exclusive assignment + expected-F0.5 decoding -> (s1, rec, p) matches."""
    if scored.height == 0:
        return pl.DataFrame(schema={"s1": pl.String, "rec": pl.String, "p": pl.Float32})
    return ef_decode(exclusive(scored.select("s1", "rec", "p")))


# ====================================================================== end to end
@dataclass
class Prepared:
    """Everything derived from a dataset before any model is applied."""
    s1n: pl.DataFrame
    recn: pl.DataFrame
    blocked: Blocked
    gen: pl.DataFrame
    feats: pl.DataFrame
    comp: pl.DataFrame
    timings: dict


def prepare(ds, keep_index: bool = False, workers: int = 1, progress=None) -> Prepared:
    progress = progress or (lambda stage: None)
    t, t0 = {}, time.time()
    progress("normalising text")
    s1n = normalize_frame(ds.s1, workers)
    recn = normalize_frame(ds.recs, workers).with_columns(ds.recs["source"])
    t["normalise_s"] = time.time() - t0
    t0 = time.time()
    progress("partitioning and blocking")
    blocked = block(s1n, recn, keep_index=keep_index)
    t["block_s"] = time.time() - t0
    t0 = time.time()
    progress("computing pair features")
    gen = generic_tokens(s1n.select("country", "name_tok"))   # adaptive vocabulary: this dataset's own S1
    feats = featurize(blocked.pairs, s1n, recn, gen)
    comp = components(recn.select(TWIN_COLS))
    t["featurise_s"] = time.time() - t0
    log.info("prepared %s: %d S1, %d records, %d candidate pairs (%.1fs)", ds.name, s1n.height, recn.height,
             blocked.pairs.height, sum(t.values()))
    return Prepared(s1n, recn, blocked, gen, feats, comp, t)


@dataclass
class Resolution:
    prepared: Prepared
    scored: pl.DataFrame      # stage-2 rows (s1, rec, p1, p, ...)
    matches: pl.DataFrame     # (s1, rec, p)
    timings: dict

    @property
    def candidates(self) -> pl.DataFrame:
        return self.prepared.blocked.pairs.select("s1", "rec")


def resolve(ds, matcher: Matcher, keep_index: bool = False, workers: int = 1, progress=None) -> Resolution:
    prep = prepare(ds, keep_index=keep_index, workers=workers, progress=progress)
    t0 = time.time()
    if progress:
        progress("scoring and decoding")
    scored = score(prep.feats, matcher, prep.comp)
    matches = decode(scored)
    timings = {**prep.timings, "score_decode_s": time.time() - t0}
    log.info("resolved %s: %d matched pairs for %d S1 (%.1fs)", ds.name, matches.height, matches["s1"].n_unique(),
             sum(timings.values()))
    return Resolution(prep, scored, matches, timings)
