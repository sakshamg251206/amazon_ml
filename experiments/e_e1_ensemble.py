"""E-E1: five individually tuned models + ensembles, all out-of-fold on the validation sample.

Data: E-M2b rows (pairs with M1 OOF p1 >= 0.01; 44 features = 34 pair + 10 collective), same
partition folds as M1/M2. Every score is macro F0.5 after exclusive + expected-F decoding.
Tuning: small grid per model, config chosen by mean F0.5 over the two folds (per-fold scores are
logged so any selection optimism is visible). Ensemble weights / stacker are CROSS-FITTED: fitted
on one fold's OOF predictions, scored on the other fold.

Usage: python -m experiments.e_e1_ensemble [build|base|ens|all]
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, QuantileTransformer, StandardScaler

from experiments.e_d1_decode import f05, ntrue_table
from resolver.collective import COLLECTIVE, collective
from competition.config import SEED, WORK_DIR
from resolver.decode import ef_decode, exclusive
from competition.evalx import log_experiment
from resolver.features import FEATURES

ENS = WORK_DIR / "ens"
DATA = ENS / "data.parquet"
COLS = FEATURES + COLLECTIVE
SRC_OOF = WORK_DIR / "e_m1_oof_k5.parquet"            # stage-1 OOF probabilities (pruned at 0.01)
SRC_FEATS = WORK_DIR / "e_m1_feats_k5" / "*.parquet"  # pair features for the same candidates
THREADS = 7  # test pipeline finished: use the machine


def build() -> None:
    from resolver.collective import components
    ENS.mkdir(exist_ok=True)
    scored = pl.read_parquet(SRC_OOF).filter(pl.col("p") >= 0.01)
    recs = pl.concat([pl.read_parquet(WORK_DIR / f"norm/train_s{s}.parquet",
                                      columns=["entity_id", "country", "name_key", "name_ns", "hn", "addr"]) for s in (2, 3)])
    comp = components(recs)
    del recs
    col = collective(scored, comp).with_columns(pl.col(c).cast(pl.Float32) for c in COLLECTIVE)
    del comp, scored
    d = pl.scan_parquet(SRC_FEATS).join(col.lazy(), on=["s1", "rec"]).collect()
    from competition.groups import GRP, group_features
    if any(c in GRP for c in COLS):   # E-G2 entity-group gains from this stage-1's p1
        recs = pl.concat([pl.read_parquet(WORK_DIR / f"norm/train_s{s}.parquet", columns=["entity_id", "name_tok", "addr", "hn"]) for s in (2, 3)]) \
                 .join(d.select(pl.col("rec").alias("entity_id")).unique(), on="entity_id")
        d = d.join(group_features(d.select("s1", "rec", "p1", "name_tset", "addr_tset"), recs), on=["s1", "rec"], how="left")
        del recs
    parts = sorted(set(d.select("country", "state").unique().rows()))
    fold_of = {p: int(f) for p, f in zip(parts, np.random.default_rng(SEED).permutation(len(parts)) % 2)}  # M1 folds
    d = d.join(pl.DataFrame({"country": [p[0] for p in parts], "state": [p[1] for p in parts],
                             "fold": [fold_of[p] for p in parts]}), on=["country", "state"])
    d.select(pl.col("s1").hash(), pl.col("rec").hash(), "y", "fold", *COLS).write_parquet(DATA)
    print(f"data: {d.height:,} rows", flush=True)


def _s1_fold(d: pl.DataFrame) -> pl.DataFrame:
    return d.select("s1", "fold").unique("s1")


def score(p: np.ndarray, d: pl.DataFrame, nt: pl.DataFrame) -> dict[str, float]:
    """Overall and per-fold macro F0.5 (S1 without any row are singletons-or-missed in both)."""
    pred = ef_decode(exclusive(d.select("s1", "rec", "y").with_columns(pl.Series("p", p))))
    out = {"f05": f05(pred, nt)["f05"]}
    sf = _s1_fold(d)
    for f in (0, 1):
        ids = sf.filter(pl.col("fold") == f).select("s1")
        out[f"f05_fold{f}"] = f05(pred.join(ids, on="s1"), nt.join(ids, on="s1"))["f05"]
    return out


class _lgb:
    """sklearn-style wrapper around lgb.train (module level so fitted models can be pickled)."""
    def __init__(self, params: dict, rounds: int):
        self.params = dict(objective="binary", bagging_freq=1, seed=SEED, verbose=-1, num_threads=THREADS, **params)
        self.rounds = rounds

    def fit(self, X, y):
        self.b = lgb.train(self.params, lgb.Dataset(X, y), self.rounds)
        return self

    def predict_proba(self, X):
        p = self.b.predict(X)
        return np.c_[1 - p, p]


class _dense:
    """sklearn models other than HistGB need finite inputs; -1 is outside every feature's range."""
    def __init__(self, model):
        self.m = model

    def fit(self, X, y):
        self.m.fit(np.nan_to_num(X, nan=-1.0), y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(np.nan_to_num(X, nan=-1.0))


def _qt():
    return QuantileTransformer(n_quantiles=200, output_distribution="normal", subsample=200_000, random_state=SEED)


GRID = {
    "lgbm": {
        "a": lambda: _lgb(dict(learning_rate=0.08, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8), 400),
        "b": lambda: _lgb(dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8), 700),
        "c": lambda: _lgb(dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, lambda_l2=1.0), 800),
    },
    "histgb": {
        "a": lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_iter=400, max_leaf_nodes=63, l2_regularization=1.0, early_stopping=False, random_state=SEED),
        "b": lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=800, max_leaf_nodes=31, min_samples_leaf=100, early_stopping=False, random_state=SEED),
    },
    "extratrees": {
        "a": lambda: _dense(ExtraTreesClassifier(n_estimators=150, min_samples_leaf=40, max_features=0.5, max_depth=24, n_jobs=THREADS, random_state=SEED)),
        "b": lambda: _dense(ExtraTreesClassifier(n_estimators=150, min_samples_leaf=10, max_features="sqrt", max_depth=24, n_jobs=THREADS, random_state=SEED)),
    },
    "logreg": {
        "a": lambda: _dense(make_pipeline(_qt(), LogisticRegression(C=1.0, max_iter=400))),
        "b": lambda: _dense(make_pipeline(_qt(), LogisticRegression(C=0.05, max_iter=400))),
    },
    "mlp": {
        "a": lambda: _dense(make_pipeline(_qt(), StandardScaler(), MLPClassifier((128, 64), alpha=1e-4, batch_size=2048, max_iter=12, random_state=SEED))),
        "b": lambda: _dense(make_pipeline(_qt(), StandardScaler(), MLPClassifier((64, 32), alpha=1e-3, batch_size=2048, max_iter=12, random_state=SEED))),
    },
}


def oof(make, d: pl.DataFrame, X: np.ndarray) -> np.ndarray:
    p = np.zeros(d.height)
    fold, y = d["fold"].to_numpy(), d["y"].to_numpy()
    for f in (0, 1):
        tr, va = fold != f, fold == f
        p[va] = make().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    return p


def base(models: list[str]) -> None:
    d = pl.read_parquet(DATA)
    X = d.select(COLS).to_numpy().astype(np.float32)
    nt = ntrue_table()
    for name in models:
        out = ENS / f"oof_{name}.parquet"
        if out.exists():
            continue
        best = None
        for cfg, make in GRID[name].items():
            t0 = time.time()
            p = oof(make, d, X)
            m = score(p, d, nt)
            log_experiment(f"E-E1:{name}:{cfg}", m, t0)
            if best is None or m["f05"] > best[1]["f05"]:
                best = (cfg, m, p)
        pl.DataFrame({"p": best[2]}).write_parquet(out)
        print(f"==> {name}: best config {best[0]} f05={best[1]['f05']:.4f}", flush=True)


def _fit_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to 1 minimising log-loss of the weighted probability average."""
    def loss(z):
        w = np.exp(z) / np.exp(z).sum()
        q = np.clip(P @ w, 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    z = minimize(loss, np.zeros(P.shape[1]), method="Nelder-Mead", options={"maxiter": 2000}).x
    return np.exp(z) / np.exp(z).sum()


def ens(models: list[str]) -> None:
    t0 = time.time()
    d = pl.read_parquet(DATA, columns=["s1", "rec", "y", "fold"])
    nt = ntrue_table()
    P = np.column_stack([pl.read_parquet(ENS / f"oof_{m}.parquet")["p"].to_numpy() for m in models])
    y, fold = d["y"].to_numpy(), d["fold"].to_numpy()
    log_experiment("E-E1:ens:mean", score(P.mean(1), d, nt), t0)
    # cross-fitted weights and stacker: fit on fold f, apply to the other fold
    pw, ps, weights = np.zeros(len(y)), np.zeros(len(y)), {}
    L = logit(np.clip(P, 1e-6, 1 - 1e-6))
    for f in (0, 1):
        fit, app = fold == f, fold != f
        w = _fit_weights(P[fit], y[fit])
        weights[f] = dict(zip(models, np.round(w, 3)))
        pw[app] = P[app] @ w
        ps[app] = LogisticRegression(C=1.0, max_iter=500).fit(L[fit], y[fit]).predict_proba(L[app])[:, 1]
    print("weights fitted per fold:", weights, flush=True)
    log_experiment("E-E1:ens:weighted_xfit", score(pw, d, nt), t0)
    log_experiment("E-E1:ens:stack_lr_xfit", score(ps, d, nt), t0)


# Round 2: one stronger config per model (logreg gets binned one-hot inputs = a GAM-like model).
GRID["lgbm"]["d"] = lambda: _lgb(dict(learning_rate=0.03, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8), 1200)
GRID["histgb"]["c"] = lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=1000, max_leaf_nodes=127, min_samples_leaf=50, l2_regularization=1.0, early_stopping=False, random_state=SEED)
GRID["extratrees"]["c"] = lambda: _dense(ExtraTreesClassifier(n_estimators=300, min_samples_leaf=20, max_features=0.5, max_depth=28, n_jobs=THREADS, random_state=SEED))
GRID["logreg"]["c"] = lambda: _dense(make_pipeline(KBinsDiscretizer(n_bins=16, encode="onehot", strategy="quantile", subsample=200_000, random_state=SEED), LogisticRegression(C=0.5, max_iter=500)))
GRID["mlp"]["c"] = lambda: _dense(make_pipeline(_qt(), StandardScaler(), MLPClassifier((256, 128), alpha=1e-4, batch_size=4096, max_iter=25, random_state=SEED)))
ROUND2 = {"lgbm": "d", "histgb": "c", "extratrees": "c", "logreg": "c", "mlp": "c"}
BEST = ENS / "best_cfg.json"  # best config per model so far (round 1 winners by default)


def _best_cfg() -> dict:
    import json
    return json.loads(BEST.read_text()) if BEST.exists() else {"lgbm": "b", "histgb": "a", "extratrees": "a", "logreg": "a", "mlp": "b"}


def base2() -> None:
    """Try each ROUND2 config; replace a model's cached OOF only if F0.5 improves."""
    import json
    d = pl.read_parquet(DATA)
    X = d.select(COLS).to_numpy().astype(np.float32)
    nt = ntrue_table()
    best = _best_cfg()
    for name, cfg in ROUND2.items():
        if best.get(name) == cfg:
            continue
        old = score(pl.read_parquet(ENS / f"oof_{name}.parquet")["p"].to_numpy(), d, nt)["f05"]
        t0 = time.time()
        p = oof(GRID[name][cfg], d, X)
        m = score(p, d, nt)
        log_experiment(f"E-E1:{name}:{cfg}", m, t0)
        if m["f05"] > old:
            pl.DataFrame({"p": p}).write_parquet(ENS / f"oof_{name}.parquet")
            best[name] = cfg
        print(f"==> {name}: {cfg} {m['f05']:.4f} vs previous {old:.4f} -> keep {best[name]}", flush=True)
        BEST.write_text(json.dumps(best))


FINAL = {"lgbm": "b", "extratrees": "a", "mlp": "b"}  # chosen from E-E1 (histgb/logreg got ~0 weight)


def final() -> None:
    """Fit the FINAL configs on all rows; plain average (E-E1 ens3 mean 0.9510 beat fitted weights 0.9508)."""
    import joblib
    d = pl.read_parquet(DATA)
    X, y = d.select(COLS).to_numpy().astype(np.float32), d["y"].to_numpy()
    w = np.full(len(FINAL), 1 / len(FINAL))
    for (name, cfg), wi in zip(FINAL.items(), w):
        t0 = time.time()
        joblib.dump(GRID[name][cfg]().fit(X, y), ENS / f"final_{name}.joblib", compress=3)
        print(f"final {name}:{cfg} weight={wi:.3f} ({time.time() - t0:.0f}s)", flush=True)
    joblib.dump(dict(zip(FINAL, w)), ENS / "final_weights.joblib")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    models = list(GRID)
    if stage == "ens3":
        ens(list(FINAL))
    if stage == "round2":
        base2()
        ens(list(GRID))
    if stage == "final":
        final()
    if stage in ("build", "all") and not DATA.exists():
        build()
    if stage in ("base", "all"):
        base(models)
    if stage in ("ens", "all"):
        ens(models)
