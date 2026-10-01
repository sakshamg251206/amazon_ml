"""E-E2: the E-E1 stage-2 recipe on the E-F2 feature set (v2).

Stage-1 = M1 with 34 + 14 features (E-F2 OOF, 0.9590 with expected-F). Stage 2 = collective
features + the three E-E1 winners (LightGBM b, ExtraTrees a, MLP b), plain average, expected-F.
Same partition folds. Kept only if it beats 0.9590 in both folds. Final models -> work/ens2/.
"""
import time

import numpy as np
import polars as pl

import experiments.e_e1_ensemble as E
from experiments.e_d1_decode import ntrue_table
from experiments.e_m2_collective import COLLECTIVE
from competition.config import WORK_DIR
from competition.evalx import log_experiment
from resolver.features import FEATURES
from resolver.adaptive import GEN
from resolver.features2 import NEW

TR = ["tr_name_tset", "tr_name_ratio", "tr_name_jw", "tr_n_tok_extra"]


def configure(src: str = "e_f2b", ens: str = "ens2") -> None:
    """src = validation feature/OOF prefix: e_f2b (v2), or e.g. e_b5_tr (v3: E-B5 + E-T1)."""
    E.ENS = WORK_DIR / ens
    E.DATA = E.ENS / "data.parquet"
    E.COLS = FEATURES + NEW + (TR if "_tr" in src else []) + (GEN if "_gen" in src else []) + COLLECTIVE
    E.SRC_OOF = WORK_DIR / f"{src}_m1_oof.parquet"
    E.SRC_FEATS = WORK_DIR / f"{src}_feats.parquet"
    E.FINAL = {"lgbm": "b", "extratrees": "a", "mlp": "b"}


configure()


def main() -> None:
    E.ENS.mkdir(exist_ok=True)
    if not E.DATA.exists():
        E.build()
    d = pl.read_parquet(E.DATA)
    X = d.select(E.COLS).to_numpy().astype(np.float32)
    nt = ntrue_table()
    t0 = time.time()
    tag = E.ENS.name
    log_experiment(f"E-E2[{tag}]:baseline_m1", E.score(d["p1"].to_numpy(), d, nt), t0)  # same rows/folds
    P = []
    for name, cfg in E.FINAL.items():
        t0 = time.time()
        p = E.oof(E.GRID[name][cfg], d, X)
        pl.DataFrame({"p": p}).write_parquet(E.ENS / f"oof_{name}.parquet")
        log_experiment(f"E-E2[{tag}]:{name}:{cfg}", E.score(p, d, nt), t0)
        P.append(p)
    t0 = time.time()
    log_experiment(f"E-E2[{tag}]:mean3", E.score(np.mean(P, axis=0), d, nt), t0)
    del X
    E.final()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        configure(sys.argv[1], sys.argv[2])
    main()
