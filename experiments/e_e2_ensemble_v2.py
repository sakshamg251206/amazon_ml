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
from src.config import WORK_DIR
from src.evalx import log_experiment
from src.features import FEATURES
from src.features2 import NEW

E.ENS = WORK_DIR / "ens2"
E.DATA = E.ENS / "data.parquet"
E.COLS = FEATURES + NEW + COLLECTIVE
E.SRC_OOF = WORK_DIR / "e_f2b_m1_oof.parquet"  # masked (E-F2b), as shipped
E.SRC_FEATS = WORK_DIR / "e_f2b_feats.parquet"
E.FINAL = {"lgbm": "b", "extratrees": "a", "mlp": "b"}


def main() -> None:
    E.ENS.mkdir(exist_ok=True)
    if not E.DATA.exists():
        E.build()
    d = pl.read_parquet(E.DATA)
    X = d.select(E.COLS).to_numpy().astype(np.float32)
    nt = ntrue_table()
    t0 = time.time()
    log_experiment("E-E2:baseline_m1_masked", E.score(d["p1"].to_numpy(), d, nt), t0)  # same rows/folds
    P = []
    for name, cfg in E.FINAL.items():
        t0 = time.time()
        p = E.oof(E.GRID[name][cfg], d, X)
        pl.DataFrame({"p": p}).write_parquet(E.ENS / f"oof_{name}.parquet")
        log_experiment(f"E-E2:{name}:{cfg}", E.score(p, d, nt), t0)
        P.append(p)
    t0 = time.time()
    log_experiment("E-E2:mean3", E.score(np.mean(P, axis=0), d, nt), t0)
    del X
    E.final()


if __name__ == "__main__":
    main()
