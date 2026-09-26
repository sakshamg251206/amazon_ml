"""Quick Submission 2a: exact-key candidates (+ any TF-IDF partitions already blocked) scored by the
M1 model. Minutes instead of hours; candidate ceiling ~0.89 vs 0.99 for the full pipeline.
Writes to output/sub2a_keys/ so earlier submissions are never overwritten."""
import shutil

import polars as pl

import src.pipeline as P
from src.block import key_candidates
from src.config import OUTPUT_DIR, WORK_DIR

_orig_dir = P._dir
P._dir = lambda split, name: _orig_dir(f"quick_{split}", name)  # -> work/pipe_quick_test/...
P.OUTPUT_DIR = OUTPUT_DIR / "sub2a_keys"


def main() -> None:
    cols = ["entity_id", "country", "name_tok", "addr", "hn", "name_key", "name_ns"]
    countries = pl.read_parquet(WORK_DIR / "norm/test_s1.parquet", columns=["country"])["country"].unique()
    for c in countries:
        path = P._dir("test", "keys") / f"keys_{c}.parquet"
        if path.exists():
            continue
        s1 = P._norm_country("test", 1, cols, c)
        recs = pl.concat([P._norm_country("test", s, cols, c) for s in (2, 3)])
        key_candidates(s1, recs).write_parquet(path)
        print(f"keys {c} done", flush=True)
    tf = P._dir("test", "tf")
    for f in (WORK_DIR / "pipe_test/tf").glob("*.parquet"):  # reuse partitions the big run finished
        shutil.copy(f, tf / f.name)
    P.features("test")
    P.decide("test")


if __name__ == "__main__":
    main()
