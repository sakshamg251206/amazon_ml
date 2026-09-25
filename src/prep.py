"""Stage `prep`: raw TSV -> normalised parquet (one file per split x source), cached in WORK_DIR.

Per-row Python normalisation is the slow part, so rows are processed in parallel chunks.
"""
from multiprocessing import Pool
from pathlib import Path

import polars as pl

from src.config import WORK_DIR
from src.data import read_source
from src.normalize import addr_parts, clean, is_nonlatin, name_tokens, nums

CHUNK = 125_000
SLICE = 1_000_000


def _norm_chunk(rows: tuple[list[str], list[str]]) -> dict[str, list]:
    names, addrs = rows
    out: dict[str, list] = {"name_tok": [], "addr": [], "parts": [], "nums": [], "nonlatin": []}
    for n, a in zip(names, addrs):
        out["name_tok"].append(" ".join(name_tokens(n)))
        out["addr"].append(clean(a))
        out["parts"].append(addr_parts(a))
        out["nums"].append(nums(a))
        out["nonlatin"].append(is_nonlatin(n))
    return out


def _norm_slice(df: pl.DataFrame, pool: Pool) -> pl.DataFrame:
    names, addrs = df["business_name"].to_list(), df["business_address"].to_list()
    chunks = [(names[i:i + CHUNK], addrs[i:i + CHUNK]) for i in range(0, len(names), CHUNK)]
    parts = pool.map(_norm_chunk, chunks)
    cols = {k: [v for p in parts for v in p[k]] for k in parts[0]}
    return df.select("entity_id", "country", (pl.col("business_address") == "").alias("empty_addr")).with_columns(
        pl.Series("name_tok", cols["name_tok"], dtype=pl.String),
        pl.Series("addr", cols["addr"], dtype=pl.String),
        pl.Series("parts", cols["parts"], dtype=pl.List(pl.String)),
        pl.Series("nums", cols["nums"], dtype=pl.List(pl.String)),
        pl.Series("nonlatin", cols["nonlatin"], dtype=pl.Boolean),
    ).with_columns(
        pl.col("name_tok").str.split(" ").list.sort().list.unique(maintain_order=True)
        .list.join(" ").alias("name_key"),
        pl.col("name_tok").str.replace_all(" ", "").alias("name_ns"),
        pl.col("nums").list.first().fill_null("").alias("hn"),
    )


def norm_path(split: str, source: int) -> Path:
    return WORK_DIR / "norm" / f"{split}_s{source}.parquet"


def prep(split: str, source: int, pool: Pool) -> Path:
    path = norm_path(split, source)
    if path.exists():
        return path
    df = read_source(split, source)
    # Process 1M-row slices so only one slice lives as Python lists at a time (8 GB machine).
    pieces = [_norm_slice(df.slice(i, SLICE), pool) for i in range(0, df.height, SLICE)]
    out = pl.concat(pieces)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path)
    return path


if __name__ == "__main__":
    import time
    with Pool(8) as pool:
        for split in ("train", "test"):
            for source in (1, 2, 3):
                t0 = time.time()
                p = prep(split, source, pool)
                print(f"{p.name}: {time.time() - t0:.0f}s", flush=True)
