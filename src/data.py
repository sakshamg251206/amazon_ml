"""Readers for the challenge TSVs. Every column is a string; empty fields are ""."""
import polars as pl

from src.config import DATA_DIR


def _read_tsv(path) -> pl.DataFrame:
    # quote_char=None: addresses contain quotes/commas; the official format has no quoting.
    return pl.read_csv(
        path, separator="\t", quote_char=None, infer_schema=False,
        empty_string_is_null=False,
    )


def read_source(split: str, source: int) -> pl.DataFrame:
    return _read_tsv(DATA_DIR / split / f"{split}_source{source}.tsv")


def read_ground_truth() -> dict[str, frozenset[str]]:
    df = _read_tsv(DATA_DIR / "train" / "train_ground_truth.tsv")
    return {s1: frozenset(m.split(",")) if m else frozenset() for s1, m in df.iter_rows()}
