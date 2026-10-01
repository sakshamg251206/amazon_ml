"""Datasets in the challenge layout, and submission files.

A dataset directory holds `<prefix>_source1.tsv`, `<prefix>_source2.tsv`, `<prefix>_source3.tsv`
and optionally `<prefix>_ground_truth.tsv` (any prefix: train, test, ...). Every column is read as a
string; the official format has no quoting, so quote_char is disabled (addresses contain quotes).
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import polars as pl

RAW_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(src) -> pl.DataFrame:
    """Path or bytes -> all-string frame."""
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    return pl.read_csv(src, separator="\t", quote_char=None, infer_schema=False, empty_string_is_null=False,
                       truncate_ragged_lines=True)


def _check_source(df: pl.DataFrame, n: int) -> pl.DataFrame:
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"source {n}: missing columns {missing} (expected {RAW_COLS}, tab-separated)")
    df = df.select(RAW_COLS).with_columns(pl.col(c).fill_null("") for c in RAW_COLS)
    if df["entity_id"].is_duplicated().any():
        raise ValueError(f"source {n}: duplicate entity_id values")
    bad = df.filter(~pl.col("entity_id").str.starts_with(f"S{n}-"))
    if bad.height:
        raise ValueError(f"source {n}: entity_id must start with 'S{n}-' (e.g. {bad['entity_id'][0]!r})")
    return df


def truth_frame(gt: pl.DataFrame) -> pl.DataFrame:
    """ground-truth TSV -> (s1, rec) true pairs."""
    return (gt.select(pl.col("source1_entity_id").alias("s1"),
                      pl.col("matched_entity_ids").fill_null("").str.split(",").alias("rec"))
              .explode("rec").filter(pl.col("rec").str.strip_chars() != "")
              .with_columns(pl.col("rec").str.strip_chars()).unique())


@dataclass
class Dataset:
    s1: pl.DataFrame                    # RAW_COLS
    recs: pl.DataFrame                  # RAW_COLS + source (2 | 3)
    truth: pl.DataFrame | None = None   # (s1, rec) true pairs
    name: str = "dataset"

    @classmethod
    def from_frames(cls, s1: pl.DataFrame, s2: pl.DataFrame, s3: pl.DataFrame,
                    truth: pl.DataFrame | None = None, name: str = "dataset") -> Dataset:
        s1, s2, s3 = _check_source(s1, 1), _check_source(s2, 2), _check_source(s3, 3)
        recs = pl.concat([s2.with_columns(pl.lit(2, pl.Int8).alias("source")),
                          s3.with_columns(pl.lit(3, pl.Int8).alias("source"))])
        return cls(s1=s1, recs=recs, truth=truth_frame(truth) if truth is not None else None, name=name)

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        path = Path(path)
        files = sorted(path.glob("*_source1.tsv"))
        if len(files) != 1:
            raise FileNotFoundError(f"{path}: expected exactly one *_source1.tsv, found {len(files)}")
        prefix = files[0].name[: -len("_source1.tsv")]
        src = [read_tsv(path / f"{prefix}_source{i}.tsv") for i in (1, 2, 3)]
        gt_path = path / f"{prefix}_ground_truth.tsv"
        gt = read_tsv(gt_path) if gt_path.exists() else None
        return cls.from_frames(*src, truth=gt, name=prefix)

    @property
    def countries(self) -> list[str]:
        return sorted(self.s1["country"].unique().to_list())

    def stats(self) -> dict:
        by = self.s1.group_by("country").len("s1").join(
            self.recs.group_by("country", "source").len("n").pivot("source", index="country", values="n"),
            on="country", how="left").sort("country")
        out = {"name": self.name, "s1": self.s1.height, "s2": int((self.recs["source"] == 2).sum()),
               "s3": int((self.recs["source"] == 3).sum()), "labelled": self.truth is not None,
               "by_country": [{"country": r["country"], "s1": r["s1"], "s2": r.get("2") or 0, "s3": r.get("3") or 0}
                              for r in by.iter_rows(named=True)]}
        if self.truth is not None:
            out["true_pairs"] = self.truth.height
            out["singletons"] = self.s1.height - self.truth["s1"].n_unique()
        return out


# ---------------------------------------------------------------- submission files
def id_lists(s1_ids: pl.Series, pairs: pl.DataFrame, col: str) -> pl.DataFrame:
    """One row per S1 (in s1_ids order), sorted de-duplicated comma-joined record ids."""
    lists = pairs.group_by("s1").agg(pl.col("rec").unique().sort().str.join(",").alias(col))
    return (pl.DataFrame({"source1_entity_id": s1_ids})
              .join(lists.rename({"s1": "source1_entity_id"}), on="source1_entity_id", how="left", maintain_order="left")
              .with_columns(pl.col(col).fill_null("")))


def submission_tables(s1_ids: pl.Series, matches: pl.DataFrame, cands: pl.DataFrame) -> dict[str, pl.DataFrame]:
    return {"matching_results.tsv": id_lists(s1_ids, matches, "matched_entity_ids"),
            "candidate_pairs.tsv": id_lists(s1_ids, cands, "candidate_entity_ids")}


def to_tsv(df: pl.DataFrame) -> str:
    return df.write_csv(separator="\t", quote_style="never")


def write_submission(out_dir: str | Path, s1_ids: pl.Series, matches: pl.DataFrame, cands: pl.DataFrame) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in submission_tables(s1_ids, matches, cands).items():
        (out_dir / name).write_text(to_tsv(df), encoding="utf-8")
        paths.append(out_dir / name)
    return paths


def validate_submission(matching: pl.DataFrame, candidates: pl.DataFrame | None, ds: Dataset) -> list[str]:
    """The challenge's submission rules (mirrors student_resource/utils/validate_submission.py).
    Returns a list of problems; empty means PASS."""
    issues: list[str] = []
    s1_ids = set(ds.s1["entity_id"].to_list())
    rec_ids = set(ds.recs["entity_id"].to_list())
    for label, df, col in (("matching_results", matching, "matched_entity_ids"),
                           ("candidate_pairs", candidates, "candidate_entity_ids")):
        if df is None:
            continue
        if df.columns != ["source1_entity_id", col]:
            issues.append(f"{label}: header must be source1_entity_id<TAB>{col}")
            continue
        ids = df["source1_entity_id"].to_list()
        if len(ids) != len(set(ids)):
            issues.append(f"{label}: duplicate source1_entity_id rows")
        if missing := s1_ids - set(ids):
            issues.append(f"{label}: {len(missing)} Source-1 entities missing")
        if extra := set(ids) - s1_ids:
            issues.append(f"{label}: {len(extra)} unknown Source-1 ids")
        for s1, lst in df.iter_rows():
            items = [x for x in lst.split(",") if x] if lst else []
            if len(items) != len(set(items)):
                issues.append(f"{label}: duplicate ids in the list of {s1}")
                break
            if bad := [x for x in items if not (x.startswith("S2-") or x.startswith("S3-")) or x not in rec_ids]:
                issues.append(f"{label}: {s1} lists ids that are not Source-2/3 records of this dataset ({bad[0]})")
                break
    if candidates is not None and not issues:
        cand = dict(candidates.iter_rows())
        for s1, lst in matching.iter_rows():
            if lst and not set(lst.split(",")) <= set((cand.get(s1) or "").split(",")):
                issues.append(f"warning: matches of {s1} are not a subset of its candidates")
                break
    return issues
