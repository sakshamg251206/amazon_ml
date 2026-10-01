"""HTTP API + static web app.

The demo index (the held-out synthetic test split, resolved with the trained model) is loaded once
at startup from artifacts/; if it is missing it is built (synthetic data -> train -> index, ~1 min).
"""
from __future__ import annotations

import io
import json
import logging
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import polars as pl
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import resolver
from resolver import config
from resolver.api.jobs import JobManager
from resolver.io import Dataset, read_tsv

log = logging.getLogger("resolver")
WEB = Path(__file__).resolve().parents[1] / "web"
STATE: dict = {}


def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def load_state() -> None:
    if not config.INDEX_PATH.exists():
        log.info("no demo index at %s: building it (synthetic data -> train -> index)", config.INDEX_PATH)
        from resolver.cli import main as cli
        cli(["demo"])
    t0 = time.time()
    index = joblib.load(config.INDEX_PATH)
    STATE.update(index=index, jobs=JobManager(index.matcher), train_report=_load_json(config.TRAIN_REPORT),
                 test_report=_load_json(config.TEST_REPORT), examples=index.examples())
    log.info("demo index loaded in %.1fs: %d S1, %d records", time.time() - t0, index.raw_s1.height, index.raw_rec.height)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_state()
    yield


app = FastAPI(title="Business Entity Resolver", version=resolver.__version__, lifespan=lifespan,
              description="Match noisy business records across sources to a reference (Amazon ML Challenge 2026).")


def _index():
    if "index" not in STATE:
        raise HTTPException(503, "index is loading")
    return STATE["index"]


class MatchRequest(BaseModel):
    name: str = Field(..., max_length=300, description="Business name as written in the record")
    address: str = Field("", max_length=500)
    country: str = Field(..., max_length=40)
    top: int = Field(5, ge=1, le=10)


@app.get("/api/health")
def health():
    return {"status": "ok" if "index" in STATE else "loading", "version": resolver.__version__}


@app.get("/api/meta")
def meta():
    idx = _index()
    return {"version": resolver.__version__, "dataset": idx.ds.stats(), "countries": sorted(idx.aliases),
            "metrics": idx.metrics, "model": {k: v for k, v in idx.matcher.meta.items() if not k.startswith("features")},
            "n_features": {"stage1": len(idx.matcher.meta["features_stage1"]), "stage2": len(idx.matcher.meta["features_stage2"])},
            "examples": STATE["examples"]}


@app.get("/api/report")
def report():
    return {"train": STATE.get("train_report"), "test": STATE.get("test_report")}


@app.post("/api/match")
def match(req: MatchRequest):
    if not req.name.strip() and not req.address.strip():
        raise HTTPException(422, "give at least a name or an address")
    return _index().match(req.name, req.address, req.country, top=req.top)


@app.get("/api/entities")
def entities(country: str | None = None, q: str | None = Query(None, max_length=100), outcome: str | None = None,
             offset: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=100)):
    return _index().list_entities(country, q, outcome, offset, limit)


@app.get("/api/entities/{s1}")
def entity(s1: str):
    d = _index().entity(s1)
    if d is None:
        raise HTTPException(404, f"unknown entity {s1}")
    return d


# ------------------------------------------------------------------ batch jobs
async def _read_upload(f: UploadFile | None, label: str):
    if f is None:
        return None
    data = await f.read(int(config.MAX_UPLOAD_MB * 1024 * 1024) + 1)
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"{label} is larger than {config.MAX_UPLOAD_MB:g} MB (demo limit)")
    try:
        df = read_tsv(data)
    except Exception as e:
        raise HTTPException(422, f"{label}: cannot read as a tab-separated file ({e})")
    if df.height > config.MAX_UPLOAD_ROWS:
        raise HTTPException(413, f"{label} has {df.height:,} rows; the demo limit is {config.MAX_UPLOAD_ROWS:,}")
    return df


@app.post("/api/jobs")
async def create_job(source1: UploadFile = File(...), source2: UploadFile = File(...), source3: UploadFile = File(...),
                     ground_truth: UploadFile | None = File(None)):
    frames = [await _read_upload(f, n) for f, n in ((source1, "source 1"), (source2, "source 2"), (source3, "source 3"))]
    gt = await _read_upload(ground_truth, "ground truth")
    try:
        ds = Dataset.from_frames(*frames, truth=gt, name=source1.filename or "upload")
    except ValueError as e:
        raise HTTPException(422, str(e))
    return STATE["jobs"].submit(ds, ds.name).public()


def _sample_dataset() -> Dataset:
    """The France slice of the held-out test split: a country the model never saw in training."""
    ds = _index().ds
    s1 = ds.s1.filter(pl.col("country") == "France")
    recs = ds.recs.filter(pl.col("country") == "France")
    truth = ds.truth.join(s1.select(pl.col("entity_id").alias("s1")), on="s1") if ds.truth is not None else None
    return Dataset(s1=s1, recs=recs, truth=truth, name="sample_france")


@app.post("/api/jobs/sample")
def create_sample_job():
    return STATE["jobs"].submit(_sample_dataset(), "sample_france").public()


@app.get("/api/sample.zip")
def sample_zip():
    ds = _sample_dataset()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i in (1, 2, 3):
            df = ds.s1 if i == 1 else ds.recs.filter(pl.col("source") == i).drop("source")
            z.writestr(f"sample_source{i}.tsv", df.write_csv(separator="\t", quote_style="never"))
        if ds.truth is not None:
            gt = ds.truth.group_by("s1").agg(pl.col("rec").sort().str.join(","))
            gt = ds.s1.select(pl.col("entity_id").alias("source1_entity_id")).join(
                gt.rename({"s1": "source1_entity_id", "rec": "matched_entity_ids"}), on="source1_entity_id", how="left") \
                .with_columns(pl.col("matched_entity_ids").fill_null(""))
            z.writestr("sample_ground_truth.tsv", gt.write_csv(separator="\t", quote_style="never"))
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": "attachment; filename=sample_france.zip"})


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = STATE["jobs"].get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job (jobs are kept in memory and expire)")
    return job.public()


@app.get("/api/jobs/{job_id}/download")
def job_download(job_id: str):
    job = STATE["jobs"].get(job_id)
    if job is None or job.status != "done":
        raise HTTPException(404, "job not finished")
    return Response(STATE["jobs"].archive(job), media_type="application/zip",
                    headers={"Content-Disposition": f"attachment; filename=resolved_{job.name}.zip"})


@app.exception_handler(ValueError)
def value_error(_, exc: ValueError):
    return JSONResponse({"detail": str(exc)}, status_code=422)


# ------------------------------------------------------------------ web app
@app.get("/", include_in_schema=False)
def home():
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=WEB), name="static")
