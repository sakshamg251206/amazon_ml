"""Background batch-resolution jobs for uploaded datasets (one worker; results kept in memory)."""
from __future__ import annotations

import io
import json
import threading
import time
import traceback
import uuid
import zipfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import polars as pl

from resolver import config
from resolver.engine import Matcher, resolve
from resolver.evaluate import candidate_metrics, summary
from resolver.io import Dataset, submission_tables, to_tsv, validate_submission


@dataclass
class Job:
    id: str
    name: str
    status: str = "queued"            # queued | running | done | failed
    stage: str = "waiting for a worker"
    created: float = field(default_factory=time.time)
    finished: float | None = None
    error: str | None = None
    result: dict | None = None
    files: dict[str, str] = field(default_factory=dict)

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "status": self.status, "stage": self.stage, "error": self.error,
                "created": self.created, "elapsed_s": round((self.finished or time.time()) - self.created, 1),
                "result": self.result}


class JobManager:
    def __init__(self, matcher: Matcher):
        self.matcher = matcher
        self.jobs: OrderedDict[str, Job] = OrderedDict()
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1)   # one job at a time keeps memory predictable

    def submit(self, ds: Dataset, name: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], name=name)
        with self.lock:
            self.jobs[job.id] = job
            while len(self.jobs) > config.MAX_JOBS:
                self.jobs.popitem(last=False)
        self.pool.submit(self._run, job, ds)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def _run(self, job: Job, ds: Dataset) -> None:
        job.status = "running"
        try:
            def progress(stage: str) -> None:
                job.stage = stage
            res = resolve(ds, self.matcher, progress=progress)
            job.stage = "writing and validating files"
            s1_ids = ds.s1["entity_id"]
            tables = submission_tables(s1_ids, res.matches, res.candidates)
            issues = validate_submission(tables["matching_results.tsv"], tables["candidate_pairs.tsv"], ds)
            job.files = {name: to_tsv(df) for name, df in tables.items()}
            job.result = self._summarise(ds, res, issues)
            job.files["report.json"] = json.dumps(job.result, indent=1, default=float)
            job.status, job.stage = "done", "finished"
        except Exception as e:  # surfaced to the user; the server keeps running
            job.status, job.stage, job.error = "failed", "failed", f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            job.finished = time.time()

    @staticmethod
    def _summarise(ds: Dataset, res, issues: list[str]) -> dict:
        m = res.matches
        by_c = (ds.s1.select(pl.col("entity_id").alias("s1"), "country")
                  .join(m.group_by("s1").len("n"), on="s1", how="left")
                  .group_by("country").agg(pl.len().alias("s1"), pl.col("n").is_not_null().mean().alias("matched_share"),
                                           pl.col("n").fill_null(0).mean().alias("matches_per_s1")).sort("country"))
        out = {
            "dataset": ds.stats(),
            "matched_pairs": m.height,
            "matched_s1": m["s1"].n_unique(),
            "candidate_pairs": res.candidates.height,
            "by_country": by_c.to_dicts(),
            "validation": {"pass": not [i for i in issues if not i.startswith("warning")], "issues": issues},
            "timings": {k: round(v, 2) for k, v in res.timings.items()},
        }
        if ds.truth is not None:
            ids = ds.s1["entity_id"]
            out["metrics"] = summary(m, ds.truth, ids)
            out["blocking"] = candidate_metrics(res.candidates, ds.truth, ids, ds.recs.height)
            out["metrics_by_country"] = [{"country": c, **summary(m, ds.truth, g["entity_id"])}
                                         for (c,), g in ds.s1.group_by("country", maintain_order=True)]
        names = dict(ds.s1.select("entity_id", "business_name").iter_rows())
        rec_names = dict(ds.recs.select("entity_id", "business_name").iter_rows())
        prev = m.sort("p", descending=True).group_by("s1", maintain_order=True).agg(pl.col("rec"), pl.col("p")).head(40)
        out["preview"] = [{"s1": r["s1"], "name": names.get(r["s1"]),
                           "matches": [{"rec": x, "name": rec_names.get(x), "p": round(float(p), 3)}
                                       for x, p in zip(r["rec"], r["p"])]} for r in prev.to_dicts()]
        return out

    def archive(self, job: Job) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for name, text in job.files.items():
                z.writestr(f"output/{name}", text)
        return buf.getvalue()
