"""Paths and runtime settings (all overridable by environment variables)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get("RESOLVER_ARTIFACTS", ROOT / "artifacts"))
DEMO_DATA = Path(os.environ.get("RESOLVER_DEMO_DATA", ROOT / "data" / "demo"))

MODEL_PATH = ARTIFACTS / "model.joblib"
TRAIN_REPORT = ARTIFACTS / "train_report.json"
TEST_REPORT = ARTIFACTS / "test_report.json"
INDEX_PATH = ARTIFACTS / "index.joblib"

# Public-demo guard rails for uploaded batch jobs.
MAX_UPLOAD_MB = float(os.environ.get("RESOLVER_MAX_UPLOAD_MB", 25))
MAX_UPLOAD_ROWS = int(os.environ.get("RESOLVER_MAX_UPLOAD_ROWS", 150_000))
MAX_JOBS = int(os.environ.get("RESOLVER_MAX_JOBS", 20))
