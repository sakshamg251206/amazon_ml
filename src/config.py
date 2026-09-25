import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("ER_DATA_DIR", ROOT / "student_resource" / "dataset"))
WORK_DIR = Path(os.environ.get("ER_WORK_DIR", ROOT / "work"))
OUTPUT_DIR = Path(os.environ.get("ER_OUTPUT_DIR", ROOT / "output"))
SEED = 42
