"""Central configuration for the EDA app."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = Path(os.environ.get("EDA_UPLOAD_DIR", BASE_DIR / "uploads"))
REPORT_DIR = Path(os.environ.get("EDA_REPORT_DIR", BASE_DIR / "reports"))
METADATA_FILE = Path(os.environ.get("EDA_METADATA_FILE", BASE_DIR / "metadata.json"))
SECRET_KEY_FILE = Path(os.environ.get("EDA_SECRET_KEY_FILE", BASE_DIR / ".secret_key"))

MAX_CONTENT_LENGTH = 100 * 1024 * 1024          # 100 MB upload limit
ALLOWED_EXTENSIONS = {"csv", "xls", "xlsx"}

# --- wizard defaults -------------------------------------------------------
HIGH_NULL_DEFAULT_PCT = 50       # columns above this % missing are pre-selected for removal
MAX_BIVARIATE_PAIRS = 20
MAX_PAIRPLOT_COLS = 6
STALE_DRAFT_HOURS = 24           # abandoned wizard sessions are purged after this

# --- performance guards (large datasets) -----------------------------------
PLOT_SAMPLE_ROWS = 50_000        # rows used for plots (stats always use all rows)
SCATTER_MAX_POINTS = 5_000
PAIRPLOT_MAX_POINTS = 2_000
MAX_UNIVARIATE_PLOTS = 60        # per column kind; extra columns still get stats tables
PROFILE_SAMPLE_ROWS = int(os.environ.get("EDA_PROFILE_SAMPLE_ROWS", 50_000))
MAX_WORKERS = int(os.environ.get("EDA_MAX_WORKERS", 2))   # concurrent report builds

HOST = "0.0.0.0"
PORT = 8501
DEBUG = os.environ.get("EDA_DEBUG", "1") == "1"
