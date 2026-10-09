# Auto EDA – guided exploratory data analysis

Upload a CSV / XLS / XLSX file (≤ 100 MB), walk through a short wizard (sheet → clean → analysis choices →
outliers) and get a self-contained HTML EDA report plus a ydata-profiling report. Reports are stored,
listed on a dashboard, viewable in an embedded iframe, and deletable.

## Run
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py                                         # http://localhost:8501  (binds 0.0.0.0:8501, debug on)
```

## Run with Docker
Build and start the app with a named volume so uploads, reports, metadata, and the generated Flask
secret key persist when the container is replaced:
```bash
docker build -t eda-app .
docker run --name eda-app --restart unless-stopped -p 8501:8501 -v eda-data:/data eda-app
```
Open http://localhost:8501. The container runs Gunicorn with one worker (the JSON registry uses
in-process locking), disables Flask debug mode, and runs as a non-root user. For production, put it
behind an HTTPS reverse proxy and restrict network access; the app does not provide authentication.

## Wizard ⇄ your 21 steps
| Wizard page | Steps | What the user decides |
|---|---|---|
| Upload | 1-2 | file + optional title (encoding/delimiter auto-detected) |
| Sheet | 3 | which Excel sheet (skipped for CSV / single-sheet files) |
| Review & clean | 4-12 | summary, rows/cols, numeric vs categorical, **type fixes** (suggested ★), missing-value chart, **high-null columns to drop** (adjustable threshold), duplicate removal → cleaned snapshot saved |
| Analysis setup | 15-18 | **bivariate pairs** (num-num, num-cat, cat-cat auto-detected), pair-plot columns/colour, correlation threshold, ydata sample size |
| Outliers | 19 | IQR or Z-score, report-only / cap / remove rows, per-column selection |
| Report | 13-14, 20-21 | built in the background with a progress bar: univariate stats, feature profiling, HTML report, ydata-profiling |

(Step 11 is absent from the brief; it is treated as "apply the removals" and is part of the clean page.)

## Large datasets
* Statistics always use every row; **plots** use a ≤ 50 k-row sample, scatter ≤ 5 k, pair plot ≤ 2 k points.
* ydata-profiling runs in **minimal mode** on a random sample (default 50 k rows, configurable in the wizard
  or via `EDA_PROFILE_SAMPLE_ROWS`).
* Report builds run in a thread pool (`EDA_MAX_WORKERS`, default 2). Measured: 900 k rows × 10 cols (94 MB CSV)
  → full report in ≈ 50 s, ≈ 0.8 GB peak RAM.

## Layout
```
app.py                Flask routes (dashboard, wizard, report viewer, delete)
config.py             limits and paths
eda/                  loader, dataset_info, cleaning, analysis, plots, report, storage
templates/, static/   Bootstrap 5 UI (base, index, upload, report_view, wizard/*, report/report.html)
uploads/<id>/         original file + wizard working state (temp pickles removed after build)
reports/<id>/         report.html, profile.html, cleaned_dataset.csv, final_dataset.csv
metadata.json         report registry (title, filename, date, rows, cols, size, type, status …)
```

## Operational notes
* **No login**: everyone who can reach the server sees all reports. Put it behind your network/VPN or a reverse proxy.
* **Debug mode** is on as specified (`EDA_DEBUG=0` turns it off). The Werkzeug debugger allows code execution –
  disable it on any shared server and use a WSGI server (single process, because metadata.json locking is in-process).
* Abandoned wizard sessions are purged after 24 h; builds interrupted by a restart are marked *failed*.
* `setuptools<81` is pinned because ydata-profiling still imports `pkg_resources`.
