"""Generic EDA tool - Flask application (guided wizard + report manager)."""
import logging
import os
import secrets
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import (Flask, abort, flash, jsonify, redirect, render_template, request,
                   send_from_directory, url_for)
from werkzeug.utils import secure_filename

import config
from eda import analysis, loader, report, storage
from eda.cleaning import CONVERT_TARGETS, apply_cleaning
from eda.dataset_info import (classify_columns, column_flags, dataset_summary,
                              detect_type_issues, missing_table, sample_values)
from eda import plots

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("eda.app")

for d in (config.UPLOAD_DIR, config.REPORT_DIR):
    d.mkdir(exist_ok=True)


def _secret_key():
    if config.SECRET_KEY_FILE.exists():
        return config.SECRET_KEY_FILE.read_text().strip()
    key = secrets.token_hex(32)
    config.SECRET_KEY_FILE.write_text(key)
    return key


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("EDA_SECRET_KEY") or _secret_key()
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS, thread_name_prefix="eda-report")
storage.mark_interrupted()
storage.cleanup_stale()

WIZARD_STEPS = ["Upload", "Sheet", "Review & clean", "Analysis setup", "Outliers", "Report"]


# ====================================================================== filters / context
@app.template_filter("filesize")
def filesize(n):
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


@app.template_filter("dt")
def fmt_dt(v):
    try:
        return datetime.fromisoformat(v).strftime("%d %b %Y, %H:%M")
    except (TypeError, ValueError):
        return v or "-"


@app.template_filter("num")
def fmt_num(v):
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return "-"


@app.context_processor
def inject_globals():
    return {"wizard_steps": WIZARD_STEPS, "max_mb": config.MAX_CONTENT_LENGTH // (1024 * 1024)}


# ====================================================================== helpers
def _entry_or_404(job_id):
    if not storage.valid_id(job_id):
        abort(404)
    entry = storage.get_entry(job_id)
    if not entry:
        abort(404)
    return entry


def _wizard_entry(job_id):
    """Entry for a wizard page; bounces away if the job is already finished/running."""
    entry = _entry_or_404(job_id)
    if entry["status"] == "completed":
        abort(redirect(url_for("report_view", job_id=job_id)))
    if entry["status"] == "processing":
        abort(redirect(url_for("wizard_progress", job_id=job_id)))
    return entry


def _read(job_id, name):
    p = storage.job_dir(job_id) / name
    if not p.exists():
        return None
    return pd.read_pickle(p)


def _step_endpoint(entry):
    if entry["status"] == "completed":
        return "report_view"
    if entry["status"] == "processing":
        return "wizard_progress"
    return {"sheet": "wizard_sheet", "clean": "wizard_clean", "configure": "wizard_configure",
            "outliers": "wizard_outliers"}.get(entry.get("step"), "wizard_clean")


def _prepare_dataset(job_id, entry, sheet=None):
    """Load the file (and chosen sheet), snapshot it, and record dimensions."""
    path = storage.job_dir(job_id) / entry["stored_name"]
    df = loader.load_dataset(path, entry["file_type"].lower(), sheet)
    df.to_pickle(storage.job_dir(job_id) / "raw.pkl")
    for stale in ("cleaned.pkl",):
        (storage.job_dir(job_id) / stale).unlink(missing_ok=True)
    storage.update_state(job_id, sheet=sheet, raw_columns=list(df.columns), config=None,
                         outliers=None, cleaning=None)
    storage.update_entry(job_id, rows=int(df.shape[0]), columns=int(df.shape[1]), sheet=sheet,
                         step="clean", error=None)
    return df


# ====================================================================== dashboard
@app.route("/")
def index():
    storage.cleanup_stale()
    entries = storage.all_entries()
    done = [e for e in entries if e["status"] == "completed"]
    stats = {
        "total_reports": len(done),
        "total_rows": sum(e.get("rows") or 0 for e in done),
        "total_cols": sum(e.get("columns") or 0 for e in done),
        "total_size": sum(e.get("file_size") or 0 for e in done),
    }
    return render_template("index.html", reports=entries, stats=stats)


# ====================================================================== upload
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("upload.html", current=0)

    file = request.files.get("file")
    if not file or not file.filename:
        flash("Please choose a file to upload.", "danger")
        return redirect(url_for("upload"))
    original = Path(file.filename).name
    ext = original.rsplit(".", 1)[-1].lower() if "." in original else ""
    if ext not in config.ALLOWED_EXTENSIONS:
        flash("Unsupported file type. Please upload a .csv, .xls or .xlsx file.", "danger")
        return redirect(url_for("upload"))

    job_id = storage.new_id()
    jdir = storage.job_dir(job_id)
    jdir.mkdir(parents=True)
    stored = secure_filename(original) or f"dataset.{ext}"
    path = jdir / stored
    file.save(path)
    size = path.stat().st_size
    if size == 0:
        storage.delete_job_files(job_id)
        flash("The uploaded file is empty.", "danger")
        return redirect(url_for("upload"))

    title = (request.form.get("title") or "").strip()[:120] or Path(original).stem
    entry = {"id": job_id, "title": title, "filename": original, "stored_name": stored,
             "file_type": ext.upper(), "file_size": size,
             "upload_date": datetime.now().isoformat(timespec="seconds"),
             "rows": None, "columns": None, "sheet": None, "status": "draft",
             "step": "clean", "error": None}
    storage.create_entry(entry)

    try:
        if ext in ("xls", "xlsx"):
            sheets = loader.list_sheets(path, ext)
            if not sheets:
                raise ValueError("The workbook contains no sheets.")
            storage.save_state(job_id, {"sheets": sheets})
            if len(sheets) > 1:
                storage.update_entry(job_id, step="sheet")
                return redirect(url_for("wizard_sheet", job_id=job_id))
            _prepare_dataset(job_id, entry, sheets[0]["name"])
        else:
            storage.save_state(job_id, {})
            _prepare_dataset(job_id, entry)
    except ValueError as exc:
        storage.purge_job(job_id)
        flash(str(exc), "danger")
        return redirect(url_for("upload"))
    except Exception:
        log.exception("Unexpected error while reading upload")
        storage.purge_job(job_id)
        flash("Something went wrong while reading that file. Is it a valid, unprotected file?", "danger")
        return redirect(url_for("upload"))
    flash("File uploaded. Review the dataset below.", "success")
    return redirect(url_for("wizard_clean", job_id=job_id))


# ====================================================================== wizard: sheet (Step 3)
@app.route("/wizard/<job_id>/sheet", methods=["GET", "POST"])
def wizard_sheet(job_id):
    entry = _wizard_entry(job_id)
    state = storage.get_state(job_id)
    sheets = state.get("sheets") or []
    if not sheets:
        return redirect(url_for("wizard_clean", job_id=job_id))
    if request.method == "POST":
        chosen = request.form.get("sheet")
        if chosen not in [s["name"] for s in sheets]:
            flash("Please choose one of the listed sheets.", "danger")
            return redirect(url_for("wizard_sheet", job_id=job_id))
        try:
            _prepare_dataset(job_id, entry, chosen)
        except ValueError as exc:
            flash(str(exc), "danger")
            return redirect(url_for("wizard_sheet", job_id=job_id))
        return redirect(url_for("wizard_clean", job_id=job_id))
    return render_template("wizard/sheet.html", entry=entry, sheets=sheets,
                           selected=state.get("sheet"), current=1)


# ====================================================================== wizard: review & clean (Steps 4-12)
@app.route("/wizard/<job_id>/clean", methods=["GET", "POST"])
def wizard_clean(job_id):
    entry = _wizard_entry(job_id)
    df = _read(job_id, "raw.pkl")
    if df is None:
        return redirect(url_for("wizard_sheet", job_id=job_id))
    cols = list(df.columns)

    if request.method == "POST":
        drop = [c for i, c in enumerate(cols) if request.form.get(f"drop_{i}")]
        if len(drop) >= len(cols):
            flash("You cannot drop every column.", "danger")
            return redirect(url_for("wizard_clean", job_id=job_id))
        conv = {}
        for i, c in enumerate(cols):
            t = request.form.get(f"type_{i}", "keep")
            if t in CONVERT_TARGETS and t != "keep" and c not in drop:
                conv[c] = t
        threshold = request.form.get("threshold", type=float) or config.HIGH_NULL_DEFAULT_PCT
        cleaned, log_lines, warns, dups = apply_cleaning(
            df.copy(), drop, conv, bool(request.form.get("drop_duplicates")))
        if cleaned.empty:
            flash("Cleaning would leave no rows. Please revise your choices.", "danger")
            return redirect(url_for("wizard_clean", job_id=job_id))
        cleaned.to_pickle(storage.job_dir(job_id) / "cleaned.pkl")          # Step 12 snapshot
        storage.update_state(job_id, cleaning={"dropped": drop, "conversions": conv, "log": log_lines,
                                               "duplicates_removed": dups, "threshold": threshold},
                             config=None, outliers=None)
        storage.update_entry(job_id, step="configure")
        for w in warns:
            flash(w, "warning")
        flash(f"Cleaned snapshot saved: {len(cleaned):,} rows × {cleaned.shape[1]} columns.", "success")
        return redirect(url_for("wizard_configure", job_id=job_id))

    threshold = request.args.get("threshold", type=float)
    state = storage.get_state(job_id)
    if threshold is None:
        threshold = (state.get("cleaning") or {}).get("threshold", config.HIGH_NULL_DEFAULT_PCT)
    threshold = min(max(threshold, 1), 100)

    summary = dataset_summary(df)
    cls = classify_columns(df)
    kind_of = {c: k for k, v in cls.items() for c in v}
    issues = detect_type_issues(df)
    mt = missing_table(df).set_index("column")
    rows = []
    for i, c in enumerate(cols):
        iss = issues.get(c)
        rows.append({
            "idx": i, "name": c, "dtype": str(df[c].dtype), "kind": kind_of.get(c, "other"),
            "unique": int(df[c].nunique()), "missing": int(mt.loc[c, "missing"]),
            "missing_pct": float(mt.loc[c, "missing_pct"]), "issue": iss,
            "flags": column_flags(df[c]), "sample": sample_values(df[c]),
            "drop": float(mt.loc[c, "missing_pct"]) > threshold,
            "convert": iss["convert"] if iss and iss["preselect"] else "keep",
        })
    return render_template(
        "wizard/clean.html", entry=entry, summary=summary, rows=rows, threshold=threshold,
        head=df.head(10).to_html(classes="table table-sm table-striped small", border=0,
                                 index=False, max_cols=50),
        missing_plot=plots.missing_bar(mt.reset_index()), n_high=sum(r["drop"] for r in rows),
        n_issues=len(issues), current=2)


# ====================================================================== wizard: analysis setup (Steps 15-18)
@app.route("/wizard/<job_id>/configure", methods=["GET", "POST"])
def wizard_configure(job_id):
    entry = _wizard_entry(job_id)
    df = _read(job_id, "cleaned.pkl")
    if df is None:
        flash("Please complete the review & clean step first.", "warning")
        return redirect(url_for("wizard_clean", job_id=job_id))
    cls = classify_columns(df)
    kind_of = {c: k for k, v in cls.items() for c in v}
    usable = [c for c in df.columns if kind_of.get(c) in ("numeric", "categorical")]

    if request.method == "POST":
        pairs, seen = [], set()
        for x, y in zip(request.form.getlist("pair_x"), request.form.getlist("pair_y")):
            if x in usable and y in usable and x != y and (x, y) not in seen:
                seen.add((x, y))
                pairs.append({"x": x, "y": y})
        pairs = pairs[: config.MAX_BIVARIATE_PAIRS]
        pp = [c for c in request.form.getlist("pp_cols") if kind_of.get(c) == "numeric"]
        pp = pp[: config.MAX_PAIRPLOT_COLS]
        hue = request.form.get("pp_hue", "")
        if hue and not (kind_of.get(hue) == "categorical" and 2 <= df[hue].nunique() <= 8):
            hue = ""
        thr = request.form.get("corr_threshold", type=float) or 0.8
        sample_rows = request.form.get("profile_sample_rows", type=int) or config.PROFILE_SAMPLE_ROWS
        storage.update_state(job_id, config={
            "pairs": pairs, "pairplot_cols": pp, "pairplot_hue": hue,
            "corr_threshold": min(max(thr, 0.1), 0.99),
            "run_profile": bool(request.form.get("run_profile")),
            "profile_sample_rows": min(max(sample_rows, 1000), 1_000_000)})
        storage.update_entry(job_id, step="outliers")
        return redirect(url_for("wizard_outliers", job_id=job_id))

    state = storage.get_state(job_id)
    saved = state.get("config") or {}
    suggestions = analysis.suggest_pairs(df, 5)
    default_pp = saved.get("pairplot_cols")
    if default_pp is None:
        default_pp = []
        for p in analysis.suggest_pairs(df, 6):
            for c in (p["x"], p["y"]):
                if c not in default_pp and len(default_pp) < 5:
                    default_pp.append(c)
        for c in cls["numeric"]:
            if len(default_pp) < min(4, len(cls["numeric"])) and c not in default_pp:
                default_pp.append(c)
    hue_options = [c for c in cls["categorical"] if 2 <= df[c].nunique() <= 8]
    return render_template(
        "wizard/configure.html", entry=entry, current=3,
        columns_meta=[{"name": c, "kind": kind_of[c]} for c in usable],
        numeric=cls["numeric"], hue_options=hue_options, suggestions=suggestions,
        saved=saved, default_pp=default_pp, shape=df.shape, max_pp=config.MAX_PAIRPLOT_COLS,
        max_pairs=config.MAX_BIVARIATE_PAIRS, default_sample=config.PROFILE_SAMPLE_ROWS)


# ====================================================================== wizard: outliers (Step 19) + build
@app.route("/wizard/<job_id>/outliers", methods=["GET", "POST"])
def wizard_outliers(job_id):
    entry = _wizard_entry(job_id)
    df = _read(job_id, "cleaned.pkl")
    state = storage.get_state(job_id)
    if df is None or not state.get("config"):
        flash("Please complete the earlier steps first.", "warning")
        return redirect(url_for("wizard_clean", job_id=job_id))
    num = classify_columns(df)["numeric"]

    if request.method == "POST":
        method = request.form.get("method", "iqr")
        action = request.form.get("action", "flag")
        if method not in ("iqr", "zscore") or action not in ("flag", "remove", "cap"):
            abort(400)
        k = min(max(request.form.get("iqr_k", type=float) or 1.5, 0.5), 10)
        z = min(max(request.form.get("z", type=float) or 3.0, 1), 10)
        cols = [c for c in request.form.getlist("cols") if c in num]
        if action != "flag" and not cols:
            action = "flag"
            flash("No columns were selected, so outliers will only be reported, not changed.", "info")
        storage.update_state(job_id, outliers={"method": method, "iqr_k": k, "z": z,
                                               "action": action, "columns": cols})
        storage.update_entry(job_id, status="processing", step="generating", progress=0,
                             progress_msg="Queued", error=None)
        executor.submit(report.generate, app, job_id)
        return redirect(url_for("wizard_progress", job_id=job_id))

    saved = state.get("outliers") or {"method": "iqr", "iqr_k": 1.5, "z": 3.0, "action": "flag",
                                      "columns": None}
    summary = analysis.outlier_summary(df, num, saved["iqr_k"], saved["z"])
    with_out = summary[summary["IQR outliers"] > 0].sort_values("IQR %", ascending=False)
    grid = plots.box_grid(analysis.plot_sample(df), list(with_out["column"])) if len(with_out) else None
    selected = saved["columns"] if saved["columns"] is not None else list(with_out["column"])
    return render_template("wizard/outliers.html", entry=entry, summary=summary.to_dict("records"),
                           saved=saved, selected=selected, grid=grid, n_rows=len(df), current=4)


# ====================================================================== wizard: progress
@app.route("/wizard/<job_id>/progress")
def wizard_progress(job_id):
    entry = _entry_or_404(job_id)
    return render_template("wizard/progress.html", entry=entry, current=5)


@app.route("/wizard/<job_id>/status")
def wizard_status(job_id):
    entry = _entry_or_404(job_id)
    return jsonify({"status": entry["status"], "progress": entry.get("progress", 0),
                    "message": entry.get("progress_msg", ""), "error": entry.get("error"),
                    "report_url": url_for("report_view", job_id=job_id)})


@app.route("/wizard/<job_id>/resume")
def wizard_resume(job_id):
    entry = _entry_or_404(job_id)
    return redirect(url_for(_step_endpoint(entry), job_id=job_id))


# ====================================================================== reports
@app.route("/report/<job_id>")
def report_view(job_id):
    entry = _entry_or_404(job_id)
    if entry["status"] != "completed":
        return redirect(url_for(_step_endpoint(entry), job_id=job_id))
    rdir = storage.report_dir(job_id)
    return render_template("report_view.html", entry=entry,
                           has_profile=(rdir / "profile.html").exists(),
                           has_final=(rdir / "final_dataset.csv").exists())


_REPORT_FILES = {"report.html", "profile.html", "cleaned_dataset.csv", "final_dataset.csv"}


@app.route("/report/<job_id>/files/<filename>")
def report_file(job_id, filename):
    _entry_or_404(job_id)
    if filename not in _REPORT_FILES:
        abort(404)
    is_csv = filename.endswith(".csv")
    return send_from_directory(storage.report_dir(job_id), filename, as_attachment=is_csv,
                               download_name=f"{Path(filename).stem}_{job_id[:8]}.csv" if is_csv else None)


@app.route("/report/<job_id>/delete", methods=["POST"])
def report_delete(job_id):
    entry = _entry_or_404(job_id)
    if entry["status"] == "processing":
        flash("This report is still being generated and can't be deleted yet.", "warning")
        return redirect(url_for("index"))
    storage.purge_job(job_id)
    flash(f"Deleted “{entry['title']}”.", "success")
    return redirect(url_for("index"))


# ====================================================================== errors
@app.errorhandler(404)
def not_found(_):
    return render_template("error.html", code=404, message="That page or report doesn't exist."), 404


@app.errorhandler(413)
def too_large(_):
    flash(f"That file is larger than the {config.MAX_CONTENT_LENGTH // (1024 * 1024)} MB limit.", "danger")
    return redirect(url_for("upload"))


@app.errorhandler(400)
def bad_request(_):
    return render_template("error.html", code=400, message="The request couldn't be understood."), 400


@app.errorhandler(500)
def server_error(exc):
    log.exception("Unhandled error: %s", exc)
    return render_template("error.html", code=500,
                           message="Something went wrong on our side. Please try again."), 500


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG,
            exclude_patterns=["*/uploads/*", "*/reports/*", "*/.secret_key"])
