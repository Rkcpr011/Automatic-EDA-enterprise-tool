"""Orchestrates Steps 12-21: runs the analyses and renders the final HTML report."""
import logging
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from flask import render_template

import config
from eda import analysis, plots, storage
from eda.dataset_info import (classify_columns, dataset_summary, missing_table)

log = logging.getLogger("eda.report")


# ------------------------------------------------------------------ formatting
def fmt_cell(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT:
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if v == 0:
            return "0"
        if abs(v) >= 1e9 or abs(v) < 1e-3:
            return f"{v:.3e}"
        return f"{v:,.2f}" if abs(v) >= 100 else f"{v:,.4f}".rstrip("0").rstrip(".")
    if isinstance(v, pd.Timestamp):
        return str(v)[:19]
    return str(v)


def df_html(df, index=True, classes="tbl"):
    if df is None or len(df) == 0:
        return "<p class='muted'>No data.</p>"
    shown = df.apply(lambda col: col.map(fmt_cell))
    return shown.to_html(classes=classes, border=0, index=index, escape=True)


# ------------------------------------------------------------------ ydata-profiling
def run_ydata(df, out_path, title, sample_rows):
    from ydata_profiling import ProfileReport
    sampled = len(df) > sample_rows
    sample = df.sample(sample_rows, random_state=42) if sampled else df
    profile = ProfileReport(sample, title=title, minimal=True, progress_bar=False, pool_size=1)
    profile.to_file(str(out_path))
    return sampled, len(sample)


# ------------------------------------------------------------------ public entry
def generate(app, job_id):
    """Run in a worker thread. Never raises - failures are recorded in metadata."""
    with app.app_context():
        try:
            _generate(job_id)
        except Exception as exc:
            log.error("Report generation failed for %s\n%s", job_id, traceback.format_exc())
            storage.update_entry(job_id, status="failed", step="outliers",
                                 error=f"{type(exc).__name__}: {str(exc)[:300]}")


def _progress(job_id, pct, msg):
    storage.update_entry(job_id, progress=pct, progress_msg=msg)


def _generate(job_id):
    meta = storage.get_entry(job_id)
    state = storage.get_state(job_id)
    jdir, rdir = storage.job_dir(job_id), storage.report_dir(job_id)
    rdir.mkdir(parents=True, exist_ok=True)
    cfg, ocfg, clean = state["config"], state["outliers"], state["cleaning"]

    _progress(job_id, 3, "Loading datasets")
    raw = pd.read_pickle(jdir / "raw.pkl")
    df = pd.read_pickle(jdir / "cleaned.pkl")          # Step 12 snapshot
    cls = classify_columns(df)
    num, cat, dt = cls["numeric"], cls["categorical"], cls["datetime"]
    ctx = {"meta": meta, "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "plot_note": None}
    if len(df) > config.PLOT_SAMPLE_ROWS:
        ctx["plot_note"] = (f"Plots use a random sample of {config.PLOT_SAMPLE_ROWS:,} of "
                            f"{len(df):,} rows; all statistics use every row.")

    # ---- Steps 4-9 (overview of the original data)
    _progress(job_id, 8, "Summarising dataset")
    ctx["raw_summary"] = dataset_summary(raw)
    ctx["clean_summary"] = dataset_summary(df)
    ctx["head_html"] = df_html(raw.head(10), index=False)
    mt = missing_table(raw)
    ctx["missing_html"] = df_html(mt[mt["missing"] > 0], index=False)
    ctx["missing_plot"] = plots.missing_bar(mt)
    ctx["null_threshold"] = clean.get("threshold")
    ctx["high_null"] = [c for c in mt[mt["missing_pct"] > clean.get("threshold", 50)]["column"]]
    ctx["clean_log"] = clean.get("log", [])
    col_rows = []
    for c in df.columns:
        k = next(k for k, v in cls.items() if c in v)
        col_rows.append({"column": c, "dtype": str(df[c].dtype), "class": k,
                         "unique": int(df[c].nunique()),
                         "missing %": round(100 * df[c].isna().mean(), 2)})
    ctx["types_html"] = df_html(pd.DataFrame(col_rows), index=False)

    # ---- Step 14
    _progress(job_id, 18, "Univariate statistics")
    ctx["num_stats_html"] = df_html(analysis.univariate_numeric_stats(df, num)) if num else None
    ctx["num_plots"] = [{"name": k, "img": v} for k, v in
                        analysis.univariate_numeric_plots(df, num).items() if v] if num else []
    ctx["n_num_unplotted"] = max(0, len(num) - config.MAX_UNIVARIATE_PLOTS)
    if cat:
        csum, cdet = analysis.univariate_categorical(df, cat)
        ctx["cat_summary_html"] = df_html(csum)
        ctx["cat_details"] = [{"name": c, "table": df_html(d["table"]), "plot": d["plot"]}
                              for c, d in cdet.items()]
    else:
        ctx["cat_summary_html"], ctx["cat_details"] = None, []
    ctx["dt_html"] = df_html(analysis.univariate_datetime(df, dt)) if dt else None

    # ---- Steps 15-16
    _progress(job_id, 35, "Bivariate analysis")
    biv = []
    for p in cfg.get("pairs", []):
        if p["x"] in df.columns and p["y"] in df.columns and p["x"] != p["y"]:
            try:
                r = analysis.bivariate_pair(df, p["x"], p["y"])
            except Exception as exc:
                r = {"x": p["x"], "y": p["y"], "kind": "error", "stats": None, "table": None,
                     "plot": None, "note": f"Could not analyse this pair: {exc}"}
            if r["stats"]:       # p-values that underflow to 0 are better shown as an inequality
                r["stats"] = {k: ("<1e-300" if "p-value" in k and v == 0 else v)
                              for k, v in r["stats"].items()}
            r["stats_html"] = df_html(pd.DataFrame([r["stats"]]).T.rename(columns={0: "value"})) \
                if r["stats"] else None
            r["table_html"] = df_html(r["table"]) if r["table"] is not None else None
            r["label"] = {"num_num": "Numerical vs Numerical", "num_cat": "Numerical vs Categorical",
                          "cat_cat": "Categorical vs Categorical"}.get(r["kind"], "Unsupported")
            biv.append(r)
    ctx["bivariate"] = biv

    # ---- Step 17
    _progress(job_id, 50, "Pair plot")
    pp_cols = cfg.get("pairplot_cols", [])
    ctx["pairplot_cols"] = pp_cols
    ctx["pairplot"] = analysis.make_pair_plot(df, pp_cols, cfg.get("pairplot_hue") or None) \
        if len(pp_cols) >= 2 else None

    # ---- Step 18
    _progress(job_id, 60, "Correlation analysis")
    corr = analysis.correlation_analysis(df, cfg.get("corr_threshold", 0.8))
    ctx["corr"] = {
        "threshold": corr["threshold"],
        "pearson_plot": corr.get("pearson_plot"), "spearman_plot": corr.get("spearman_plot"),
        "heatmap_note": corr.get("heatmap_note"),
        "top_pairs": df_html(corr["top_pairs"], index=False) if "top_pairs" in corr else None,
        "high_pairs": df_html(corr["high_pairs"], index=False) if "high_pairs" in corr else None,
        "n_high": len(corr["high_pairs"]) if "high_pairs" in corr else 0,
        "pearson_html": df_html(corr["pearson"]) if "pearson" in corr and len(corr["pearson"]) <= 20 else None,
        "cramers_plot": corr.get("cramers_plot"),
    }

    # ---- Step 19
    _progress(job_id, 72, "Outlier analysis")
    method, k, z, action = ocfg["method"], ocfg["iqr_k"], ocfg["z"], ocfg["action"]
    osum = analysis.outlier_summary(df, num, k, z)
    ctx["outlier_summary_html"] = df_html(osum, index=False) if len(osum) else None
    ocols = [c for c in ocfg.get("columns", []) if c in df.columns]
    final = df
    ctx["outlier_action"], ctx["outlier_method"] = action, method
    ctx["outlier_param"] = f"IQR multiplier {k}" if method == "iqr" else f"|z| > {z}"
    ctx["outlier_details_html"], ctx["outlier_images"], ctx["outlier_removed"] = None, [], 0
    if ocols:
        final, details, removed = analysis.treat_outliers(df, ocols, method, k, z, action)
        ctx["outlier_removed"] = removed
        ctx["outlier_details_html"] = df_html(pd.DataFrame(details).set_index("column"))
        if action != "flag":
            for e in sorted(details, key=lambda e: -e["outliers"])[:6]:
                if e["outliers"] > 0:
                    ctx["outlier_images"].append(
                        {"name": e["column"],
                         "img": plots.before_after_box(df[e["column"]], final[e["column"]], e["column"])})
    ctx["final_shape"] = final.shape

    # ---- Step 20
    _progress(job_id, 80, "Feature profiling")
    out_pct = {r["column"]: r["IQR %"] for _, r in osum.iterrows()} if len(osum) else {}
    fp = analysis.feature_profile(final, out_pct, corr.get("pair_lookup", {}))
    ctx["feature_html"] = df_html(fp, index=False, classes="tbl feat")

    # ---- Step 12 / outputs: dataset snapshots
    _progress(job_id, 86, "Saving cleaned dataset")
    df.to_csv(rdir / "cleaned_dataset.csv", index=False)
    has_final = action != "flag" and bool(ocols)
    if has_final:
        final.to_csv(rdir / "final_dataset.csv", index=False)
    ctx["has_final"] = has_final

    # ---- ydata-profiling (minimal mode on a sample)
    ctx["profile_note"], has_profile = None, False
    if cfg.get("run_profile", True):
        _progress(job_id, 90, "Running ydata-profiling (minimal mode)")
        try:
            sampled, n_used = run_ydata(final, rdir / "profile.html", meta["title"],
                                        cfg.get("profile_sample_rows", config.PROFILE_SAMPLE_ROWS))
            has_profile = True
            ctx["profile_note"] = (f"Minimal-mode ydata-profiling report on a random sample of "
                                   f"{n_used:,} rows." if sampled else
                                   "Minimal-mode ydata-profiling report on the full dataset.")
        except Exception as exc:
            log.error("ydata-profiling failed: %s", traceback.format_exc())
            ctx["profile_note"] = f"ydata-profiling could not be generated: {exc}"
    ctx["has_profile"] = has_profile

    # ---- Step 21: HTML report
    _progress(job_id, 96, "Rendering HTML report")
    html = render_template("report/report.html", **ctx)
    (rdir / "report.html").write_text(html, encoding="utf-8")

    for f in ("raw.pkl", "cleaned.pkl"):          # free disk; originals stay in uploads/
        (jdir / f).unlink(missing_ok=True)

    storage.update_entry(job_id, status="completed", step="done", progress=100,
                         progress_msg="Done", error=None, has_profile=has_profile,
                         has_final=has_final, final_rows=int(final.shape[0]),
                         final_cols=int(final.shape[1]),
                         finished_date=datetime.now().isoformat(timespec="seconds"))
