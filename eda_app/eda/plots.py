"""Matplotlib helpers that return base64 PNG strings.

Uses the object-oriented Figure API (not pyplot) so it is safe to call from
request threads and background report threads at the same time.
"""
import base64
import io
import logging

import matplotlib
import matplotlib.lines
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

logging.getLogger("matplotlib.category").setLevel(logging.WARNING)

BLUE, ORANGE, GREY, RED = "#4c78a8", "#f58518", "#9aa0a6", "#e45756"
CMAP_CAT = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2",
            "#ff9da6", "#9d755d", "#bab0ac", "#eeca3b"]


def _b64(fig, dpi=90):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def _short(label, n=18):
    label = str(label)
    return label if len(label) <= n else label[: n - 1] + "…"


# ------------------------------------------------------------------ missing values
def missing_bar(mt: pd.DataFrame, top=30):
    t = mt[mt["missing"] > 0].head(top).iloc[::-1]
    if t.empty:
        return None
    fig = Figure(figsize=(7, max(1.6, 0.28 * len(t) + 0.8)))
    ax = fig.subplots()
    ax.barh([_short(c, 28) for c in t["column"]], t["missing_pct"], color=ORANGE)
    ax.set_xlabel("% missing")
    ax.set_xlim(0, 100)
    _style(ax)
    return _b64(fig)


# ------------------------------------------------------------------ univariate
def hist_box(s: pd.Series, name: str):
    x = s.dropna().astype(float)
    if x.empty:
        return None
    fig = Figure(figsize=(5.2, 3.6))
    ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    bins = int(np.clip(np.sqrt(len(x)), 10, 50))
    ax1.hist(x, bins=bins, color=BLUE, alpha=0.85, density=True)
    if x.nunique() > 3:
        try:
            from scipy.stats import gaussian_kde
            xs = np.linspace(x.min(), x.max(), 200)
            samp = x.sample(min(len(x), 5000), random_state=0)
            ax1.plot(xs, gaussian_kde(samp)(xs), color=ORANGE, lw=1.6)
        except Exception:
            pass
    ax1.set_ylabel("density", fontsize=8)
    ax1.set_title(_short(name, 40), fontsize=9)
    ax2.boxplot(x, vert=False, widths=0.6, flierprops=dict(markersize=2, markerfacecolor=RED,
                                                            markeredgecolor=RED))
    ax2.set_yticks([])
    for a in (ax1, ax2):
        _style(a)
    fig.tight_layout()
    return _b64(fig)


def category_bar(counts: pd.Series, name: str, top=15):
    c = counts.head(top).iloc[::-1]
    if c.empty:
        return None
    fig = Figure(figsize=(5.2, max(1.8, 0.28 * len(c) + 1)))
    ax = fig.subplots()
    ax.barh([_short(i, 24) for i in c.index], c.values, color=BLUE)
    ax.set_title(_short(name, 40), fontsize=9)
    _style(ax)
    fig.tight_layout()
    return _b64(fig)


# ------------------------------------------------------------------ bivariate
def scatter_fit(x, y, xn, yn, max_points):
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) > max_points:
        d = d.sample(max_points, random_state=0)
    fig = Figure(figsize=(5.4, 4))
    ax = fig.subplots()
    ax.scatter(d.iloc[:, 0], d.iloc[:, 1], s=8, alpha=0.4, color=BLUE, edgecolors="none")
    try:
        if d.iloc[:, 0].nunique() > 1:
            m, b = np.polyfit(d.iloc[:, 0], d.iloc[:, 1], 1)
            xs = np.array([d.iloc[:, 0].min(), d.iloc[:, 0].max()])
            ax.plot(xs, m * xs + b, color=ORANGE, lw=2)
    except Exception:
        pass
    ax.set_xlabel(_short(xn, 40), fontsize=9)
    ax.set_ylabel(_short(yn, 40), fontsize=9)
    _style(ax)
    fig.tight_layout()
    return _b64(fig)


def box_by_category(groups: dict, num_name, cat_name):
    labels = list(groups.keys())
    fig = Figure(figsize=(max(5, 0.6 * len(labels) + 3), 4))
    ax = fig.subplots()
    ax.boxplot([groups[k] for k in labels], tick_labels=[_short(k, 14) for k in labels],
               patch_artist=True, boxprops=dict(facecolor="#c6dbef"),
               medianprops=dict(color=ORANGE, lw=2),
               flierprops=dict(markersize=2, markerfacecolor=RED, markeredgecolor=RED))
    ax.set_ylabel(_short(num_name, 40), fontsize=9)
    ax.set_xlabel(_short(cat_name, 40), fontsize=9)
    ax.tick_params(axis="x", rotation=40)
    _style(ax)
    fig.tight_layout()
    return _b64(fig)


def heatmap(matrix: pd.DataFrame, title="", cmap="Blues", vmin=None, vmax=None,
            annotate=True, fmt="{:.0f}", center_zero=False, figsize=None):
    r, c = matrix.shape
    fig = Figure(figsize=figsize or (max(4.5, 0.65 * c + 2.5), max(3.5, 0.55 * r + 1.5)))
    ax = fig.subplots()
    im = ax.imshow(matrix.values.astype(float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(c))
    ax.set_yticks(range(r))
    ax.set_xticklabels([_short(i, 16) for i in matrix.columns], rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels([_short(i, 16) for i in matrix.index], fontsize=8)
    if annotate:
        vals = matrix.values.astype(float)
        thr = np.nanmax(np.abs(vals)) * 0.6 if vals.size else 0
        for i in range(r):
            for j in range(c):
                v = vals[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > thr else "black")
    if title:
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _b64(fig)


def corr_heatmap(corr: pd.DataFrame, title=""):
    n = len(corr)
    size = max(4.5, min(14, 0.5 * n + 2.5))
    return heatmap(corr, title=title, cmap="RdBu_r", vmin=-1, vmax=1,
                   annotate=n <= 15, fmt="{:.2f}", figsize=(size, size * 0.85))


# ------------------------------------------------------------------ multivariate
def pair_plot(df: pd.DataFrame, cols, hue=None):
    k = len(cols)
    fig = Figure(figsize=(2.3 * k + 0.6, 2.3 * k + 0.6))
    axes = fig.subplots(k, k, squeeze=False)
    if hue is not None:
        levels = list(df[hue].value_counts().head(8).index)
        colors = {lv: CMAP_CAT[i % len(CMAP_CAT)] for i, lv in enumerate(levels)}
    for i, yc in enumerate(cols):
        for j, xc in enumerate(cols):
            ax = axes[i][j]
            _style(ax)
            ax.tick_params(labelsize=6)
            if i == j:
                if hue is None:
                    ax.hist(df[xc].dropna(), bins=20, color=BLUE, alpha=0.85)
                else:
                    for lv in levels:
                        ax.hist(df.loc[df[hue] == lv, xc].dropna(), bins=20, alpha=0.5,
                                color=colors[lv], label=_short(lv, 12))
            else:
                if hue is None:
                    ax.scatter(df[xc], df[yc], s=4, alpha=0.4, color=BLUE, edgecolors="none")
                else:
                    for lv in levels:
                        m = df[hue] == lv
                        ax.scatter(df.loc[m, xc], df.loc[m, yc], s=4, alpha=0.5,
                                   color=colors[lv], edgecolors="none")
            if i == k - 1:
                ax.set_xlabel(_short(xc, 16), fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(_short(yc, 16), fontsize=8)
            else:
                ax.set_yticklabels([])
    if hue is not None:
        handles = [matplotlib.lines.Line2D([0], [0], marker="o", ls="", color=colors[lv],
                                           label=_short(lv, 16)) for lv in levels]
        fig.legend(handles=handles, title=_short(hue, 20), loc="center left",
                   bbox_to_anchor=(1.0, 0.5), fontsize=8, title_fontsize=8)
    fig.tight_layout()
    return _b64(fig, dpi=80)


# ------------------------------------------------------------------ outliers
def box_grid(df: pd.DataFrame, cols, max_cols=12):
    cols = list(cols)[:max_cols]
    if not cols:
        return None
    ncol = min(4, len(cols))
    nrow = int(np.ceil(len(cols) / ncol))
    fig = Figure(figsize=(3 * ncol, 2.6 * nrow))
    axes = fig.subplots(nrow, ncol, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, c in zip(axes.ravel(), cols):
        ax.axis("on")
        ax.boxplot(df[c].dropna(), widths=0.5,
                   flierprops=dict(markersize=2, markerfacecolor=RED, markeredgecolor=RED))
        ax.set_title(_short(c, 22), fontsize=8)
        ax.set_xticks([])
        _style(ax)
    fig.tight_layout()
    return _b64(fig)


def before_after_box(before: pd.Series, after: pd.Series, name):
    fig = Figure(figsize=(4.4, 2.8))
    ax = fig.subplots()
    ax.boxplot([before.dropna(), after.dropna()], vert=False, tick_labels=["before", "after"],
               flierprops=dict(markersize=2, markerfacecolor=RED, markeredgecolor=RED))
    ax.set_title(_short(name, 40), fontsize=9)
    _style(ax)
    fig.tight_layout()
    return _b64(fig)
