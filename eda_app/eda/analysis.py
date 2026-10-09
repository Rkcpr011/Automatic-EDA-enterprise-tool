"""Steps 14-20: univariate, bivariate, multivariate, correlation, outliers, feature profiling."""
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import config
from eda import plots
from eda.dataset_info import classify_columns


def plot_sample(df):
    return df if len(df) <= config.PLOT_SAMPLE_ROWS else df.sample(config.PLOT_SAMPLE_ROWS, random_state=42)


# ================================================================== Step 14: univariate
def univariate_numeric_stats(df, cols) -> pd.DataFrame:
    rows = {}
    n = len(df)
    for c in cols:
        s = df[c]
        nn = s.dropna().astype(float)
        if nn.empty:
            rows[c] = {"count": 0, "missing": n, "missing_%": 100.0}
            continue
        q1, med, q3 = nn.quantile([0.25, 0.5, 0.75])
        mode = nn.mode()
        mean, std = nn.mean(), nn.std()
        rows[c] = {
            "count": int(nn.count()), "missing": int(n - nn.count()),
            "missing_%": round(100 * (n - nn.count()) / n, 2), "unique": int(nn.nunique()),
            "mean": mean, "std": std, "variance": nn.var(), "min": nn.min(), "Q1": q1,
            "median": med, "Q3": q3, "max": nn.max(), "range": nn.max() - nn.min(),
            "IQR": q3 - q1, "mode": mode.iloc[0] if len(mode) else np.nan,
            "skew": nn.skew(), "kurtosis": nn.kurt(),
            "CV": (std / mean) if (mean and not np.isnan(mean)) else np.nan,
            "zeros": int((nn == 0).sum()), "negatives": int((nn < 0).sum()),
        }
    return pd.DataFrame(rows).T


def univariate_numeric_plots(df, cols):
    d = plot_sample(df)
    return {c: plots.hist_box(d[c], c) for c in cols[: config.MAX_UNIVARIATE_PLOTS]}


def univariate_categorical(df, cols):
    n = len(df)
    summary, details = {}, {}
    for c in cols:
        s = df[c]
        vc = s.value_counts(dropna=True)
        cnt = int(s.notna().sum())
        p = vc / cnt if cnt else vc
        entropy = float(-(p * np.log2(p)).sum()) if cnt else np.nan
        summary[c] = {
            "count": cnt, "missing": n - cnt, "missing_%": round(100 * (n - cnt) / n, 2),
            "unique": int(vc.shape[0]),
            "top": vc.index[0] if len(vc) else None,
            "top_freq": int(vc.iloc[0]) if len(vc) else 0,
            "top_%": round(100 * vc.iloc[0] / cnt, 2) if cnt else np.nan,
            "entropy (bits)": round(entropy, 3),
        }
        top = vc.head(10)
        details[c] = {
            "table": pd.DataFrame({"count": top, "%": (100 * top / cnt).round(2)}) if cnt else None,
            "plot": plots.category_bar(vc, c) if len(details) < config.MAX_UNIVARIATE_PLOTS else None,
        }
    return pd.DataFrame(summary).T, details


def univariate_datetime(df, cols) -> pd.DataFrame:
    rows = {}
    for c in cols:
        s = df[c].dropna()
        rows[c] = {"count": len(s), "missing": int(df[c].isna().sum()),
                   "min": s.min() if len(s) else None, "max": s.max() if len(s) else None,
                   "span (days)": (s.max() - s.min()).days if len(s) else None,
                   "unique": int(s.nunique())}
    return pd.DataFrame(rows).T


# ================================================================== Steps 15-16: bivariate
def bivariate_kind(kx, ky):
    if kx == "numeric" and ky == "numeric":
        return "num_num"
    if {kx, ky} == {"numeric", "categorical"}:
        return "num_cat"
    if kx == "categorical" and ky == "categorical":
        return "cat_cat"
    return "unsupported"


def _cramers_v(ct: pd.DataFrame):
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan, np.nan, np.nan
    chi2, p, _, _ = stats.chi2_contingency(ct, correction=False)
    n = ct.values.sum()
    k = min(ct.shape) - 1
    v = np.sqrt(chi2 / (n * k)) if n and k else np.nan
    return chi2, p, v


def bivariate_pair(df, x, y):
    cls = classify_columns(df)
    kind_of = {c: k for k, cs in cls.items() for c in cs}
    kind = bivariate_kind(kind_of[x], kind_of[y])
    res = {"x": x, "y": y, "kind": kind, "stats": None, "table": None, "plot": None, "note": None}

    if kind == "num_num":
        d = df[[x, y]].dropna().astype(float)
        res["note"] = f"{len(d):,} complete pairs."
        if len(d) < 3 or d[x].nunique() < 2 or d[y].nunique() < 2:
            res["note"] += " Not enough variation to compute correlations."
            return res
        pr, pp = stats.pearsonr(d[x], d[y])
        sr, sp = stats.spearmanr(d[x], d[y])
        slope, intercept = np.polyfit(d[x], d[y], 1)
        res["stats"] = {"Pearson r": pr, "Pearson p-value": pp, "Spearman ρ": sr,
                        "Spearman p-value": sp, "R² (linear)": pr ** 2,
                        "Slope": slope, "Intercept": intercept}
        res["plot"] = plots.scatter_fit(d[x], d[y], x, y, config.SCATTER_MAX_POINTS)

    elif kind == "num_cat":
        num, cat = (x, y) if kind_of[x] == "numeric" else (y, x)
        d = df[[num, cat]].dropna()
        top = d[cat].astype(str).value_counts().head(10).index
        d = d.assign(**{cat: d[cat].astype(str)})
        d = d[d[cat].isin(top)]
        g = d.groupby(cat)[num].agg(["count", "mean", "median", "std", "min", "max"])
        g = g.sort_values("count", ascending=False)
        res["table"] = g
        groups = {k: v[num].astype(float).values for k, v in d.groupby(cat) if len(v) >= 1}
        valid = [v for v in groups.values() if len(v) >= 2]
        stat = {}
        if len(valid) >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f, p = stats.f_oneway(*valid)
                h, kp = stats.kruskal(*valid)
            grand = d[num].astype(float)
            ssb = sum(len(v) * (v.mean() - grand.mean()) ** 2 for v in groups.values())
            sst = ((grand - grand.mean()) ** 2).sum()
            stat = {"ANOVA F": f, "ANOVA p-value": p, "Kruskal-Wallis H": h,
                    "Kruskal p-value": kp, "η² (effect size)": ssb / sst if sst else np.nan}
        res["stats"] = stat or None
        res["note"] = (f"Showing top {len(groups)} categories of '{cat}' by frequency."
                       if cat_levels(df[cat]) > 10 else None)
        order = list(g.index)
        res["plot"] = plots.box_by_category({k: groups[k] for k in order if k in groups}, num, cat)

    elif kind == "cat_cat":
        d = df[[x, y]].dropna().astype(str)
        tx, ty = d[x].value_counts().head(15).index, d[y].value_counts().head(15).index
        d = d[d[x].isin(tx) & d[y].isin(ty)]
        ct = pd.crosstab(d[x], d[y])
        res["table"] = ct if ct.shape[0] <= 12 and ct.shape[1] <= 12 else None
        chi2, p, v = _cramers_v(ct)
        res["stats"] = {"Chi-square": chi2, "p-value": p, "Cramér's V": v}
        if ct.size:
            norm = (ct.div(ct.sum(axis=1), axis=0) * 100)
            res["plot"] = plots.heatmap(norm, title=f"% of '{x}' rows in each '{y}' level",
                                        annotate=ct.shape[0] <= 12 and ct.shape[1] <= 12,
                                        fmt="{:.0f}")
        res["note"] = "Top 15 levels per variable are shown." if (
            cat_levels(df[x]) > 15 or cat_levels(df[y]) > 15) else None
    else:
        res["note"] = "Datetime / unsupported column types are not compared in bivariate analysis."
    return res


def cat_levels(s):
    return s.nunique(dropna=True)


def suggest_pairs(df, k=5):
    """Top-k most correlated numeric pairs, used as one-click suggestions."""
    num = classify_columns(df)["numeric"]
    if len(num) < 2:
        return []
    d = df[num[:60]]
    if len(d) > 50_000:
        d = d.sample(50_000, random_state=0)
    corr = d.corr(numeric_only=True).abs()
    pairs = []
    for a, b in itertools.combinations(corr.columns, 2):
        v = corr.loc[a, b]
        if not np.isnan(v):
            pairs.append((v, a, b))
    pairs.sort(reverse=True)
    return [{"x": a, "y": b, "r": round(float(v), 2)} for v, a, b in pairs[:k]]


# ================================================================== Step 18: correlation
def correlation_analysis(df, threshold=0.8):
    cls = classify_columns(df)
    num = [c for c in cls["numeric"] if df[c].nunique() > 1]
    out = {"threshold": threshold, "numeric_cols": num}
    if len(num) >= 2:
        d = plot_sample(df)[num] if len(df) > 200_000 else df[num]
        pear = d.corr(method="pearson")
        spear = d.corr(method="spearman")
        out["pearson"], out["spearman"] = pear, spear
        shown = list(pear.abs().mean().sort_values(ascending=False).index[:40])
        out["heatmap_note"] = (f"Heatmaps show the {len(shown)} columns with the highest average "
                               f"|correlation|." if len(num) > 40 else None)
        out["pearson_plot"] = plots.corr_heatmap(pear.loc[shown, shown], "Pearson correlation")
        out["spearman_plot"] = plots.corr_heatmap(spear.loc[shown, shown], "Spearman correlation")
        pairs = []
        for a, b in itertools.combinations(num, 2):
            pairs.append({"feature 1": a, "feature 2": b, "pearson": pear.loc[a, b],
                          "spearman": spear.loc[a, b]})
        pdf = pd.DataFrame(pairs)
        pdf["|pearson|"] = pdf["pearson"].abs()
        pdf = pdf.sort_values("|pearson|", ascending=False).drop(columns="|pearson|")
        out["top_pairs"] = pdf.head(15).reset_index(drop=True)
        out["high_pairs"] = pdf[pdf["pearson"].abs() >= threshold].reset_index(drop=True)
        out["pair_lookup"] = {(r["feature 1"], r["feature 2"]): r["pearson"] for _, r in pdf.iterrows()
                              if abs(r["pearson"]) >= threshold}
    else:
        out["pair_lookup"] = {}

    # categorical association (Cramér's V)
    cats = [c for c in cls["categorical"] if 2 <= df[c].nunique() <= 30][:15]
    if len(cats) >= 2:
        d = plot_sample(df)[cats].astype(str).where(plot_sample(df)[cats].notna())
        m = pd.DataFrame(np.eye(len(cats)), index=cats, columns=cats)
        for a, b in itertools.combinations(cats, 2):
            sub = d[[a, b]].dropna()
            ct = pd.crosstab(sub[a], sub[b])
            v = _cramers_v(ct)[2]
            m.loc[a, b] = m.loc[b, a] = v
        out["cramers"] = m
        out["cramers_plot"] = plots.heatmap(m, "Categorical association (Cramér's V)", cmap="Purples",
                                            vmin=0, vmax=1, annotate=len(cats) <= 12, fmt="{:.2f}")
    return out


# ================================================================== Step 17: pair plot
def make_pair_plot(df, cols, hue):
    cols = [c for c in cols if c in df.columns][: config.MAX_PAIRPLOT_COLS]
    if len(cols) < 2:
        return None
    use = cols + ([hue] if hue and hue in df.columns else [])
    d = df[use].dropna()
    if len(d) > config.PAIRPLOT_MAX_POINTS:
        d = d.sample(config.PAIRPLOT_MAX_POINTS, random_state=0)
    d = d.copy()
    if hue and hue in d.columns:
        d[hue] = d[hue].astype(str)
    return plots.pair_plot(d, cols, hue if hue in d.columns else None)


# ================================================================== Step 19: outliers
def _bounds(s: pd.Series, method, k, z):
    nn = s.dropna().astype(float)
    if method == "zscore":
        mu, sd = nn.mean(), nn.std()
        return (mu - z * sd, mu + z * sd) if sd and not np.isnan(sd) else (-np.inf, np.inf)
    q1, q3 = nn.quantile(0.25), nn.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def outlier_summary(df, cols, k=1.5, z=3.0) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in cols:
        s = df[c].astype(float)
        nn = s.dropna()
        if nn.empty or nn.nunique() < 2:
            continue
        lo, hi = _bounds(s, "iqr", k, z)
        zlo, zhi = _bounds(s, "zscore", k, z)
        n_iqr = int(((s < lo) | (s > hi)).sum())
        n_z = int(((s < zlo) | (s > zhi)).sum())
        rows.append({"column": c, "min": nn.min(), "max": nn.max(),
                     "IQR lower": lo, "IQR upper": hi, "IQR outliers": n_iqr,
                     "IQR %": round(100 * n_iqr / n, 2), "Z outliers": n_z,
                     "Z %": round(100 * n_z / n, 2)})
    return pd.DataFrame(rows)


def treat_outliers(df, cols, method, k, z, action):
    """action: flag | remove | cap. Returns (df_out, per_column_details, rows_removed)."""
    details, mask_any = [], pd.Series(False, index=df.index)
    out = df.copy() if action == "cap" else df
    for c in cols:
        s = df[c].astype(float)
        lo, hi = _bounds(s, method, k, z)
        mask = ((s < lo) | (s > hi)).fillna(False)
        mask_any |= mask
        before = s.dropna()
        entry = {"column": c, "outliers": int(mask.sum()), "lower": lo, "upper": hi,
                 "before_mean": before.mean(), "before_std": before.std(),
                 "before_min": before.min(), "before_max": before.max()}
        if action == "cap":
            out[c] = s.clip(lower=lo, upper=hi)
        details.append(entry)
    removed = 0
    if action == "remove":
        out = df[~mask_any].reset_index(drop=True)
        removed = int(mask_any.sum())
    for e in details:
        after = out[e["column"]].astype(float).dropna() if e["column"] in out else pd.Series(dtype=float)
        e.update({"after_mean": after.mean(), "after_std": after.std(),
                  "after_min": after.min() if len(after) else np.nan,
                  "after_max": after.max() if len(after) else np.nan})
    return out, details, removed


# ================================================================== Step 20: feature profiling
def feature_profile(df, outlier_pct: dict, high_pairs: dict) -> pd.DataFrame:
    cls = classify_columns(df)
    kind_of = {c: k for k, cs in cls.items() for c in cs}
    n = len(df)
    partner = {}
    for (a, b), r in high_pairs.items():
        partner.setdefault(a, (b, r))
        partner.setdefault(b, (a, r))
    rows = []
    for c in df.columns:
        s = df[c]
        kind = kind_of.get(c, "other")
        nn = s.dropna()
        nun = int(nn.nunique())
        miss = 100 * (n - len(nn)) / n if n else 0
        recs, role, key = [], "", ""
        if nun <= 1:
            role = "Constant"
            recs.append("No information - drop")
        elif kind == "numeric":
            is_int = bool(np.all(np.mod(nn.to_numpy(dtype=float), 1) == 0))
            if is_int and nun / len(nn) > 0.98 and len(nn) > 50:
                role = "Possible identifier"
                recs.append("Looks like an ID - exclude from modelling")
            elif is_int and nun <= 10:
                role = "Discrete numeric"
                recs.append("Consider treating as categorical")
            else:
                role = "Continuous numeric"
            skew = nn.skew()
            key = f"mean={nn.mean():.4g}, skew={skew:.2f}"
            if abs(skew) > 1:
                recs.append("Highly skewed - log / Box-Cox transform" if nn.min() > 0
                            else "Highly skewed - Yeo-Johnson transform")
            if outlier_pct.get(c, 0) > 5:
                recs.append(f"{outlier_pct[c]:.1f}% outliers - consider capping")
            if (nn == 0).mean() > 0.5:
                recs.append("Zero-inflated (>50% zeros)")
        elif kind == "categorical":
            ratio = nun / len(nn) if len(nn) else 0
            if ratio > 0.9 and len(nn) > 50:
                role = "Possible identifier / free text"
                recs.append("Near-unique values - drop or engineer features")
            elif nun <= 10:
                role = "Low-cardinality categorical"
                recs.append("One-hot encode")
            elif nun <= 50:
                role = "Medium-cardinality categorical"
                recs.append("Target / ordinal encoding")
            else:
                role = "High-cardinality categorical"
                recs.append("Hash / target encoding, or group rare levels")
            top = nn.value_counts(normalize=True)
            key = f"top='{str(top.index[0])[:20]}' ({100 * top.iloc[0]:.1f}%)" if len(top) else ""
            if len(top) and top.iloc[0] > 0.95:
                recs.append("Dominated by one level (>95%)")
        elif kind == "datetime":
            role = "Datetime"
            key = f"{nn.min():%Y-%m-%d} → {nn.max():%Y-%m-%d}" if len(nn) else ""
            recs.append("Extract year / month / weekday features")
        else:
            role = "Other"
        if miss > 40:
            recs.append("Very high missingness - consider dropping")
        elif miss > 5:
            recs.append("Impute missing values")
        if c in partner:
            recs.append(f"Highly correlated with '{partner[c][0]}' (r={partner[c][1]:.2f})")
        rows.append({"feature": c, "kind": kind, "role": role, "dtype": str(s.dtype),
                     "missing %": round(miss, 2), "unique": nun,
                     "unique %": round(100 * nun / n, 2) if n else 0, "key stat": key,
                     "outlier % (IQR)": round(outlier_pct.get(c, np.nan), 2) if c in outlier_pct else np.nan,
                     "recommendations": "; ".join(recs) if recs else "OK"})
    return pd.DataFrame(rows)
