"""Steps 4-9: summary, column classification, dtype verification, missing values."""
import re
import warnings

import numpy as np
import pandas as pd
from pandas.api import types as pdt

_BOOL_WORDS = {"true", "false", "yes", "no", "y", "n", "t", "f", "0", "1"}
_DATE_PATTERN = re.compile(
    r"(?:\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4})|(?:\d{1,2}:\d{2})|"
    r"(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? \d)", re.I)


# ---------------------------------------------------------------- classification
def col_kind(s: pd.Series) -> str:
    if pdt.is_bool_dtype(s):
        return "categorical"
    if pdt.is_numeric_dtype(s):
        return "numeric"
    if pdt.is_datetime64_any_dtype(s):
        return "datetime"
    if isinstance(s.dtype, pd.CategoricalDtype) or pdt.is_object_dtype(s) or pdt.is_string_dtype(s):
        return "categorical"
    return "other"


def classify_columns(df: pd.DataFrame) -> dict:
    out = {"numeric": [], "categorical": [], "datetime": [], "other": []}
    for c in df.columns:
        out[col_kind(df[c])].append(c)
    return out


# ---------------------------------------------------------------- summary
def dataset_summary(df: pd.DataFrame) -> dict:
    cls = classify_columns(df)
    total_cells = df.shape[0] * df.shape[1]
    missing = int(df.isna().sum().sum())
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2),
        "duplicates": int(df.duplicated().sum()),
        "missing_cells": missing,
        "missing_pct": round(100 * missing / total_cells, 2) if total_cells else 0.0,
        "n_numeric": len(cls["numeric"]),
        "n_categorical": len(cls["categorical"]),
        "n_datetime": len(cls["datetime"]),
    }


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    m = df.isna().sum()
    t = pd.DataFrame({"column": m.index, "missing": m.values,
                      "missing_pct": (100 * m.values / n) if n else 0.0})
    return t.sort_values("missing_pct", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- dtype verification (Step 7)
def _sample_non_null(s: pd.Series, n=5000) -> pd.Series:
    s = s.dropna()
    if len(s) > n:
        s = s.sample(n, random_state=0)
    return s


def _numeric_like(strs: pd.Series) -> bool:
    cleaned = strs.str.replace(r"[,\s$€£%]", "", regex=True)
    cleaned = cleaned[cleaned != ""]
    if cleaned.empty:
        return False
    if cleaned.str.match(r"^-?0\d+$").any():      # leading zeros -> codes / zip codes, not numbers
        return False
    ok = pd.to_numeric(cleaned, errors="coerce").notna().mean()
    return ok >= 0.95 and len(cleaned) / len(strs) >= 0.9


def _datetime_like(strs: pd.Series) -> bool:
    if strs.str.contains(_DATE_PATTERN).mean() < 0.9:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(strs, errors="coerce")
    return parsed.notna().mean() >= 0.9


def detect_type_issues(df: pd.DataFrame) -> dict:
    """Return {column: {'convert': target, 'reason': text, 'preselect': bool}}."""
    issues = {}
    for c in df.columns:
        s = df[c]
        if pdt.is_object_dtype(s) or pdt.is_string_dtype(s):
            sample = _sample_non_null(s)
            if sample.empty:
                continue
            strs = sample.astype(str).str.strip()
            low = set(strs.str.lower().unique())
            if low <= _BOOL_WORDS and len(low) <= 2 and not low <= {"0", "1"}:
                issues[c] = {"convert": "boolean", "preselect": True,
                             "reason": "Only two yes/no-style values"}
            elif _numeric_like(strs):
                issues[c] = {"convert": "numeric", "preselect": True,
                             "reason": "Numbers stored as text"}
            elif _datetime_like(strs):
                issues[c] = {"convert": "datetime", "preselect": True,
                             "reason": "Dates stored as text"}
        elif pdt.is_numeric_dtype(s) and not pdt.is_bool_dtype(s):
            nn = s.dropna()
            if nn.empty:
                continue
            nun = nn.nunique()
            if nun == 2 and set(nn.unique()) <= {0, 1}:
                issues[c] = {"convert": "categorical", "preselect": False,
                             "reason": "Binary 0/1 flag"}
            elif nun <= 10 and len(nn) > 50 and np.all(np.mod(nn.to_numpy(dtype=float), 1) == 0) \
                    and nun / len(nn) < 0.05:
                issues[c] = {"convert": "categorical", "preselect": False,
                             "reason": f"Only {nun} distinct integer values - may be codes"}
    return issues


def column_flags(s: pd.Series) -> list:
    """Short warning badges shown next to a column."""
    flags = []
    nn = s.dropna()
    n = len(nn)
    if n == 0:
        return ["empty"]
    nun = nn.nunique()
    if nun <= 1:
        flags.append("constant")
    elif n > 50 and nun / n > 0.98 and col_kind(s) != "datetime":
        if col_kind(s) == "categorical" or pdt.is_integer_dtype(s):
            flags.append("id-like")
    if col_kind(s) == "categorical" and nun > 50 and nun / n < 0.98:
        flags.append("high-cardinality")
    return flags


def sample_values(s: pd.Series, k=3) -> str:
    vals = s.dropna().unique()[:k]
    txt = ", ".join(str(v)[:25] for v in vals)
    return txt
