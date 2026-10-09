"""Steps 10-12: apply the user's cleaning decisions."""
import warnings

import pandas as pd

BOOL_MAP = {"true": True, "yes": True, "y": True, "t": True, "1": True,
            "false": False, "no": False, "n": False, "f": False, "0": False}

CONVERT_TARGETS = ("keep", "numeric", "categorical", "datetime", "boolean")


def _to_numeric(s):
    if pd.api.types.is_numeric_dtype(s):
        return s
    cleaned = s.astype("string").str.strip().str.replace(r"[,\s$€£%]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def _to_datetime(s):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(s, errors="coerce")


def _to_boolean(s):
    mapped = s.astype("string").str.strip().str.lower().map(BOOL_MAP)
    return mapped.map({True: "True", False: "False"}).astype("category")


def _to_categorical(s):
    if isinstance(s.dtype, pd.CategoricalDtype):
        return s
    return s.astype("object").where(s.notna(), None).map(
        lambda v: v if v is None or isinstance(v, str) else str(v)).astype("category")


def apply_cleaning(df, drop_cols, conversions, drop_duplicates=False):
    """Return (cleaned_df, log_lines, warnings_list)."""
    log, warn = [], []
    out = df

    if drop_cols:
        out = out.drop(columns=[c for c in drop_cols if c in out.columns])
        log.append(f"Dropped {len(drop_cols)} column(s): " + ", ".join(drop_cols))

    for col, target in conversions.items():
        if col not in out.columns or target in (None, "", "keep"):
            continue
        before_na = int(out[col].isna().sum())
        try:
            if target == "numeric":
                out[col] = _to_numeric(out[col])
            elif target == "datetime":
                out[col] = _to_datetime(out[col])
            elif target == "boolean":
                out[col] = _to_boolean(out[col])
            elif target == "categorical":
                out[col] = _to_categorical(out[col])
        except Exception as exc:
            warn.append(f"Could not convert '{col}' to {target}: {exc}")
            continue
        lost = int(out[col].isna().sum()) - before_na
        msg = f"Converted '{col}' to {target}"
        if lost > 0:
            msg += f" ({lost} value(s) could not be parsed and became missing)"
            warn.append(f"'{col}': {lost} value(s) became missing after converting to {target}.")
        log.append(msg)

    removed_dups = 0
    if drop_duplicates:
        before = len(out)
        out = out.drop_duplicates().reset_index(drop=True)
        removed_dups = before - len(out)
        log.append(f"Removed {removed_dups} duplicate row(s)")

    if not log:
        log.append("No cleaning actions were applied.")
    return out, log, warn, removed_dups
