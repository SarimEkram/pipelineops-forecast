# backend/app/time_integrity.py
from __future__ import annotations

from typing import Any
import pandas as pd
import numpy as np


def _guess_interval_label(minutes: float) -> str:
    if np.isnan(minutes):
        return "unknown"
    if abs(minutes - 60) <= 2:
        return "hourly"
    if abs(minutes - 30) <= 1:
        return "30min"
    if abs(minutes - 15) <= 1:
        return "15min"
    if abs(minutes - 5) <= 0.5:
        return "5min"
    return f"~{minutes:.1f}min"


def analyze_time_integrity(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    expected_freq: str = "h",
    tolerance_minutes: int = 2,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "expected_freq": expected_freq,
        "tolerance_minutes": tolerance_minutes,
        "duplicates": 0,
        "interval_median_minutes": None,
        "interval_mode_minutes": None,
        "interval_label": "unknown",
        "hourly_match_rate": None,
        "is_hourly": False,
        "missing_hours": None,
        "missing_rate": None,
        "max_gap_hours": None,
        "expected_count": None,
        "observed_unique_count": None,
    }

    if timestamp_col not in df.columns:
        return out

    ts = pd.to_datetime(df[timestamp_col], errors="coerce").dropna()
    if len(ts) < 2:
        out["observed_unique_count"] = int(ts.nunique())
        return out

    # If timezone-aware, convert to naive
    try:
        ts = ts.dt.tz_convert(None)
    except Exception:
        pass

    ts = ts.sort_values()
    out["duplicates"] = int(ts.duplicated().sum())

    ts_unique = ts.drop_duplicates()
    out["observed_unique_count"] = int(len(ts_unique))
    if len(ts_unique) < 2:
        return out

    diffs = ts_unique.diff().dropna()
    if diffs.empty:
        return out

    median_minutes = float(diffs.median().total_seconds() / 60.0)
    out["interval_median_minutes"] = median_minutes

    vc = diffs.value_counts()
    mode_minutes = float(vc.index[0].total_seconds() / 60.0) if len(vc) else median_minutes
    out["interval_mode_minutes"] = mode_minutes

    out["interval_label"] = _guess_interval_label(median_minutes)

    minutes = diffs.dt.total_seconds() / 60.0
    hourly_mask = (minutes - 60.0).abs() <= tolerance_minutes
    match_rate = float(hourly_mask.mean()) if len(minutes) else 0.0
    out["hourly_match_rate"] = match_rate

    out["is_hourly"] = (abs(median_minutes - 60.0) <= tolerance_minutes) and (match_rate >= 0.80)

    out["max_gap_hours"] = float(diffs.max().total_seconds() / 3600.0)

    if out["is_hourly"]:
        start = ts_unique.iloc[0]
        end = ts_unique.iloc[-1]
        expected = pd.date_range(start=start, end=end, freq=expected_freq.lower())
        expected_count = int(len(expected))
        out["expected_count"] = expected_count

        missing = expected.difference(pd.DatetimeIndex(ts_unique))
        missing_hours = int(len(missing))
        out["missing_hours"] = missing_hours
        out["missing_rate"] = float(missing_hours / expected_count) if expected_count > 0 else None

    return out


def raise_if_not_hourly(integrity: dict[str, Any], context: str = "operation") -> None:
    if not integrity.get("is_hourly", False):
        label = integrity.get("interval_label", "unknown")

        med = integrity.get("interval_median_minutes")
        if isinstance(med, (int, float)):
            raise ValueError(f"{context}: expected hourly data, got {label} (median interval ~{med:.1f} minutes).")
        raise ValueError(f"{context}: expected hourly data, got {label}.")

    missing = integrity.get("missing_hours")
    if isinstance(missing, int) and missing > 0:
        rate = integrity.get("missing_rate")
        pct = f"{(rate * 100):.1f}%" if isinstance(rate, (int, float)) else "unknown"
        raise ValueError(
            f"{context}: dataset has missing hours ({missing} missing, missing_rate={pct}). "
            f"Fix gaps (resample/fill) before training/forecasting."
        )
