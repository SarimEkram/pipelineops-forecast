from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
import uuid

import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Inside Docker, ./storage on your laptop is mounted to /data in the container.
DATASETS_DIR = Path("/data/datasets")
MODELS_DIR = Path("/data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainResult:
    model_id: str
    rows_used: int
    train_rows: int
    test_rows: int
    mae: float
    rmse: float
    model_path: str
    feature_cols: list[str]


def _build_features(df: pd.DataFrame, timestamp_col: str, target_col: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Turns a time-series into a supervised ML dataset.

    We predict the NEXT hour flow_rate using:
    - time features: hour of day, day of week
    - lag features: previous hour, previous day
    - rolling mean: 24-hour moving average (shifted so we don't leak the future)
    """
    df = df.copy()

    # time based features
    df["hour"] = df[timestamp_col].dt.hour
    df["day_of_week"] = df[timestamp_col].dt.dayofweek

    # lag features (past values)
    df["lag_1"] = df[target_col].shift(1)
    df["lag_24"] = df[target_col].shift(24)

    # rolling mean (use past 24 hours, then shift by 1 so it uses only past info)
    df["roll_mean_24"] = df[target_col].rolling(window=24, min_periods=24).mean().shift(1)

    # target we want to predict = next timestep value
    df["y"] = df[target_col].shift(-1)

    feature_cols = ["hour", "day_of_week", "lag_1", "lag_24", "roll_mean_24"]
    return df, feature_cols



