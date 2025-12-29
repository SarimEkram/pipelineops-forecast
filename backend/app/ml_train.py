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


def train_ridge_model(
    dataset_id: str,
    timestamp_col: str = "timestamp",
    target_col: str = "flow_rate",
    test_size: float = 0.2,
    alpha: float = 1.0,
) -> TrainResult:
    """
    Loads a dataset from disk, builds features, trains a Ridge regression baseline,
    evaluates on a time-based split, and saves a model artifact to /data/models.
    """
    dataset_path = DATASETS_DIR / f"{dataset_id}.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # read and basic validation
    df = pd.read_csv(dataset_path)
    if timestamp_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"CSV must contain columns: {timestamp_col}, {target_col}")

    # parse timestamps, sort
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    # make sure target is numeric
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    # build ML features
    df_feat, feature_cols = _build_features(df, timestamp_col, target_col)

    # drop rows that don't have enough history for lags/rolling and also drop last row (y is NaN there)
    df_feat = df_feat.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)

    if len(df_feat) < 60:
        raise ValueError(
            f"Not enough usable rows after feature building. Need ~60+, got {len(df_feat)}. "
            f"(Tip: use at least a few days of hourly data.)"
        )

    # time-based split (no shuffling)
    split_idx = int(len(df_feat) * (1.0 - test_size))
    split_idx = max(1, min(split_idx, len(df_feat) - 1))

    train_df = df_feat.iloc[:split_idx]
    test_df = df_feat.iloc[split_idx:]

    X_train = train_df[feature_cols].astype(float)
    y_train = train_df["y"].astype(float)

    X_test = test_df[feature_cols].astype(float)
    y_test = test_df["y"].astype(float)

    # train baseline model
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)

    # evaluate
    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    # save model artifact
    model_id = uuid.uuid4().hex[:8]
    model_path = MODELS_DIR / f"{model_id}.joblib"

    artifact = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "timestamp_col": timestamp_col,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "model": model,
    }

    joblib.dump(artifact, model_path)

    return TrainResult(
        model_id=model_id,
        rows_used=int(len(df_feat)),
        train_rows=int(len(train_df)),
        test_rows=int(len(test_df)),
        mae=mae,
        rmse=rmse,
        model_path=str(model_path),
        feature_cols=feature_cols,
    )
