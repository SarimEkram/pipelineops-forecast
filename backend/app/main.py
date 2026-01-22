# --- FastAPI imports ---
# FastAPI = the web server framework
# UploadFile + File = how FastAPI receives an uploaded file (multipart/form-data)
# HTTPException = how we return clean errors (400/404/etc) instead of crashing
from fastapi import FastAPI, UploadFile, File, HTTPException

# pandas = easiest way to read + clean CSV data
import pandas as pd

# Path = safer/easier way to work with file paths than raw strings
from pathlib import Path

# uuid = generates unique IDs so each uploaded dataset gets its own ID
import uuid

# BytesIO = treats raw bytes as a file-like object (so pandas can read it)
from io import BytesIO

from pydantic import BaseModel, Field

from .ml_train import train_ridge_model, forecast_next_hours, load_model_artifact

import logging
import time
from fastapi import Request
from .config import DATASETS_DIR, MODELS_DIR, MAX_UPLOAD_MB, LOG_LEVEL
from .time_integrity import analyze_time_integrity, gate_from_integrity

# Create the FastAPI app (this is the server)
# title shows up in the docs UI at /docs
app = FastAPI(title="PipelineOps API")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        raise
    duration_ms = (time.time() - start) * 1000
    logger.info("%s %s -> %s (%.1fms)", request.method, request.url.path, response.status_code, duration_ms)
    return response


# In Docker, we mounted: ./storage  ->  /data
# That means anything we save under /data will appear in your repo's storage/ folder
# directories come from config.py (env-driven)
logger = logging.getLogger("pipelineops")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

# These are the minimum columns we need for forecasting later
# timestamp = time column, flow_rate = what we want to forecast
REQUIRED_COLS = {"timestamp", "flow_rate"}


# Health check endpoint (used by your UI to confirm backend is alive)
@app.get("/health")
def health():
    # Returns JSON: {"status": "ok"}
    return {"status": "ok"}


# Upload endpoint: user sends a CSV file and we store it
@app.post("/datasets/upload")
async def upload_dataset(
        # file: variable name (must match the form field name "file" when uploading)
        # UploadFile: FastAPI type for an uploaded file (gives you filename, content, etc.)
        # File(...): says it's required (if missing, FastAPI returns an error automatically)
        file: UploadFile = File(...)
):
    # 1) Simple validation: make sure user is uploading a CSV file
    # We check the filename extension (not perfect security, but good for MVP)
    if not file.filename.lower().endswith(".csv"):
        # Return HTTP 400 (bad request) with a clear message
        raise HTTPException(status_code=400, detail="please upload a .csv file")

    # 2) Read the entire uploaded file into memory as raw bytes
    # await is needed because UploadFile is async
    raw = await file.read()

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail=f"file too large (max {MAX_UPLOAD_MB} MB)")

    # 3) Try to parse CSV bytes into a pandas DataFrame
    try:
        # BytesIO(raw) makes the raw bytes act like a file object
        df = pd.read_csv(BytesIO(raw))
    except Exception:
        # If pandas fails to read the CSV, return a 400 error
        raise HTTPException(status_code=400, detail="could not read csv")

    # 4) Check that required columns exist
    # df.columns is the list of column headers in the CSV
    missing = REQUIRED_COLS - set(df.columns)

    # If anything required is missing, reject the upload
    if missing:
        # sorted() makes the message consistent and readable
        raise HTTPException(
            status_code=400,
            detail=f"missing columns: {sorted(list(missing))}"
        )

    # 5) Clean timestamp column
    # Convert timestamp strings into real datetime objects
    # errors="coerce" turns invalid timestamps into NaT (like null)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Drop rows where timestamp couldn't be parsed (NaT)
    df = df.dropna(subset=["timestamp"])

    # Sort the data by time so it's in correct order for forecasting
    df = df.sort_values("timestamp")

    # Make sure flow_rate is numeric (drop non-numeric rows)
    df["flow_rate"] = pd.to_numeric(df["flow_rate"], errors="coerce")
    df = df.dropna(subset=["flow_rate"])

    # If duplicate timestamps exist, aggregate flow_rate by mean (keeps time series sane)
    dup_count = int(df["timestamp"].duplicated().sum())
    if dup_count > 0:
        other_cols = [c for c in df.columns if c not in ["timestamp", "flow_rate"]]
        agg = {"flow_rate": "mean"}
        for c in other_cols:
            agg[c] = "first"
        df = df.groupby("timestamp", as_index=False).agg(agg).sort_values("timestamp")

    # 6) Basic sanity check: ensure the dataset isn't tiny
    if len(df) < 10:
        raise HTTPException(
            status_code=400,
            detail="dataset too small (need at least 10 valid rows)"
        )

    # 7) Create a dataset ID (short unique string)
    # uuid4 gives a random unique ID; [:8] makes it short and readable
    dataset_id = str(uuid.uuid4())[:8]

    # 8) Build the output path where the dataset will be saved
    # Example: /data/datasets/a3f9c2d1.csv
    out_path = DATASETS_DIR / f"{dataset_id}.csv"

    # 9) Save the cleaned dataset to disk (index=False avoids adding an extra index column)
    df.to_csv(out_path, index=False)

    integrity = analyze_time_integrity(df, timestamp_col="timestamp")
    gate = gate_from_integrity(integrity)

    warnings = []

    if dup_count > 0:
        warnings.append(
            f"found duplicate timestamps: {dup_count} (duplicates were aggregated on upload)"
        )

    if not integrity.get("is_hourly", False):
        med = integrity.get("interval_median_minutes")
        med_str = f"{med:.1f}" if isinstance(med, (int, float)) else "unknown"
        warnings.append(
            f"expected hourly data, got {integrity.get('interval_label')} "
            f"(median ~{med_str} minutes). training/forecasting will be blocked until hourly."
        )

    missing_hours = integrity.get("missing_hours")
    if isinstance(missing_hours, int) and missing_hours > 0:
        warnings.append(
            f"missing hours detected: {missing_hours} missing between first and last timestamp. "
            f"training/forecasting will be blocked until gaps are fixed."
        )

    # 10) Return metadata so the UI can display “upload successful”
    return {
        # ID the UI will store and use later for training/forecast
        "dataset_id": dataset_id,

        # How many rows after cleaning
        "rows": int(len(df)),

        # Earliest and latest timestamp in the uploaded dataset
        "min_ts": str(df["timestamp"].min()),
        "max_ts": str(df["timestamp"].max()),

        # Columns the backend detected in the CSV
        "columns": list(df.columns),

        # Helpful for debugging (shows where it saved inside the container)
        "saved_path": str(out_path),

        "integrity": integrity,
        "warnings": warnings,
        "gate": gate,

    }


# This endpoint is for the UI preview:
# - dataset_id comes from the URL path (example: /datasets/2ae51a31/sample)
# - rows is a query param (example: ?rows=200)
@app.get("/datasets/{dataset_id}/sample")
def dataset_sample(
        dataset_id: str,  # PATH PARAM: which dataset file to load
        rows: int = 200  # QUERY PARAM: how many rows to return (default 200)
):
    # Safety: don’t let someone request a million rows and freeze the server
    if rows < 1 or rows > 2000:
        raise HTTPException(status_code=400, detail="rows must be between 1 and 2000")

    # Build the exact file path we expect this dataset to live at
    # Example: /data/datasets/2ae51a31.csv
    path = DATASETS_DIR / f"{dataset_id}.csv"

    # If the file doesn’t exist, return 404 (dataset not found)
    if not path.exists():
        raise HTTPException(status_code=404, detail="dataset not found")

    # Read the CSV from disk
    df = pd.read_csv(path)

    # Convert timestamp to datetime (so sorting behaves correctly)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Drop bad timestamp rows (if any)
    df = df.dropna(subset=["timestamp"])

    # Sort by timestamp so preview is always in time order
    df = df.sort_values("timestamp")

    df["flow_rate"] = pd.to_numeric(df["flow_rate"], errors="coerce")
    df = df.dropna(subset=["flow_rate"])

    # Take first N rows requested
    preview = df.tail(rows).copy()

    # Convert timestamp back to string so JSON serialization is clean
    preview["timestamp"] = preview["timestamp"].astype(str)

    # Return a simple JSON payload the UI can consume easily
    return {
        "dataset_id": dataset_id,
        "rows_returned": int(len(preview)),
        "data": preview.to_dict(orient="records")  # list of {timestamp, flow_rate:}
    }


@app.get("/datasets/{dataset_id}/info")
def dataset_info(dataset_id: str):
    path = DATASETS_DIR / f"{dataset_id}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="dataset not found")

    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "flow_rate" not in df.columns:
        raise HTTPException(status_code=400, detail="dataset missing timestamp/flow_rate")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["flow_rate"] = pd.to_numeric(df["flow_rate"], errors="coerce")
    df = df.dropna(subset=["timestamp", "flow_rate"]).sort_values("timestamp")

    integrity = analyze_time_integrity(df, timestamp_col="timestamp")
    gate = gate_from_integrity(integrity)

    return {
        "dataset_id": dataset_id,
        "rows": int(len(df)),
        "min_ts": str(df["timestamp"].min()),
        "max_ts": str(df["timestamp"].max()),
        "integrity": integrity,
        "gate": gate,
    }


# ML Training Request Schema


# This defines the JSON shape the client must send to /models/train
class TrainModelRequest(BaseModel):
    # dataset_id = the id we got from /datasets/upload (example: "141a1070")
    dataset_id: str = Field(..., description="Dataset id returned from /datasets/upload")

    # test_size = how much of the newest data we keep for testing (time-based split)
    # ge/le = validation rules (prevents weird values like -2 or 0.99)
    test_size: float = Field(
        0.2,
        ge=0.05,
        le=0.5,
        description="Fraction of data reserved for testing (time-based split)",
    )

    # alpha = Ridge regularization strength
    # gt = must be greater than 0
    alpha: float = Field(1.0, gt=0.0, description="Ridge regularization strength")


# ML Training Endpoint


@app.post("/models/train")
def train_model(req: TrainModelRequest):
    """
    Trains a baseline ML model (Ridge regression) to forecast the next hour of flow_rate.

    Input:
    - dataset_id (which dataset to train on)
    - test_size (time-based split)
    - alpha (Ridge regularization)

    Output:
    - model_id + metrics (MAE/RMSE) + where the model was saved
    """
    try:
        # Run the actual training pipeline (feature creation + model fit + evaluation)
        result = train_ridge_model(
            dataset_id=req.dataset_id,
            test_size=req.test_size,
            alpha=req.alpha,
        )

        # Return a clean JSON response the UI can display
        return {
            "model_id": result.model_id,  # id for this trained model artifact
            "dataset_id": req.dataset_id,  # dataset used for training
            "rows_used": result.rows_used,  # rows after feature building
            "train_rows": result.train_rows,  # rows used for training
            "test_rows": result.test_rows,  # rows used for testing
            "mae": result.mae,  # mean absolute error
            "rmse": result.rmse,  # root mean squared error
            "nmae": result.nmae,
            "model_path": result.model_path,  # where the .joblib was saved (inside container)
            "feature_cols": result.feature_cols,  # which features the model trained on
            "model_type": "RidgeRegression",
            "prediction_target": "next_hour_flow_rate",
        }

    # If the dataset file doesn't exist on disk, return 404
    except FileNotFoundError as e:
        logger.warning("train_model dataset not found: %s", e)
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        logger.warning("train_model invalid input: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.exception("train_model crashed")
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")


@app.get("/datasets")
def list_datasets():
    files = sorted(DATASETS_DIR.glob("*.csv"))
    return {"datasets": [f.stem for f in files]}  # stem = filename without .csv


@app.get("/models")
def list_models():
    files = sorted(MODELS_DIR.glob("*.joblib"))
    return {"models": [f.stem for f in files]}


class PredictRequest(BaseModel):
    model_id: str = Field(..., description="Model id returned from /models/train")
    dataset_id: str | None = Field(None, description="Optional dataset id (defaults to the model's dataset_id)")
    horizon: int = Field(24, ge=1, le=168, description="How many hours ahead to forecast (1-168)")


@app.post("/models/predict")
def predict(req: PredictRequest):
    try:
        result = forecast_next_hours(
            model_id=req.model_id,
            dataset_id=req.dataset_id,
            horizon=req.horizon,
        )
        return result

    except FileNotFoundError as e:
        logger.warning("predict not found: %s", e)
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        logger.warning("predict invalid input: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.exception("predict crashed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/models/{model_id}/info")
def model_info(model_id: str):
    """
    Returns metadata about a trained model artifact (no sklearn object returned).
    """
    try:
        artifact = load_model_artifact(model_id)

        return {
            "model_id": artifact.get("model_id", model_id),
            "dataset_id": artifact.get("dataset_id"),
            "created_at": artifact.get("created_at"),
            "timestamp_col": artifact.get("timestamp_col"),
            "target_col": artifact.get("target_col"),
            "feature_cols": artifact.get("feature_cols", []),
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load model info: {e}")


@app.get("/models/{model_id}/metrics")
def model_metrics(model_id: str):
    try:
        artifact = load_model_artifact(model_id)
        return {
            "model_id": artifact.get("model_id", model_id),
            "dataset_id": artifact.get("dataset_id"),
            "params": artifact.get("params", {}),
            "metrics": artifact.get("metrics", {}),
            "created_at": artifact.get("created_at"),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load model metrics: {e}")


@app.get("/datasets")
def list_datasets():
    files = sorted(DATASETS_DIR.glob("*.csv"))
    return {"datasets": [f.stem for f in files]}


@app.get("/models")
def list_models():
    files = sorted(MODELS_DIR.glob("*.joblib"))
    return {"models": [f.stem for f in files]}


def _validate_id(id_str: str, kind: str):
    # simple safety: prevent path traversal / weird filenames
    if not id_str or len(id_str) > 64 or (not id_str.replace("-", "").isalnum()):
        raise HTTPException(status_code=400, detail=f"invalid {kind}_id")


@app.delete("/models/{model_id}")
def delete_model(model_id: str):
    _validate_id(model_id, "model")

    path = MODELS_DIR / f"{model_id}.joblib"
    if not path.exists():
        raise HTTPException(status_code=404, detail="model not found")

    path.unlink()
    return {"deleted": True, "model_id": model_id}


@app.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    _validate_id(dataset_id, "dataset")

    path = DATASETS_DIR / f"{dataset_id}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="dataset not found")

    # block deleting a dataset if any model references it
    used_by = []
    for f in sorted(MODELS_DIR.glob("*.joblib")):
        mid = f.stem
        try:
            art = load_model_artifact(mid)
            if art.get("dataset_id") == dataset_id:
                used_by.append(mid)
        except Exception:
            # ignore unreadable/old artifacts
            continue

    if used_by:
        raise HTTPException(
            status_code=409,
            detail=f"cannot delete dataset {dataset_id}; used by models: {used_by}. delete those models first."
        )

    path.unlink()
    return {"deleted": True, "dataset_id": dataset_id}
