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

from .ml_train import train_ridge_model


# Create the FastAPI app (this is the server)
# title shows up in the docs UI at /docs
app = FastAPI(title="PipelineOps API")

# In Docker, we mounted: ./storage  ->  /data
# That means anything we save under /data will appear in your repo's storage/ folder
DATA_DIR = Path("/data")

# We'll store uploaded datasets in /data/datasets/
DATASETS_DIR = DATA_DIR / "datasets"

# Make sure the folder exists (parents=True means create missing parent folders too)
# exist_ok=True means "don't error if it already exists"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

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

    # Take first N rows requested
    preview = df.head(rows).copy()

    # Convert timestamp back to string so JSON serialization is clean
    preview["timestamp"] = preview["timestamp"].astype(str)

    # Return a simple JSON payload the UI can consume easily
    return {
        "dataset_id": dataset_id,
        "rows_returned": int(len(preview)),
        "data": preview.to_dict(orient="records")  # list of {timestamp, flow_rate:}
    }


# -------------------------
# ML Training Request Schema
# -------------------------

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


# -------------------------
# ML Training Endpoint
# -------------------------

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
            "model_path": result.model_path,  # where the .joblib was saved (inside container)
            "feature_cols": result.feature_cols,  # which features the model trained on
            "model_type": "RidgeRegression",
            "prediction_target": "next_hour_flow_rate",
        }

    # If the dataset file doesn't exist on disk, return 404
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # If the dataset exists but is invalid (missing cols / too small / etc), return 400
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Any unexpected error becomes a 500
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

@app.get("/datasets")
def list_datasets():
    files = sorted(DATASETS_DIR.glob("*.csv"))
    return {"datasets": [f.stem for f in files]}  # stem = filename without .csv
