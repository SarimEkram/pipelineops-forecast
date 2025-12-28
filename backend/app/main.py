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
