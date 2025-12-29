# Backend (FastAPI) - PipelineOps API

This service is the “data + model” layer for the PipelineOps Forecast project.
The Streamlit UI calls this API to upload time-series datasets, preview them, and (next) train/forecast.

## What this backend does (current scope)
1. Accepts a dataset upload (CSV)
2. Validates and cleans it into a standard format
3. Persists the cleaned dataset to disk using a dataset id
4. Provides a preview endpoint for the UI to display a table and chart

## Architecture at a glance
- `FastAPI` provides HTTP endpoints
- `pandas` handles CSV parsing and time-series cleaning
- Docker Compose mounts `./storage` on the host to `/data` in the container
  - This makes uploaded datasets persist even if containers restart

## Directory overview
```txt
backend/
  app/
    main.py            # FastAPI app + endpoints
  requirements.txt
  Dockerfile
