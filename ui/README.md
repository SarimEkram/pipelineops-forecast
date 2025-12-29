# UI (Streamlit) - PipelineOps Forecast Dashboard

This Streamlit app is the front-end for the PipelineOps Forecast project.
It provides a simple dashboard that lets a user upload a dataset, preview it, and (next) train/forecast.

## What the UI does (current scope)
1. Confirms the backend is reachable (System Check page)
2. Uploads a CSV dataset to the backend
3. Stores the returned `dataset_id` in Streamlit session state
4. Requests a preview sample and displays:
   - a table preview
   - a line chart of `flow_rate` over time

The UI does not store datasets itself. It relies on the backend for persistence.

## Architecture (how it works)
- Streamlit renders the web app and handles user interactions
- The UI communicates with the backend using HTTP calls via `requests`
- The backend returns JSON; the UI converts preview JSON to a pandas DataFrame for display/charting

### Why the API URL looks “weird”
When running with Docker Compose, the UI uses:
- `API_URL = "http://backend:8000"`

This works because:
- `backend` is the Docker Compose service name
- containers can reach each other using service names on the Compose network

If you run the UI locally (not inside Docker), `backend` won’t resolve.
In that case, change API_URL to:
- `http://localhost:8000`

## Directory overview
```txt
ui/
  streamlit_app.py       # Streamlit dashboard
  requirements.txt
  Dockerfile
  README.md
