# PipelineOps Forecast

Dockerized time-series forecasting dashboard for “pipeline operations” style data.
Upload CSV/XLSX → validate data → train a baseline model → generate next-hours forecasts with plots + CSV export.

## What it does (MVP)
- Upload time-series datasets (CSV or XLSX → mapped to `timestamp` + `flow_rate`)
- Preview datasets (table + line chart)
- Run data-quality checks (blocks train/forecast if data is invalid)
- Train a baseline Ridge regression model (time-ordered split)
- Forecast next hours (plot overlay: recent actuals vs. predictions) + download forecast CSV
- View and compare trained models (metrics table)

## Tech Stack
- UI: Streamlit
- Backend API: FastAPI
- ML/Data: pandas, scikit-learn
- Containerization: Docker + Docker Compose
- Storage: local `storage/` mounted into containers

## Architecture
- Streamlit UI calls the FastAPI backend over HTTP
- Backend persists datasets/models under the Docker volume mount (`/data`)

