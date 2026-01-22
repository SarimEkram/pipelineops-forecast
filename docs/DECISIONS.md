# Engineering Decisions

- **Why Docker Compose:** One command spins up the full stack (UI + API) with consistent environments and ports, so anyone can run the project locally without manual setup.

- **Why FastAPI + Streamlit:** FastAPI keeps the backend modular/testable (clean API boundaries), while Streamlit enables fast iteration on the dashboard and workflows.

- **Why store artifacts in `/storage` (mounted volume):** Uploaded datasets and trained models persist across container restarts and stay separated from source code; also avoids committing runtime files to git.

- **Why dataset IDs + model IDs:** Each upload/train produces a unique, traceable artifact (dataset_id/model_id) so runs are reproducible and you can compare results without overwriting files.

- **Why data-quality gates before training:** Prevents “garbage in, garbage out” by blocking training/forecasting when the dataset format is invalid or misleading (e.g., wrong columns, broken timestamps).

- **Schema validation (what we check):** Required columns exist, target column is numeric, file is non-empty / meets a minimum row threshold, and values can be parsed safely.

- **Timestamp integrity checks (what we check):** Timestamp column parses to datetime, rows sort correctly, sampling frequency is hourly, and gaps (missing hours) are detected and flagged/blocked.

- **Why time-ordered train/test split:** Preserves causality and avoids leakage (future data influencing the past), which is essential for time-series forecasting.

- **Why Ridge regression baseline:** Strong, simple baseline that trains quickly, handles correlated features well, and gives a reliable benchmark before trying more complex models.

- **Why feature set (hour/day + lags + rolling):** Time-based features capture seasonality, lag features capture short-term dependency, and rolling stats smooth noise while keeping the model lightweight.
