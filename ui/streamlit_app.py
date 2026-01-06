# Streamlit = builds the web UI in Python
import streamlit as st

# requests = lets the UI talk to the backend API over HTTP (GET/POST)
import requests

# pandas = used for tables + time parsing + sorting (makes charting easy)
import pandas as pd

# -------------------------
# App configuration / header
# -------------------------

# Sets basic settings for the Streamlit page (tab title + wide layout)
st.set_page_config(page_title="PipelineOps Forecast", layout="wide")

# Big title at the top of the page
st.title("PipelineOps Forecast")

# Smaller text under the title
st.caption("Pipeline operations dashboard (MVP)")

# -------------------------
# Backend address
# -------------------------

# Inside docker-compose, "backend" is the service name, not localhost.
# If you run UI outside Docker, change to "http://localhost:8000".
API_URL = "http://backend:8000"

# -------------------------
# Session state (memory between reruns)
# -------------------------

# Streamlit reruns the script any time you interact with the UI.
# session_state is how we "remember" things like dataset_id/model_id between reruns.
if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None  # nothing uploaded yet

if "dataset_rows" not in st.session_state:
    st.session_state.dataset_rows = None  # set after upload (helps cap the preview slider)

if "model_id" not in st.session_state:
    st.session_state.model_id = None  # set after training

# --- upload page specific state (prevents old previews from showing) ---
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0  # used to reset file_uploader

if "upload_preview_dataset_id" not in st.session_state:
    st.session_state.upload_preview_dataset_id = None  # only for Upload Data page preview

if "upload_preview_rows" not in st.session_state:
    st.session_state.upload_preview_rows = None

if "last_page" not in st.session_state:
    st.session_state.last_page = None

# --- forecast page state ---
if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None

if "forecast_model_prev" not in st.session_state:
    st.session_state.forecast_model_prev = None

# -------------------------
# Sidebar navigation
# -------------------------

page = st.sidebar.radio("Navigation", ["System Check", "Upload Data", "Train Model", "Models", "Forecast"])

# reset Upload Data page UI/preview each time user navigates into it
if st.session_state.last_page != page:
    if page == "Upload Data":
        st.session_state.upload_preview_dataset_id = None
        st.session_state.upload_preview_rows = None
        st.session_state.uploader_key += 1  # clears file_uploader selection

    if page == "Forecast":
        st.session_state.forecast_result = None  # prevents old forecast from showing like a "ghost" result

    st.session_state.last_page = page

# -------------------------
# Page 1: System Check
# -------------------------
if page == "System Check":
    st.subheader("System Check")

    try:
        # Ping backend health endpoint so we know the API is reachable
        r = requests.get(f"{API_URL}/health", timeout=2)

        if r.status_code == 200:
            st.success(f"Backend API: OK ({r.json()})")
        else:
            st.error(f"Backend API returned {r.status_code}: {r.text}")

    except Exception as e:
        st.error(f"Backend API not reachable ({e})")


# -------------------------
# Page 2: Upload Data
# -------------------------
elif page == "Upload Data":
    st.subheader("Upload Data (CSV or Excel)")
    st.write("CSV must have columns: timestamp, flow_rate. Excel can be mapped using dropdowns below.")

    # Clear button (optional but helpful)
    if st.button("Clear Upload Page"):
        st.session_state.upload_preview_dataset_id = None
        st.session_state.upload_preview_rows = None
        st.session_state.uploader_key += 1
        st.rerun()

    uploaded = st.file_uploader(
        "Choose a CSV or XLSX file",
        type=["csv", "xlsx"],
        key=f"uploader_{st.session_state.uploader_key}"
    )

    excel_df = None
    ts_col = None
    val_col = None

    if uploaded is not None:
        st.info(f"Selected: {uploaded.name}")

        ext = uploaded.name.split(".")[-1].lower()

        # If XLSX, load it locally so user can choose columns (e.g., DT_MST + Calgary)
        if ext == "xlsx":
            try:
                excel_df = pd.read_excel(uploaded, engine="openpyxl")
                st.caption(f"Excel loaded: {excel_df.shape[0]} rows × {excel_df.shape[1]} cols")

                cols = list(excel_df.columns)

                # Good defaults for AESO-style file
                default_ts = cols.index("DT_MST") if "DT_MST" in cols else 0
                default_val = cols.index("Calgary") if "Calgary" in cols else (1 if len(cols) > 1 else 0)

                c1, c2 = st.columns(2)
                with c1:
                    ts_col = st.selectbox("Timestamp column", cols, index=default_ts)
                with c2:
                    val_col = st.selectbox("Value column (flow_rate)", cols, index=default_val)

                st.caption("Preview (first 10 rows of selected columns)")
                st.dataframe(excel_df[[ts_col, val_col]].head(10), use_container_width=True)

            except Exception as e:
                st.error(f"Could not read Excel file: {e}")
                st.stop()

        # Upload happens only when the user clicks (prevents repeated uploads on rerun)
        if st.button("Upload to Backend"):
            try:
                # If XLSX: convert to standard CSV (timestamp, flow_rate) before sending
                if ext == "xlsx":
                    if excel_df is None:
                        st.error("Excel not loaded. Please re-select the file.")
                        st.stop()

                    if ts_col is None or val_col is None:
                        st.error("Please choose the timestamp and value columns.")
                        st.stop()

                    df_send = excel_df[[ts_col, val_col]].copy()
                    df_send = df_send.rename(columns={ts_col: "timestamp", val_col: "flow_rate"})

                    # Convert to CSV bytes for backend
                    csv_bytes = df_send.to_csv(index=False).encode("utf-8")
                    upload_name = uploaded.name.rsplit(".", 1)[0] + ".csv"
                    files = {"file": (upload_name, csv_bytes, "text/csv")}

                else:
                    # CSV: send as-is
                    files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}

                r = requests.post(f"{API_URL}/datasets/upload", files=files, timeout=30)

                if r.status_code == 200:
                    data = r.json()

                    # Store dataset details so other pages can use them
                    st.session_state.dataset_id = data["dataset_id"]
                    st.session_state.dataset_rows = data.get("rows")

                    # Upload page preview should show ONLY what was uploaded here
                    st.session_state.upload_preview_dataset_id = data["dataset_id"]
                    st.session_state.upload_preview_rows = data.get("rows")

                    # Optional: reset model_id when a new dataset is uploaded
                    st.session_state.model_id = None

                    st.success(f"Upload successful. Dataset ID: {st.session_state.dataset_id}")
                    st.json(data)
                else:
                    st.error(r.text)

            except Exception as e:
                st.error(f"Upload failed: {e}")

    # Dataset preview (shows ONLY if something was uploaded on this Upload page visit)
    if st.session_state.upload_preview_dataset_id:
        st.divider()
        st.subheader("Dataset Preview")
        st.caption(f"dataset_id: {st.session_state.upload_preview_dataset_id}")

        total_rows = st.session_state.upload_preview_rows
        max_preview = 500 if total_rows is None else min(500, int(total_rows))
        max_preview = max(10, max_preview)

        rows = st.slider(
            "Rows to preview",
            min_value=10,
            max_value=max_preview,
            value=min(200, max_preview),
            step=10
        )

        try:
            r = requests.get(
                f"{API_URL}/datasets/{st.session_state.upload_preview_dataset_id}/sample",
                params={"rows": rows},
                timeout=10
            )

            if r.status_code == 200:
                payload = r.json()
                df = pd.DataFrame(payload["data"])

                st.dataframe(df, use_container_width=True)

                if "timestamp" in df.columns and "flow_rate" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
                    st.line_chart(df.set_index("timestamp")["flow_rate"])
            else:
                st.error(r.text)

        except Exception as e:
            st.error(f"Preview failed: {e}")


# -------------------------
# Page 3: Train Model
# -------------------------
elif page == "Train Model":
    st.subheader("Train Model (Ridge Regression)")

    # 1) fetch existing datasets from backend (you need a /datasets endpoint for this)
    r = requests.get(f"{API_URL}/datasets", timeout=5)
    datasets = r.json().get("datasets", [])

    if not datasets:
        st.warning("No datasets found. Upload one first.")
        st.stop()

    chosen = st.selectbox("Choose an existing dataset_id", datasets, key="dataset_choice")
    st.session_state.dataset_id = chosen
    st.caption(f"dataset_id: {st.session_state.dataset_id}")

    # If no dataset uploaded yet, we can’t train anything
    if not st.session_state.dataset_id:
        st.info("No dataset selected yet. Loading existing datasets from storage...")

        try:
            r = requests.get(f"{API_URL}/datasets", timeout=5)
            if r.status_code == 200:
                datasets = r.json().get("datasets", [])
                if datasets:
                    chosen = st.selectbox("Choose an existing dataset_id", datasets)
                    if st.button("Use this dataset"):
                        st.session_state.dataset_id = chosen
                        st.rerun()
                else:
                    st.warning("No datasets found. Upload one first.")
                    st.stop()
            else:
                st.error(r.text)
                st.stop()
        except Exception as e:
            st.error(f"Could not reach backend: {e}")
            st.stop()

    # st.caption(f"dataset_id: {st.session_state.dataset_id}")  # <- removed to prevent duplicate

    # Training controls (matches your FastAPI TrainModelRequest schema)
    col1, col2 = st.columns(2)

    with col1:
        test_size = st.slider(
            "Test size (fraction of most recent data held out)",
            min_value=0.05,
            max_value=0.50,
            value=0.20,
            step=0.05
        )

    with col2:
        alpha = st.number_input(
            "Ridge alpha (regularization strength)",
            min_value=0.01,
            value=1.0,
            step=0.1
        )

    # Train button (only runs when clicked)
    if st.button("Train Model"):
        try:
            payload = {
                "dataset_id": st.session_state.dataset_id,
                "test_size": float(test_size),
                "alpha": float(alpha),
            }

            r = requests.post(f"{API_URL}/models/train", json=payload, timeout=60)

            if r.status_code == 200:
                data = r.json()
                st.session_state.model_id = data.get("model_id")

                st.success(f"Training complete. model_id: {st.session_state.model_id}")

                # Show the key results first (easy to read)
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col1.metric("MAE", f"{data.get('mae', 0):.4f}")
                metric_col2.metric("RMSE", f"{data.get('rmse', 0):.4f}")
                metric_col3.metric("Rows used", str(data.get("rows_used", "")))

                # Full JSON for debugging / transparency
                st.json(data)

            else:
                st.error(r.text)

        except Exception as e:
            st.error(f"Training failed: {e}")

    # If we already trained in this session, show the last model id
    if st.session_state.model_id:
        st.divider()
        st.subheader("Latest trained model")
        st.caption(f"model_id: {st.session_state.model_id}")
        st.info(
            "Model files are saved by the backend under `storage/models/<model_id>.joblib` (via the /data volume mount).")


elif page == "Models":
    st.subheader("Models (Leaderboard)")
    st.write("Compare trained models and pick one to use for forecasting.")

    # Fetch models list
    try:
        r = requests.get(f"{API_URL}/models", timeout=5)
        model_ids = r.json().get("models", [])
    except Exception as e:
        st.error(f"Could not load models: {e}")
        st.stop()

    if not model_ids:
        st.warning("No models found. Train a model first.")
        st.stop()

    rows = []
    with st.spinner("Loading model metrics..."):
        for mid in model_ids:
            try:
                rm = requests.get(f"{API_URL}/models/{mid}/metrics", timeout=5)
                if rm.status_code != 200:
                    # Older models might not have metrics; keep them but mark missing
                    rows.append({"model_id": mid, "mae": None, "rmse": None, "dataset_id": None, "created_at": None})
                    continue

                payload = rm.json()
                metrics = payload.get("metrics", {}) or {}
                params = payload.get("params", {}) or {}

                rows.append({
                    "model_id": payload.get("model_id", mid),
                    "dataset_id": payload.get("dataset_id"),
                    "created_at": payload.get("created_at"),
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "rows_used": metrics.get("rows_used"),
                    "alpha": params.get("alpha"),
                    "test_size": params.get("test_size"),
                })
            except Exception:
                rows.append({"model_id": mid, "mae": None, "rmse": None, "dataset_id": None, "created_at": None})

    df = pd.DataFrame(rows)

    # Sort best-first (lowest MAE), pushing None to bottom
    if "mae" in df.columns:
        df["mae_sort"] = pd.to_numeric(df["mae"], errors="coerce")
        df = df.sort_values(["mae_sort"], ascending=True).drop(columns=["mae_sort"])

    st.dataframe(df, use_container_width=True)

    st.divider()
    best_default = st.session_state.model_id if st.session_state.model_id in model_ids else model_ids[0]
    chosen_model = st.selectbox("Select a model to use", model_ids, index=model_ids.index(best_default))

    if st.button("Use selected model for Forecast"):
        st.session_state.model_id = chosen_model
        st.success(f"Selected model_id set to: {chosen_model}. Go to the Forecast page and run prediction.")


# -------------------------
# Page 4: Forecast
# -------------------------
elif page == "Forecast":
    st.subheader("Forecast (Next-Hours Prediction)")
    st.write("Pick a dataset + a trained model, then generate a next-hours forecast.")

    # Fetch datasets
    try:
        r_ds = requests.get(f"{API_URL}/datasets", timeout=5)
        datasets = r_ds.json().get("datasets", [])
    except Exception as e:
        st.error(f"Could not reach backend for datasets: {e}")
        st.stop()

    if not datasets:
        st.warning("No datasets found. Upload one first.")
        st.stop()

    # Fetch models
    try:
        r_m = requests.get(f"{API_URL}/models", timeout=5)
        models = r_m.json().get("models", [])
    except Exception as e:
        st.error(f"Could not reach backend for models: {e}")
        st.stop()

    if not models:
        st.warning("No models found. Train a model first.")
        st.stop()

    # Default model (use session_state if available)
    m_default = st.session_state.model_id if st.session_state.model_id in models else models[0]
    m_index = models.index(m_default)

    colA, colB, colC = st.columns([2, 2, 1])

    # 1) Pick model first (so we can auto-pick its dataset)
    with colB:
        model_id = st.selectbox("Model", models, index=m_index, key="forecast_model")

    # 2) Fetch model metadata from backend
    model_info = None
    try:
        r_info = requests.get(f"{API_URL}/models/{model_id}/info", timeout=5)
        if r_info.status_code == 200:
            model_info = r_info.json()
        else:
            model_info = None
    except Exception:
        model_info = None

    # 3) Choose dataset default:
    # prefer the model's dataset_id if it exists; otherwise fall back to session_state or first dataset
    model_ds = (model_info or {}).get("dataset_id")
    if model_ds in datasets:
        ds_default = model_ds
    else:
        ds_default = st.session_state.dataset_id if st.session_state.dataset_id in datasets else datasets[0]

    # Force dataset dropdown to follow the selected model when the model changes
    if st.session_state.forecast_model_prev != model_id:
        st.session_state["forecast_dataset"] = ds_default
        st.session_state.forecast_result = None
        st.session_state.forecast_model_prev = model_id

    ds_index = datasets.index(st.session_state.get("forecast_dataset", ds_default))

    with colA:
        dataset_id = st.selectbox("Dataset", datasets, index=ds_index, key="forecast_dataset")

    with colC:
        horizon = st.number_input("Horizon (hours)", min_value=1, max_value=168, value=24, step=1)

    st.caption(f"dataset_id: {dataset_id}")
    st.caption(f"model_id: {model_id}")

    # Optional: show extra info if available
    if model_info:
        st.caption(f"model trained on: {model_info.get('dataset_id')}")
        st.caption(f"created_at: {model_info.get('created_at')}")

    if st.button("Run Forecast"):
        try:
            payload = {
                "model_id": model_id,
                "dataset_id": dataset_id,  # explicit so there's no confusion
                "horizon": int(horizon),
            }

            r = requests.post(f"{API_URL}/models/predict", json=payload, timeout=60)

            if r.status_code == 200:
                st.session_state.forecast_result = r.json()
                st.success("Forecast generated successfully.")
            else:
                st.error(r.text)

        except Exception as e:
            st.error(f"Forecast failed: {e}")

    # Display results
    if st.session_state.forecast_result:
        result = st.session_state.forecast_result

        preds = result.get("predictions", [])
        if not preds:
            st.warning("No predictions returned.")
            st.stop()

        pred_df = pd.DataFrame(preds)
        pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], errors="coerce")
        pred_df = pred_df.dropna(subset=["timestamp"]).sort_values("timestamp")
        pred_df = pred_df.set_index("timestamp")

        st.divider()
        st.subheader("Export")

        export_df = pred_df.reset_index().copy()
        export_csv = export_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download forecast as CSV",
            data=export_csv,
            file_name=f"forecast_{dataset_id}_{model_id}_{int(horizon)}h.csv",
            mime="text/csv"
        )

        # Pull recent actuals (we request up to 2000, then take the tail)
        try:
            r = requests.get(
                f"{API_URL}/datasets/{dataset_id}/sample",
                params={"rows": 2000},
                timeout=15
            )
            if r.status_code == 200:
                payload = r.json()
                actual_df = pd.DataFrame(payload.get("data", []))

                if "timestamp" in actual_df.columns and "flow_rate" in actual_df.columns:
                    actual_df["timestamp"] = pd.to_datetime(actual_df["timestamp"], errors="coerce")
                    actual_df["flow_rate"] = pd.to_numeric(actual_df["flow_rate"], errors="coerce")
                    actual_df = actual_df.dropna(subset=["timestamp", "flow_rate"]).sort_values("timestamp")

                    # show last N actual points
                    actual_df = actual_df.tail(200).set_index("timestamp")
                else:
                    actual_df = None
            else:
                actual_df = None
                st.error(r.text)
        except Exception as e:
            actual_df = None
            st.error(f"Could not load actuals for plotting: {e}")

        st.divider()
        st.subheader("Forecast Plot")

        # Build a combined frame so Streamlit can chart both series
        combined = pd.DataFrame(index=pd.Index([], name="timestamp"))

        if actual_df is not None and len(actual_df) > 0:
            combined = combined.join(actual_df.rename(columns={"flow_rate": "actual_flow_rate"}), how="outer")

        combined = combined.join(pred_df.rename(columns={"predicted_flow_rate": "forecast_flow_rate"}), how="outer")
        combined = combined.sort_index()

        st.line_chart(combined, use_container_width=True)

        st.divider()
        st.subheader("Forecast Table (next hours)")
        st.dataframe(pred_df.reset_index(), use_container_width=True)
