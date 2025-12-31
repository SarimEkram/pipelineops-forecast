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


# -------------------------
# Sidebar navigation
# -------------------------

page = st.sidebar.radio("Navigation", ["System Check", "Upload Data", "Train Model"])

# reset Upload Data page UI/preview each time user navigates into it
if st.session_state.last_page != page:
    if page == "Upload Data":
        st.session_state.upload_preview_dataset_id = None
        st.session_state.upload_preview_rows = None
        st.session_state.uploader_key += 1  # clears file_uploader selection
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
    st.subheader("Upload Data (CSV)")
    st.write("Required columns: timestamp, flow_rate")

    # Clear button (optional but helpful)
    if st.button("Clear Upload Page"):
        st.session_state.upload_preview_dataset_id = None
        st.session_state.upload_preview_rows = None
        st.session_state.uploader_key += 1
        st.rerun()

    # File picker UI (only CSV allowed) — key makes it reset cleanly
    uploaded = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded is not None:
        st.info(f"Selected: {uploaded.name}")

        # Upload happens only when the user clicks (prevents repeated uploads on rerun)
        if st.button("Upload to Backend"):
            files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}

            try:
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

        # Cap preview rows to protect the API + UI from huge responses
        total_rows = st.session_state.upload_preview_rows
        max_preview = 500 if total_rows is None else min(500, int(total_rows))
        max_preview = max(10, max_preview)  # safety: max must be >= min

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

                # Plot flow_rate over time if columns are present
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

    st.caption(f"dataset_id: {st.session_state.dataset_id}")

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
        st.info("Model files are saved by the backend under `storage/models/<model_id>.joblib` (via the /data volume mount).")
