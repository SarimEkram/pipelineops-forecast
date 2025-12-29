# Streamlit = the library that builds the web UI in Python
import streamlit as st

# requests = lets our UI talk to the backend API over HTTP (GET/POST)
import requests

# pandas = used for tables + time parsing + sorting (makes charting easy)
import pandas as pd

# Sets basic settings for the Streamlit page (tab title + wide layout)
st.set_page_config(page_title="PipelineOps Forecast", layout="wide")

# Big title at the top of the page
st.title("PipelineOps Forecast")

# Smaller text under the title
st.caption("Pipeline operations dashboard (MVP)")

# This is where our backend lives INSIDE docker-compose.
# "backend" is the service name from docker-compose.yml, not localhost.
API_URL = "http://backend:8000"

# Streamlit apps re-run top-to-bottom any time the user clicks something.
# session_state is how we "remember" values between reruns.
# Here we create a storage slot for dataset_id if it doesn't exist yet.
if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None  # Nothing uploaded yet

# Sidebar navigation: user selects which page to view.
# radio() returns the selected string.
page = st.sidebar.radio("Navigation", ["System Check", "Upload Data"])

# -------------------------
# Page 1: System Check
# -------------------------
if page == "System Check":
    # Header for this section
    st.subheader("System Check")

    # We wrap backend calls in try/except so the UI doesn't crash if backend is down.
    try:
        # Call the backend /health endpoint to check if backend is running.
        # timeout=2 means: don't hang longer than 2 seconds.
        r = requests.get(f"{API_URL}/health", timeout=2)

        # If the backend returns HTTP 200, it means it's healthy.
        if r.status_code == 200:
            # Show a green success box and print the JSON response.
            st.success(f"Backend API: OK ({r.json()})")
        else:
            # If backend responded but not with 200, show the status code.
            st.error(f"Backend API returned {r.status_code}")

    except Exception as e:
        # If the backend can't be reached at all, show the error message.
        st.error(f"Backend API not reachable ({e})")

# -------------------------
# Page 2: Upload Data
# -------------------------
elif page == "Upload Data":
    # Header for upload page
    st.subheader("Upload Data (CSV)")

    # Instructions for the user about required column names
    st.write("Required columns: timestamp, flow_rate")

    # File picker UI.
    # type=["csv"] means user can only pick CSV files.
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    # If the user picked a file, uploaded will not be None.
    if uploaded is not None:
        # Show which file was selected so user has confirmation.
        st.info(f"Selected: {uploaded.name}")

        # We only upload when the user presses the button.
        # This prevents uploading repeatedly on every rerun.
        if st.button("Upload to Backend"):
            # Build a "multipart/form-data" payload for FastAPI to receive.
            # uploaded.getvalue() gives the raw file bytes.
            files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}

            try:
                # POST the file to the backend upload endpoint.
                # timeout=30 because upload + parsing can take longer.
                r = requests.post(f"{API_URL}/datasets/upload", files=files, timeout=30)

                # If upload succeeded (HTTP 200), backend returns JSON details.
                if r.status_code == 200:
                    data = r.json()  # convert backend response to a Python dict

                    # Save dataset_id so we can use it later for preview/training.
                    # session_state keeps this value during the app session.
                    st.session_state.dataset_id = data["dataset_id"]

                    # Show success message + dataset id so user can see it.
                    st.success(f"Upload successful. Dataset ID: {st.session_state.dataset_id}")

                    # Show full backend response for transparency/debugging.
                    st.json(data)

                else:
                    # If backend returned an error, show the raw error text.
                    st.error(r.text)

            except Exception as e:
                # Network error / backend down / etc.
                st.error(f"Upload failed: {e}")

    # If we have a dataset_id saved, we can preview it.
    # This is how the UI "knows" a dataset exists to work with.
    if st.session_state.dataset_id:
        # Horizontal divider for visual separation
        st.divider()

        # Header for preview section
        st.subheader("Dataset Preview")

        # Show which dataset the preview belongs to
        st.caption(f"dataset_id: {st.session_state.dataset_id}")

        # Slider lets user choose how many rows to fetch for preview.
        # We limit it to avoid huge API responses.
        rows = st.slider("Rows to preview", min_value=10, max_value=500, value=200, step=10)

        try:
            # Call the backend sample endpoint to fetch the first N rows.
            # params={"rows": rows} becomes ?rows=200 in the URL.
            r = requests.get(
                f"{API_URL}/datasets/{st.session_state.dataset_id}/sample",
                params={"rows": rows},
                timeout=10
            )

            # If backend returned 200, we got preview data back.
            if r.status_code == 200:
                payload = r.json()  # dict with keys like dataset_id, data, rows_returned

                # payload["data"] is a list of rows (each row is a dict).
                # Turning it into a pandas DataFrame makes it easy to display + chart.
                df = pd.DataFrame(payload["data"])

                # Show the table in the UI (scrollable).
                st.dataframe(df, use_container_width=True)

                # Only chart if the columns exist (basic safety check).
                if "timestamp" in df.columns and "flow_rate" in df.columns:
                    # Convert timestamp strings into actual datetimes (so sorting works properly).
                    df["timestamp"] = pd.to_datetime(df["timestamp"])

                    # Ensure the data is in time order before plotting.
                    df = df.sort_values("timestamp")

                    # Streamlit line_chart uses the index as the x-axis.
                    # We set timestamp as the index and plot flow_rate as the y-axis.
                    st.line_chart(df.set_index("timestamp")["flow_rate"])

            else:
                # Backend responded but with an error (show it).
                st.error(r.text)

        except Exception as e:
            # Network error / backend down / endpoint missing / etc.
            st.error(f"Preview failed: {e}")
