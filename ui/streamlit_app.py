import streamlit as st
import requests

st.set_page_config(page_title="PipelineOps Forecast", layout="wide")
st.title("PipelineOps Forecast")
st.caption("Pipeline operations dashboard (MVP)")

API_URL = "http://backend:8000"  # docker-compose service name

st.subheader("System Check")

try:
    r = requests.get(f"{API_URL}/health", timeout=2)
    if r.status_code == 200:
        st.success(f"Backend API: Ok  ({r.json()})")
    else:
        st.error(f"Backend API: returned {r.status_code}")
except Exception as e:
    st.error(f"Backend API: not reachable   ({e})")
