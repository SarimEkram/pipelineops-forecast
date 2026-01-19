import os
import importlib
from datetime import datetime, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient


def _make_hourly_csv(rows: int = 200) -> bytes:
    start = datetime(2025, 1, 1, 0, 0, 0)
    ts = [start + timedelta(hours=i) for i in range(rows)]
    df = pd.DataFrame({
        "timestamp": [t.isoformat() for t in ts],
        "flow_rate": [100 + (i % 24) for i in range(rows)],
    })
    return df.to_csv(index=False).encode("utf-8")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # force backend to use temp storage
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("MAX_UPLOAD_MB", "50")

    # IMPORTANT: reload module so it picks up env vars
    from app import main as main_mod  # if your package name differs, adjust import
    importlib.reload(main_mod)

    return TestClient(main_mod.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_upload_rejects_non_csv(client):
    r = client.post(
        "/datasets/upload",
        files={"file": ("data.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400


def test_upload_missing_columns(client):
    bad_csv = b"a,b\n1,2\n"
    r = client.post(
        "/datasets/upload",
        files={"file": ("bad.csv", bad_csv, "text/csv")},
    )
    assert r.status_code == 400
    assert "missing columns" in r.text


def test_upload_train_predict_flow(client):
    csv_bytes = _make_hourly_csv(250)

    # upload
    up = client.post(
        "/datasets/upload",
        files={"file": ("ok.csv", csv_bytes, "text/csv")},
    )
    assert up.status_code == 200
    dataset_id = up.json()["dataset_id"]

    # sample
    smp = client.get(f"/datasets/{dataset_id}/sample", params={"rows": 50})
    assert smp.status_code == 200
    assert smp.json()["rows_returned"] == 50

    # train
    tr = client.post("/models/train", json={"dataset_id": dataset_id, "test_size": 0.2, "alpha": 1.0})
    assert tr.status_code == 200
    model_id = tr.json()["model_id"]
    assert "mae" in tr.json()

    # predict
    pr = client.post("/models/predict", json={"model_id": model_id, "dataset_id": dataset_id, "horizon": 24})
    assert pr.status_code == 200
    payload = pr.json()
    assert payload["horizon"] == 24
    assert len(payload["predictions"]) == 24
