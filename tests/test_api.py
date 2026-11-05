"""Tests that run automatically on every push via GitHub Actions.

The fixtures are six images from the NEU-DET *test* split (one per defect
class), so they were never seen during training.
"""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import model as model_module
from app.main import app

FIXTURES = Path(__file__).parent / "fixtures"
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# Release gate: CI fails if someone commits weights worse than this.
MIN_TEST_MAP50 = 0.72  # measured: 0.758 (models/metrics.json)


def fixture_for(label: str) -> Path:
    return next(FIXTURES.glob(f"{label}_*.jpg"))


def upload(path: Path) -> dict:
    return {"file": (path.name, path.read_bytes(), "image/jpeg")}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    body = client.get("/model").json()
    assert body["classes"] == CLASSES
    assert body["metrics"]["split"] == "test"


@pytest.mark.parametrize("label", CLASSES)
def test_detects_each_defect_type(client, label):
    path = fixture_for(label)
    response = client.post("/predict", files=upload(path))

    assert response.status_code == 200
    body = response.json()
    assert body["defect_count"] == len(body["detections"]) > 0
    assert label in {d["label"] for d in body["detections"]}


def test_boxes_are_inside_the_image(client):
    body = client.post("/predict", files=upload(fixture_for("scratches"))).json()
    for d in body["detections"]:
        box = d["box"]
        assert 0 <= box["x1"] < box["x2"] <= body["image_width"]
        assert 0 <= box["y1"] < box["y2"] <= body["image_height"]
        assert 0.0 <= d["confidence"] <= 1.0


def test_detections_sorted_by_confidence(client):
    body = client.post("/predict", files=upload(fixture_for("inclusion"))).json()
    scores = [d["confidence"] for d in body["detections"]]
    assert scores == sorted(scores, reverse=True)


def test_high_confidence_threshold_filters_detections(client):
    path = fixture_for("inclusion")
    low = client.post("/predict", files=upload(path), params={"conf": 0.05}).json()
    high = client.post("/predict", files=upload(path), params={"conf": 0.9}).json()
    assert high["defect_count"] <= low["defect_count"]
    assert all(d["confidence"] >= 0.9 for d in high["detections"])


def test_accepts_png_and_color_images(client):
    gray = Image.open(fixture_for("patches"))
    buffer = io.BytesIO()
    gray.convert("RGB").save(buffer, format="PNG")
    files = {"file": ("patches.png", buffer.getvalue(), "image/png")}
    assert client.post("/predict", files=files).status_code == 200


def test_predict_image_returns_jpeg(client):
    response = client.post("/predict/image", files=upload(fixture_for("scratches")))
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(response.content)).format == "JPEG"


def test_rejects_non_image(client):
    files = {"file": ("notes.txt", b"definitely not an image", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 422
    assert "not a valid image" in response.json()["detail"]


def test_rejects_empty_file(client):
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    assert client.post("/predict", files=files).status_code == 422


def test_rejects_missing_file(client):
    assert client.post("/predict").status_code == 422


def test_rejects_out_of_range_confidence(client):
    path = fixture_for("scratches")
    response = client.post("/predict", files=upload(path), params={"conf": 2})
    assert response.status_code == 422


def test_rejects_oversized_file(client):
    big = b"\xff" * (model_module.MAX_IMAGE_BYTES + 1)
    files = {"file": ("big.jpg", big, "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 422
    assert "larger than" in response.json()["detail"]


def test_committed_model_meets_accuracy_bar():
    """Guard against committing weights that are quietly worse."""
    metrics = model_module.load_metrics()
    assert metrics is not None, "models/metrics.json is missing"
    assert metrics["mAP50"] >= MIN_TEST_MAP50, f"test mAP50 is {metrics['mAP50']}"
