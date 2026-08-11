"""Tests that run automatically on every push via GitHub Actions."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_digits

from app import model as model_module
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_image():
    """One real image from the dataset, with its true label."""
    X, y = load_digits(return_X_y=True)
    return X[0].tolist(), int(y[0])


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_predict_returns_correct_label(client, sample_image):
    pixels, true_label = sample_image
    response = client.post("/predict", json={"pixels": pixels})

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == true_label
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_wrong_input_size(client):
    response = client.post("/predict", json={"pixels": [0.0] * 10})
    assert response.status_code == 422


def test_model_accuracy_above_threshold():
    """Guard against a change that quietly makes the model worse."""
    accuracy = model_module.train(save=False)
    assert accuracy > 0.90, f"accuracy dropped to {accuracy:.3f}"


def test_predict_output_shape():
    model = model_module.load()
    result = model_module.predict(model, np.zeros(64).tolist())
    assert set(result) == {"prediction", "confidence"}
