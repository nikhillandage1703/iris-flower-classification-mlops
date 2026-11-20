"""Automated integration tests for FastAPI REST API endpoints."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.api.app import app, load_model_and_metadata


@pytest.fixture(scope="module")
def client():
    """Provides FastAPI test client fixture with model loaded."""
    load_model_and_metadata()
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Verify that root GET / returns 200 OK (dashboard or JSON)."""
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint(client):
    """Verify that GET /health returns operational status and model readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["model_loaded"] is True
    assert data["status"] == "healthy"


def test_model_info_endpoint(client):
    """Verify that GET /model-info returns metadata of the trained best model."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "algorithm" in data
    assert "best_model_name" in data
    assert "feature_names" in data
    assert "best_model_metrics" in data
    assert "accuracy" in data["best_model_metrics"]


def test_single_prediction_valid_input(client):
    """Verify that POST /predict returns correct species and probability distribution."""
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["predicted_species"] == "Iris-setosa"
    assert data["predicted_class"] == 0
    assert "probabilities" in data
    assert "Iris-setosa" in data["probabilities"]
    assert data["probabilities"]["Iris-setosa"] > 0.8


def test_batch_prediction_valid_input(client):
    """Verify that POST /predict-batch correctly handles multiple flower inputs."""
    payload = {
        "inputs": [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },
            {
                "sepal_length": 6.7,
                "sepal_width": 3.1,
                "petal_length": 4.7,
                "petal_width": 1.5,
            },
            {
                "sepal_length": 6.9,
                "sepal_width": 3.1,
                "petal_length": 5.4,
                "petal_width": 2.1,
            },
        ]
    }
    response = client.post("/predict-batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_predictions"] == 3
    assert len(data["predictions"]) == 3
    species_list = [p["predicted_species"] for p in data["predictions"]]
    assert species_list[0] == "Iris-setosa"


def test_invalid_input_negative_values(client):
    """Verify that POST /predict rejects negative or zero physical measurements with 422."""
    payload = {
        "sepal_length": -5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_invalid_input_missing_field(client):
    """Verify that POST /predict rejects incomplete request payloads with 422."""
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        # missing petal_length and petal_width
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_invalid_input_wrong_type(client):
    """Verify that POST /predict rejects non-numeric types with 422."""
    payload = {
        "sepal_length": "very_long",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_metrics_api_endpoint(client):
    """Verify that GET /api/metrics returns the model comparison metrics."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "decision_tree" in data
    assert "random_forest" in data
    assert "best_model" in data


def test_dataset_stats_api_endpoint(client):
    """Verify that GET /api/dataset-stats returns dataset summary statistics."""
    response = client.get("/api/dataset-stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_samples"] == 150
    assert data["total_features"] == 4
    assert "statistics" in data
