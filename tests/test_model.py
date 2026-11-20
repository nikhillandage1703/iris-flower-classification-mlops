"""Automated tests for Machine Learning Model training, serialization, loading, and prediction."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import (
    BEST_MODEL_FILE,
    DECISION_TREE_FILE,
    RANDOM_FOREST_FILE,
    METADATA_FILE,
    FEATURE_NAMES,
    CLASS_NAMES,
)
from src.models.train import train_models


@pytest.fixture(scope="module")
def trained_models():
    """Fixture ensuring models are trained and serialized before running tests."""
    if not BEST_MODEL_FILE.exists() or not METADATA_FILE.exists():
        train_models()
    return {
        "best_model": joblib.load(BEST_MODEL_FILE),
        "decision_tree": joblib.load(DECISION_TREE_FILE),
        "random_forest": joblib.load(RANDOM_FOREST_FILE),
    }


def test_model_files_exist(trained_models):
    """Verify that all serialized model files exist on disk."""
    assert BEST_MODEL_FILE.exists(), "best_model.joblib does not exist"
    assert DECISION_TREE_FILE.exists(), "decision_tree.joblib does not exist"
    assert RANDOM_FOREST_FILE.exists(), "random_forest.joblib does not exist"
    assert METADATA_FILE.exists(), "model_metadata.json does not exist"


def test_model_can_be_loaded(trained_models):
    """Verify that models can be deserialized using joblib."""
    for name, model in trained_models.items():
        assert model is not None, f"Model {name} failed to load"
        assert hasattr(model, "predict"), f"Model {name} missing .predict() method"


def test_prediction_returns_valid_classes(trained_models):
    """Verify that prediction output returns valid class indexes (0, 1, or 2)."""
    # Sample inputs representing Setosa, Versicolor, and Virginica
    samples = pd.DataFrame(
        [
            [5.1, 3.5, 1.4, 0.2],  # Classic Setosa
            [6.0, 2.7, 4.2, 1.3],  # Classic Versicolor
            [7.2, 3.2, 6.0, 1.8],  # Classic Virginica
        ],
        columns=FEATURE_NAMES,
    )

    for name, model in trained_models.items():
        preds = model.predict(samples)
        assert len(preds) == 3, f"{name}: Expected 3 predictions, got {len(preds)}"
        for pred in preds:
            assert int(pred) in [0, 1, 2], f"{name}: Predicted invalid class {pred}"


def test_prediction_probabilities_validity(trained_models):
    """Verify that predicted probabilities sum to ~1.0 and lie in [0, 1]."""
    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=FEATURE_NAMES)

    for name, model in trained_models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(sample)[0]
            assert len(probs) == 3, f"{name}: Expected 3 class probabilities"
            assert np.all(probs >= 0.0) and np.all(probs <= 1.0), f"{name}: Probabilities outside [0, 1]"
            assert np.isclose(np.sum(probs), 1.0, atol=1e-4), f"{name}: Probabilities do not sum to 1.0"


def test_setosa_prediction_accuracy(trained_models):
    """Verify that unmistakable Setosa feature measurements reliably classify as Setosa (0)."""
    setosa_sample = pd.DataFrame([[5.0, 3.6, 1.4, 0.2]], columns=FEATURE_NAMES)
    best_model = trained_models["best_model"]
    pred = int(best_model.predict(setosa_sample)[0])
    assert pred == 0, f"Expected class 0 (Setosa), but got {pred}"
