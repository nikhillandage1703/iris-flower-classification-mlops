"""Central configuration module for Iris MLOps Project.

Uses pathlib for cross-platform compatibility across Windows, Linux, and macOS.
"""

from pathlib import Path
from typing import Any, Dict
import yaml

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Core directory paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_FILE = RAW_DATA_DIR / "iris.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.csv"

MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_FILE = MODELS_DIR / "best_model.joblib"
DECISION_TREE_FILE = MODELS_DIR / "decision_tree.joblib"
RANDOM_FOREST_FILE = MODELS_DIR / "random_forest.joblib"
METADATA_FILE = MODELS_DIR / "model_metadata.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_FILE = REPORTS_DIR / "metrics.json"

PARAMS_FILE = PROJECT_ROOT / "params.yaml"

# Static directory for API frontend
STATIC_DIR = PROJECT_ROOT / "src" / "api" / "static"

# Canonical feature names and target mapping
FEATURE_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

FEATURE_LABELS = {
    "sepal_length": "Sepal Length (cm)",
    "sepal_width": "Sepal Width (cm)",
    "petal_length": "Petal Length (cm)",
    "petal_width": "Petal Width (cm)",
}

TARGET_COLUMN = "target"
SPECIES_COLUMN = "species"

CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
CLASS_MAPPING = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}


def load_params(params_path: Path = PARAMS_FILE) -> Dict[str, Any]:
    """Safely loads parameters from YAML file."""
    if not params_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {params_path}")
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
