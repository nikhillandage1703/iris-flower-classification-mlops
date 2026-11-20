"""Stage 5: Train, Evaluate, Compare, and Serialize Machine Learning Models with MLflow tracking.

Primary Model: Decision Tree Classifier
Comparison Model: Random Forest Classifier
"""

import os
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import joblib

# Ensure MLflow compatibility and suppress unnecessary verbose hints
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import (
    TRAIN_DATA_FILE,
    TEST_DATA_FILE,
    MODELS_DIR,
    DECISION_TREE_FILE,
    RANDOM_FOREST_FILE,
    BEST_MODEL_FILE,
    METADATA_FILE,
    METRICS_FILE,
    FEATURE_NAMES,
    TARGET_COLUMN,
    CLASS_NAMES,
    load_params,
    PROJECT_ROOT,
)
from src.models.evaluate import compute_metrics, plot_and_save_confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def train_models():
    """Trains Decision Tree and Random Forest models, tracks with MLflow, and selects the best model."""
    params = load_params()
    train_cfg = params.get("train", {})
    mlflow_cfg = params.get("mlflow", {})

    train_path = Path(train_cfg.get("train_data_path", TRAIN_DATA_FILE))
    test_path = Path(train_cfg.get("test_data_path", TEST_DATA_FILE))

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Training/Testing datasets missing. Looked for: {train_path} and {test_path}"
        )

    # Load splits
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[FEATURE_NAMES]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_NAMES]
    y_test = test_df[TARGET_COLUMN]

    # Setup MLflow Tracking
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = mlflow_cfg.get("experiment_name", "Iris_Classification_Experiment")
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow tracking URI: {tracking_uri}, Experiment: {experiment_name}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # ==========================================
    # 1. Primary Model: Decision Tree Classifier
    # ==========================================
    dt_params = train_cfg.get("decision_tree", {
        "criterion": "gini",
        "max_depth": 4,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
    })

    logger.info("--- Training Decision Tree Classifier ---")
    dt_model = DecisionTreeClassifier(**dt_params)
    dt_model.fit(X_train, y_train)

    y_train_pred_dt = dt_model.predict(X_train)
    y_test_pred_dt = dt_model.predict(X_test)

    dt_train_metrics = compute_metrics(y_train, y_train_pred_dt, split_name="train")
    dt_test_metrics = compute_metrics(y_test, y_test_pred_dt, split_name="test")

    # Generate and save DT confusion matrix
    dt_cm_path = plot_and_save_confusion_matrix(
        dt_test_metrics["test_confusion_matrix"],
        model_name="Decision Tree",
        output_filename="dt_confusion_matrix.png",
    )

    # MLflow Run: Decision Tree
    with mlflow.start_run(run_name="Decision_Tree_Run") as run:
        mlflow.log_param("algorithm", "DecisionTreeClassifier")
        for k, v in dt_params.items():
            mlflow.log_param(k, v)

        # Log metrics
        for k, v in dt_train_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
        for k, v in dt_test_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        mlflow.log_artifact(str(dt_cm_path), artifact_path="figures")
        mlflow.sklearn.log_model(
            dt_model,
            artifact_path="decision_tree_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
            serialization_format="cloudpickle",
        )
        dt_run_id = run.info.run_id

    joblib.dump(dt_model, DECISION_TREE_FILE)
    logger.info(f"Decision Tree saved to {DECISION_TREE_FILE}")

    results["decision_tree"] = {
        "model": dt_model,
        "run_id": dt_run_id,
        "params": dt_params,
        "train_metrics": dt_train_metrics,
        "test_metrics": dt_test_metrics,
        "artifact_path": str(DECISION_TREE_FILE),
    }

    # ==========================================
    # 2. Comparison Model: Random Forest Classifier
    # ==========================================
    rf_params = train_cfg.get("random_forest", {
        "n_estimators": 100,
        "criterion": "gini",
        "max_depth": 4,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
    })

    logger.info("--- Training Random Forest Classifier ---")
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train, y_train)

    y_train_pred_rf = rf_model.predict(X_train)
    y_test_pred_rf = rf_model.predict(X_test)

    rf_train_metrics = compute_metrics(y_train, y_train_pred_rf, split_name="train")
    rf_test_metrics = compute_metrics(y_test, y_test_pred_rf, split_name="test")

    # Generate and save RF confusion matrix
    rf_cm_path = plot_and_save_confusion_matrix(
        rf_test_metrics["test_confusion_matrix"],
        model_name="Random Forest",
        output_filename="rf_confusion_matrix.png",
    )

    # MLflow Run: Random Forest
    with mlflow.start_run(run_name="Random_Forest_Run") as run:
        mlflow.log_param("algorithm", "RandomForestClassifier")
        for k, v in rf_params.items():
            mlflow.log_param(k, v)

        # Log metrics
        for k, v in rf_train_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
        for k, v in rf_test_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        mlflow.log_artifact(str(rf_cm_path), artifact_path="figures")
        mlflow.sklearn.log_model(
            rf_model,
            artifact_path="random_forest_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
            serialization_format="cloudpickle",
        )
        rf_run_id = run.info.run_id

    joblib.dump(rf_model, RANDOM_FOREST_FILE)
    logger.info(f"Random Forest saved to {RANDOM_FOREST_FILE}")

    results["random_forest"] = {
        "model": rf_model,
        "run_id": rf_run_id,
        "params": rf_params,
        "train_metrics": rf_train_metrics,
        "test_metrics": rf_test_metrics,
        "artifact_path": str(RANDOM_FOREST_FILE),
    }

    # ==========================================
    # 3. Best Model Selection & Metadata Export
    # ==========================================
    dt_score = dt_test_metrics["test_f1_weighted"]
    rf_score = rf_test_metrics["test_f1_weighted"]

    # When scores are tied or RF >= DT, select the best generalization model
    if rf_score >= dt_score:
        best_name = "random_forest"
        best_display = "Random Forest Classifier"
        best_model = rf_model
        best_metrics = rf_test_metrics
        best_params = rf_params
        best_run_id = rf_run_id
    else:
        best_name = "decision_tree"
        best_display = "Decision Tree Classifier"
        best_model = dt_model
        best_metrics = dt_test_metrics
        best_params = dt_params
        best_run_id = dt_run_id

    # Serialize best model
    joblib.dump(best_model, BEST_MODEL_FILE)
    logger.info(f"Selected Best Model: {best_display} (F1: {best_metrics['test_f1_weighted']:.4f})")
    logger.info(f"Saved best model to {BEST_MODEL_FILE}")

    # Save comprehensive metadata
    metadata = {
        "project_name": params.get("base", {}).get("project_name", "Iris Flower Classification"),
        "best_model_name": best_name,
        "best_model_display": best_display,
        "algorithm": best_model.__class__.__name__,
        "mlflow_run_id": best_run_id,
        "mlflow_experiment": experiment_name,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "test_size": params.get("preprocess", {}).get("test_size", 0.2),
        "random_state": params.get("base", {}).get("random_state", 42),
        "best_model_hyperparameters": best_params,
        "best_model_metrics": {
            "accuracy": best_metrics["test_accuracy"],
            "precision_weighted": best_metrics["test_precision_weighted"],
            "recall_weighted": best_metrics["test_recall_weighted"],
            "f1_score_weighted": best_metrics["test_f1_weighted"],
        },
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Exported model metadata to {METADATA_FILE}")

    # Save full metrics comparison for dashboard & DVC
    metrics_export = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "best_model": best_name,
        "decision_tree": {
            "params": dt_params,
            "metrics": {
                "train_accuracy": dt_train_metrics["train_accuracy"],
                "test_accuracy": dt_test_metrics["test_accuracy"],
                "test_precision_weighted": dt_test_metrics["test_precision_weighted"],
                "test_precision_macro": dt_test_metrics["test_precision_macro"],
                "test_recall_weighted": dt_test_metrics["test_recall_weighted"],
                "test_recall_macro": dt_test_metrics["test_recall_macro"],
                "test_f1_weighted": dt_test_metrics["test_f1_weighted"],
                "test_f1_macro": dt_test_metrics["test_f1_macro"],
                "confusion_matrix": dt_test_metrics["test_confusion_matrix"],
                "classification_report": dt_test_metrics["test_classification_report"],
            },
        },
        "random_forest": {
            "params": rf_params,
            "metrics": {
                "train_accuracy": rf_train_metrics["train_accuracy"],
                "test_accuracy": rf_test_metrics["test_accuracy"],
                "test_precision_weighted": rf_test_metrics["test_precision_weighted"],
                "test_precision_macro": rf_test_metrics["test_precision_macro"],
                "test_recall_weighted": rf_test_metrics["test_recall_weighted"],
                "test_recall_macro": rf_test_metrics["test_recall_macro"],
                "test_f1_weighted": rf_test_metrics["test_f1_weighted"],
                "test_f1_macro": rf_test_metrics["test_f1_macro"],
                "confusion_matrix": rf_test_metrics["test_confusion_matrix"],
                "classification_report": rf_test_metrics["test_classification_report"],
            },
        },
    }

    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2)
    logger.info(f"Exported metrics report to {METRICS_FILE}")

    return metadata, metrics_export


if __name__ == "__main__":
    train_models()
