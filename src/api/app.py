"""FastAPI Application providing REST endpoints and serving the Interactive MLOps Dashboard."""

from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import (
    BEST_MODEL_FILE,
    METADATA_FILE,
    METRICS_FILE,
    RAW_DATA_FILE,
    FEATURE_NAMES,
    CLASS_MAPPING,
    CLASS_NAMES,
    STATIC_DIR,
)
from src.api.schemas import (
    IrisFeatures,
    BatchIrisFeatures,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Application state
model_holder: Dict[str, Any] = {
    "model": None,
    "metadata": None,
}


def load_model_and_metadata():
    """Loads the serialized model and metadata from disk."""
    if BEST_MODEL_FILE.exists():
        try:
            model_holder["model"] = joblib.load(BEST_MODEL_FILE)
            logger.info(f"Loaded serialized model from: {BEST_MODEL_FILE}")
        except Exception as e:
            logger.error(f"Failed to load model file: {e}")
            model_holder["model"] = None
    else:
        logger.warning(f"Model file not found at: {BEST_MODEL_FILE}")
        model_holder["model"] = None

    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                model_holder["metadata"] = json.load(f)
            logger.info("Loaded model metadata.")
        except Exception as e:
            logger.error(f"Failed to load metadata file: {e}")
            model_holder["metadata"] = None
    else:
        model_holder["metadata"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to ensure model is loaded at startup."""
    load_model_and_metadata()
    yield


app = FastAPI(
    title="Iris Flower Classification MLOps API",
    description="Production-grade REST API for Iris species prediction and MLOps metrics inspection.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serves the interactive MLOps frontend dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "message": "Iris Flower Classification MLOps API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint indicating API operational status and model readiness."""
    model_loaded = model_holder["model"] is not None
    model_name = (
        model_holder["metadata"].get("best_model_display")
        if model_holder["metadata"]
        else ("Loaded" if model_loaded else "Not Loaded")
    )
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_name=model_name,
        version="1.0.0",
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Info"])
async def get_model_info():
    """Returns metadata and metrics of the currently deployed best model."""
    if not model_holder["metadata"]:
        # If in-memory metadata is None, attempt on-disk reload
        load_model_and_metadata()

    if not model_holder["metadata"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metadata artifact not found. Please train models first.",
        )
    return model_holder["metadata"]


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_single(features: IrisFeatures):
    """Predicts the Iris flower species for a single feature vector."""
    model = model_holder.get("model")
    if model is None:
        load_model_and_metadata()
        model = model_holder.get("model")

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model is not loaded. Train the model using DVC/MLflow first.",
        )

    # Prepare DataFrame matching training feature columns
    input_data = pd.DataFrame(
        [
            [
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width,
            ]
        ],
        columns=FEATURE_NAMES,
    )

    try:
        class_idx = int(model.predict(input_data)[0])
        species_name = CLASS_MAPPING.get(class_idx, f"Class {class_idx}")

        # Compute prediction probabilities if model supports it
        probabilities = {}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_data)[0]
            for idx, prob in enumerate(probs):
                probabilities[CLASS_MAPPING.get(idx, f"Class {idx}")] = round(float(prob), 4)
        else:
            # Fallback for models without predict_proba
            for idx in range(3):
                probabilities[CLASS_MAPPING[idx]] = 1.0 if idx == class_idx else 0.0

        model_name = (
            model_holder["metadata"].get("best_model_display", model.__class__.__name__)
            if model_holder["metadata"]
            else model.__class__.__name__
        )

        return PredictionResponse(
            predicted_class=class_idx,
            predicted_species=species_name,
            probabilities=probabilities,
            model_name=model_name,
            status="success",
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing model inference: {str(e)}",
        )


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(batch: BatchIrisFeatures):
    """Performs high-throughput batch prediction for multiple flowers."""
    model = model_holder.get("model")
    if model is None:
        load_model_and_metadata()
        model = model_holder.get("model")

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model is not loaded.",
        )

    if not batch.inputs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch list cannot be empty.",
        )

    rows = [
        [
            item.sepal_length,
            item.sepal_width,
            item.petal_length,
            item.petal_width,
        ]
        for item in batch.inputs
    ]
    input_data = pd.DataFrame(rows, columns=FEATURE_NAMES)

    try:
        preds = model.predict(input_data)
        has_proba = hasattr(model, "predict_proba")
        probs_matrix = model.predict_proba(input_data) if has_proba else None

        results: List[PredictionResponse] = []
        model_name = (
            model_holder["metadata"].get("best_model_display", model.__class__.__name__)
            if model_holder["metadata"]
            else model.__class__.__name__
        )

        for i, class_idx in enumerate(preds):
            class_idx = int(class_idx)
            species_name = CLASS_MAPPING.get(class_idx, f"Class {class_idx}")
            probabilities = {}
            if has_proba:
                for c_idx, prob in enumerate(probs_matrix[i]):
                    probabilities[CLASS_MAPPING.get(c_idx, f"Class {c_idx}")] = round(float(prob), 4)
            else:
                for c_idx in range(3):
                    probabilities[CLASS_MAPPING[c_idx]] = 1.0 if c_idx == class_idx else 0.0

            results.append(
                PredictionResponse(
                    predicted_class=class_idx,
                    predicted_species=species_name,
                    probabilities=probabilities,
                    model_name=model_name,
                    status="success",
                )
            )

        return BatchPredictionResponse(
            total_predictions=len(results),
            model_name=model_name,
            predictions=results,
        )
    except Exception as e:
        logger.error(f"Batch inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing batch inference: {str(e)}",
        )


@app.get("/api/metrics", tags=["Dashboard Data"])
async def get_metrics():
    """Returns evaluation metrics comparing Decision Tree and Random Forest."""
    if not METRICS_FILE.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics report file not found. Train models first.",
        )
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/dataset-stats", tags=["Dashboard Data"])
async def get_dataset_stats():
    """Returns descriptive statistics of the Iris dataset for the explorer dashboard."""
    if not RAW_DATA_FILE.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Raw dataset not found.",
        )
    df = pd.read_csv(RAW_DATA_FILE)

    stats = df[FEATURE_NAMES].describe().to_dict()
    species_counts = df["species_name"].value_counts().to_dict()
    sample_preview = df.head(10).to_dict(orient="records")

    return {
        "total_samples": len(df),
        "total_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "class_counts": species_counts,
        "statistics": stats,
        "preview": sample_preview,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
