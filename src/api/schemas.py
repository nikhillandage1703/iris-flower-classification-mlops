"""Pydantic schemas for request validation and response models in FastAPI."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class IrisFeatures(BaseModel):
    """Input features schema for a single flower prediction."""

    sepal_length: float = Field(
        ...,
        gt=0.0,
        lt=20.0,
        description="Sepal length in centimeters (typical Iris range: 4.0 - 8.0 cm)",
        examples=[5.1],
    )
    sepal_width: float = Field(
        ...,
        gt=0.0,
        lt=20.0,
        description="Sepal width in centimeters (typical Iris range: 2.0 - 5.0 cm)",
        examples=[3.5],
    )
    petal_length: float = Field(
        ...,
        gt=0.0,
        lt=20.0,
        description="Petal length in centimeters (typical Iris range: 1.0 - 7.0 cm)",
        examples=[1.4],
    )
    petal_width: float = Field(
        ...,
        gt=0.0,
        lt=20.0,
        description="Petal width in centimeters (typical Iris range: 0.1 - 3.0 cm)",
        examples=[0.2],
    )


class BatchIrisFeatures(BaseModel):
    """Schema for batch prediction requests."""

    inputs: List[IrisFeatures] = Field(
        ...,
        min_length=1,
        description="List of Iris flower feature sets to predict in batch.",
    )


class PredictionResponse(BaseModel):
    """Schema for single prediction response."""

    predicted_class: int = Field(..., description="Target class index (0, 1, or 2)")
    predicted_species: str = Field(..., description="Canonical flower species name")
    probabilities: Dict[str, float] = Field(
        ..., description="Prediction probability distribution across classes"
    )
    model_name: str = Field(..., description="Name of the model serving predictions")
    status: str = Field(default="success", description="Prediction status flag")


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""

    total_predictions: int
    model_name: str
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Schema for system health check endpoint."""

    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    version: str


class ModelInfoResponse(BaseModel):
    """Schema for model information endpoint."""

    project_name: str
    best_model_name: str
    best_model_display: str
    algorithm: str
    mlflow_experiment: str
    training_timestamp: str
    feature_names: List[str]
    class_names: List[str]
    best_model_metrics: Dict[str, float]
    best_model_hyperparameters: Dict[str, object]
