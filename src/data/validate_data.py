"""Stage 2: Validate Dataset integrity, schema, types, missing values, and distribution."""

import logging
from pathlib import Path
import pandas as pd
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import (
    RAW_DATA_FILE,
    FEATURE_NAMES,
    TARGET_COLUMN,
    CLASS_MAPPING,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Custom exception raised when dataset validation checks fail."""
    pass


def validate_iris_dataframe(df: pd.DataFrame) -> bool:
    """Performs rigorous validation on the Iris DataFrame.

    Checks:
    1. Row and column dimensions (150 samples, expected columns)
    2. Missing / null / NaN values
    3. Numerical column data types
    4. Positive, non-zero values for physical measurements
    5. Expected target classes and class distribution
    """
    logger.info("Starting dataset validation...")

    # 1. Check shape
    if df.shape[0] != 150:
        raise DataValidationError(f"Expected 150 rows, but found {df.shape[0]} rows.")
    
    # 2. Check required columns
    required_cols = FEATURE_NAMES + [TARGET_COLUMN]
    for col in required_cols:
        if col not in df.columns:
            raise DataValidationError(f"Missing required column: '{col}' in dataset.")

    # 3. Check null values
    null_counts = df[required_cols].isnull().sum().to_dict()
    if any(count > 0 for count in null_counts.values()):
        raise DataValidationError(f"Dataset contains null values: {null_counts}")

    # 4. Check data types and value ranges
    for feature in FEATURE_NAMES:
        if not pd.api.types.is_numeric_dtype(df[feature]):
            raise DataValidationError(f"Feature '{feature}' must be numeric, got {df[feature].dtype}")
        if (df[feature] <= 0).any():
            raise DataValidationError(f"Feature '{feature}' has non-positive or zero physical dimensions.")

    # 5. Check target classes
    unique_targets = set(df[TARGET_COLUMN].unique())
    expected_targets = set(CLASS_MAPPING.keys())
    if unique_targets != expected_targets:
        raise DataValidationError(
            f"Expected target classes {expected_targets}, but got {unique_targets}"
        )

    # 6. Check class balance (50 samples per class)
    class_counts = df[TARGET_COLUMN].value_counts().to_dict()
    for target_class, count in class_counts.items():
        if count != 50:
            raise DataValidationError(
                f"Class {target_class} has {count} samples, expected 50 samples."
            )

    logger.info("Dataset validation passed successfully! All integrity checks verified.")
    return True


def run_validation(data_path: Path = RAW_DATA_FILE) -> bool:
    """Reads dataset from disk and executes validation."""
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot validate dataset: file not found at {data_path}")
    df = pd.read_csv(data_path)
    return validate_iris_dataframe(df)


if __name__ == "__main__":
    run_validation()
