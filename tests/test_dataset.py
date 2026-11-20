"""Automated tests for Iris Dataset integrity, shape, schema, and quality."""

from pathlib import Path
import pytest
import pandas as pd
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import (
    RAW_DATA_FILE,
    TRAIN_DATA_FILE,
    TEST_DATA_FILE,
    FEATURE_NAMES,
    TARGET_COLUMN,
    CLASS_MAPPING,
)
from src.data.load_data import load_and_save_raw_data
from src.data.validate_data import validate_iris_dataframe


@pytest.fixture(scope="module")
def raw_dataset():
    """Fixture ensuring raw dataset is available for testing."""
    if not RAW_DATA_FILE.exists():
        load_and_save_raw_data(RAW_DATA_FILE)
    df = pd.read_csv(RAW_DATA_FILE)
    return df


def test_dataset_file_exists():
    """Verify that the raw Iris dataset file exists on disk."""
    assert RAW_DATA_FILE.exists(), f"Dataset file does not exist at {RAW_DATA_FILE}"


def test_dataset_shape(raw_dataset):
    """Verify the dataset has exactly 150 rows and the required columns."""
    assert raw_dataset.shape[0] == 150, f"Expected 150 rows, got {raw_dataset.shape[0]}"
    for feature in FEATURE_NAMES:
        assert feature in raw_dataset.columns, f"Feature column '{feature}' is missing"
    assert TARGET_COLUMN in raw_dataset.columns, f"Target column '{TARGET_COLUMN}' is missing"


def test_no_missing_or_null_values(raw_dataset):
    """Verify zero NaN, null, or missing entries across all features and target."""
    cols_to_check = FEATURE_NAMES + [TARGET_COLUMN]
    null_counts = raw_dataset[cols_to_check].isnull().sum()
    assert null_counts.sum() == 0, f"Found missing values in dataset: {null_counts.to_dict()}"


def test_feature_data_types_and_positive_ranges(raw_dataset):
    """Verify that features are numeric floats and strictly positive dimensions."""
    for col in FEATURE_NAMES:
        assert pd.api.types.is_numeric_dtype(raw_dataset[col]), f"Column '{col}' is not numeric"
        assert (raw_dataset[col] > 0).all(), f"Found non-positive values in column '{col}'"
        assert (raw_dataset[col] < 30.0).all(), f"Found unrealistic outlier in column '{col}'"


def test_class_balance_and_labels(raw_dataset):
    """Verify that each of the 3 classes contains exactly 50 samples."""
    target_counts = raw_dataset[TARGET_COLUMN].value_counts().to_dict()
    assert len(target_counts) == 3, f"Expected 3 target classes, found {len(target_counts)}"
    for class_idx in CLASS_MAPPING.keys():
        assert target_counts.get(class_idx) == 50, (
            f"Class {class_idx} has {target_counts.get(class_idx)} samples, expected 50"
        )


def test_validate_dataframe_function(raw_dataset):
    """Verify that the standalone validate_iris_dataframe helper executes and returns True."""
    assert validate_iris_dataframe(raw_dataset) is True
