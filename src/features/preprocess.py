"""Stage 4: Data Preprocessing, Feature/Target Separation, and Stratified Splitting.

Guarantees zero data leakage by strictly separating train and test partitions.
Uses stratified sampling to preserve balanced class proportions across both splits.
"""

import logging
from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import (
    RAW_DATA_FILE,
    TRAIN_DATA_FILE,
    TEST_DATA_FILE,
    FEATURE_NAMES,
    TARGET_COLUMN,
    load_params,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def preprocess_and_split(
    raw_path: Path = RAW_DATA_FILE,
    train_path: Path = TRAIN_DATA_FILE,
    test_path: Path = TEST_DATA_FILE,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads raw dataset, validates columns, performs stratified split, and persists partitions.

    Args:
        raw_path: Path to raw dataset CSV.
        train_path: Output path for training set.
        test_path: Output path for testing set.
        test_size: Fraction of samples reserved for testing (default: 0.2).
        random_state: Random seed for deterministic reproducibility.
        stratify: Whether to use stratified sampling based on the target class.

    Returns:
        tuple containing (train_df, test_df).
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {raw_path}")

    logger.info(f"Loading raw dataset from {raw_path}...")
    df = pd.read_csv(raw_path)

    required_cols = FEATURE_NAMES + [TARGET_COLUMN]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from raw dataset.")

    # Features and Target
    X = df[FEATURE_NAMES]
    y = df[TARGET_COLUMN]

    stratify_target = y if stratify else None

    logger.info(
        f"Splitting dataset: test_size={test_size}, random_state={random_state}, stratify={stratify}"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(
        f"Preprocessing and splitting complete! Train shape: {train_df.shape}, Test shape: {test_df.shape}"
    )
    logger.info(f"Train class distribution: {train_df[TARGET_COLUMN].value_counts().to_dict()}")
    logger.info(f"Test class distribution: {test_df[TARGET_COLUMN].value_counts().to_dict()}")

    return train_df, test_df


if __name__ == "__main__":
    params = load_params()
    prep_cfg = params.get("preprocess", {})
    preprocess_and_split(
        raw_path=Path(prep_cfg.get("raw_data_path", RAW_DATA_FILE)),
        train_path=Path(prep_cfg.get("train_data_path", TRAIN_DATA_FILE)),
        test_path=Path(prep_cfg.get("test_data_path", TEST_DATA_FILE)),
        test_size=float(prep_cfg.get("test_size", 0.2)),
        random_state=int(prep_cfg.get("random_state", 42)),
        stratify=bool(prep_cfg.get("stratify", True)),
    )
