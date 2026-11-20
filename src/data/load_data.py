"""Stage 1: Load Dataset from Scikit-Learn and persist to data/raw/iris.csv."""

import logging
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris

import sys
# Add project root to sys.path so config can be imported directly
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import (
    RAW_DATA_FILE,
    RAW_DATA_DIR,
    FEATURE_NAMES,
    TARGET_COLUMN,
    CLASS_MAPPING,
    load_params,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def load_and_save_raw_data(output_path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Loads Iris dataset from scikit-learn and saves it as raw CSV."""
    logger.info("Loading Iris dataset from scikit-learn...")
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()

    # Standardize column names
    # Default names are 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'target'
    rename_mapping = {
        iris.feature_names[0]: FEATURE_NAMES[0],
        iris.feature_names[1]: FEATURE_NAMES[1],
        iris.feature_names[2]: FEATURE_NAMES[2],
        iris.feature_names[3]: FEATURE_NAMES[3],
        "target": TARGET_COLUMN,
    }
    df = df.rename(columns=rename_mapping)
    df["species_name"] = df[TARGET_COLUMN].map(CLASS_MAPPING)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Raw Iris dataset successfully saved to: {output_path} (Shape: {df.shape})")
    return df


if __name__ == "__main__":
    params = load_params()
    raw_path = Path(params.get("data_load", {}).get("raw_data_path", RAW_DATA_FILE))
    load_and_save_raw_data(raw_path)
