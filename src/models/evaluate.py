"""Model evaluation utilities for computing metrics, confusion matrices, and plots."""

import json
import logging
from pathlib import Path
from typing import Any, Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from config.config import CLASS_NAMES, FIGURES_DIR, STATIC_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, split_name: str = "test") -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics for multiclass classification.

    Averaging Method Explanation:
    - 'weighted' averaging calculates metrics for each label, and finds their average weighted by support (number of true instances for each label).
      This accounts for label imbalance (if any).
    - 'macro' averaging calculates metrics for each label independently, and finds their unweighted mean.
      This treats all classes equally regardless of support.
    For the balanced Iris dataset (n=10 per class in test split), weighted and macro yield mathematically identical or near-identical results.
    Both are reported for technical thoroughness.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )

    return {
        f"{split_name}_accuracy": acc,
        f"{split_name}_precision_weighted": prec_weighted,
        f"{split_name}_precision_macro": prec_macro,
        f"{split_name}_recall_weighted": rec_weighted,
        f"{split_name}_recall_macro": rec_macro,
        f"{split_name}_f1_weighted": f1_weighted,
        f"{split_name}_f1_macro": f1_macro,
        f"{split_name}_confusion_matrix": cm,
        f"{split_name}_classification_report": report,
    }


def plot_and_save_confusion_matrix(
    cm: list[list[int]],
    model_name: str,
    output_filename: str,
) -> Path:
    """Renders an attractive heatmap of the confusion matrix and saves to disk."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_arr = np.array(cm)

    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        cbar=True,
        square=True,
        linewidths=1.2,
        linecolor="#ffffff",
        annot_kws={"size": 13, "weight": "bold"}
    )
    ax.set_title(f"{model_name} Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Species", fontsize=11, labelpad=8)
    ax.set_ylabel("True Species", fontsize=11, labelpad=8)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    report_path = FIGURES_DIR / output_filename
    fig.savefig(report_path, dpi=300, bbox_inches="tight")

    static_fig_dir = STATIC_DIR / "figures"
    static_fig_dir.mkdir(parents=True, exist_ok=True)
    static_path = static_fig_dir / output_filename
    fig.savefig(static_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    logger.info(f"Saved confusion matrix plot: {report_path}")
    return report_path
