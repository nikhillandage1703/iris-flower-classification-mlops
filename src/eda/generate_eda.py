"""Stage 3: Exploratory Data Analysis (EDA).

Generates meaningful, publication-quality visualizations from actual Iris dataset values:
1. Class distribution bar chart
2. Sepal length distribution with KDE
3. Sepal width distribution with KDE
4. Petal length distribution with KDE
5. Petal width distribution with KDE
6. Feature correlation heatmap
7. Feature comparison plots (pairplot / multi-feature scatter)
"""

import logging
from pathlib import Path
import shutil
import sys
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import (
    RAW_DATA_FILE,
    FIGURES_DIR,
    FEATURE_NAMES,
    FEATURE_LABELS,
    STATIC_DIR,
    CLASS_MAPPING,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Professional color palette
PALETTE = ["#3b82f6", "#10b981", "#f59e0b"]  # Blue, Green, Amber


def setup_style():
    """Sets consistent styling for clean, professional charts."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#cbd5e1"
    plt.rcParams["axes.linewidth"] = 0.8


def save_chart(fig: plt.Figure, filename: str, output_dirs: list[Path]):
    """Saves figure to designated output directories."""
    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / filename
        fig.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure artifact: {filename}")


def generate_eda_figures(data_path: Path = RAW_DATA_FILE) -> list[str]:
    """Generates all required EDA figures and returns their filenames."""
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot perform EDA: dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    if "species_name" not in df.columns:
        df["species_name"] = df["target"].map(CLASS_MAPPING)

    setup_style()
    static_fig_dir = STATIC_DIR / "figures"
    target_dirs = [FIGURES_DIR, static_fig_dir]
    generated_files = []

    # 1. Class Distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = df["species_name"].value_counts()
    bars = ax.bar(counts.index, counts.values, color=PALETTE, width=0.55, edgecolor="#1e293b", linewidth=0.8)
    ax.set_title("Iris Species Class Distribution (Balanced n=50 each)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Species Class", fontsize=11, labelpad=8)
    ax.set_ylabel("Sample Count", fontsize=11, labelpad=8)
    ax.set_ylim(0, 60)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    fontweight="bold", color="#1e293b")
    save_chart(fig, "class_distribution.png", target_dirs)
    generated_files.append("class_distribution.png")

    # 2-5. Individual Feature Distributions with KDE
    for feature in FEATURE_NAMES:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for idx, (species, group) in enumerate(df.groupby("species_name")):
            sns.kdeplot(
                data=group[feature],
                ax=ax,
                label=species,
                color=PALETTE[idx % len(PALETTE)],
                fill=True,
                alpha=0.35,
                linewidth=2
            )
        ax.set_title(f"Distribution of {FEATURE_LABELS[feature]} by Species", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(FEATURE_LABELS[feature], fontsize=11, labelpad=8)
        ax.set_ylabel("Density Estimate", fontsize=11, labelpad=8)
        ax.legend(title="Species", frameon=True)
        fname = f"{feature}_distribution.png"
        save_chart(fig, fname, target_dirs)
        generated_files.append(fname)

    # 6. Feature Correlation Heatmap
    fig, ax = plt.subplots(figsize=(6.5, 5))
    corr = df[FEATURE_NAMES].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        ax=ax,
        cbar=True,
        square=True,
        linewidths=1.0,
        linecolor="#f1f5f9",
        annot_kws={"size": 11, "weight": "bold"}
    )
    ax.set_title("Iris Feature Correlation Matrix (Pearson r)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURE_NAMES], rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels([FEATURE_LABELS[f] for f in FEATURE_NAMES], rotation=0, fontsize=9)
    save_chart(fig, "correlation_heatmap.png", target_dirs)
    generated_files.append("correlation_heatmap.png")

    # 7. Feature Comparison Plot (Petal Length vs Petal Width Scatter)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for idx, (species, group) in enumerate(df.groupby("species_name")):
        ax.scatter(
            group["petal_length"],
            group["petal_width"],
            label=species,
            color=PALETTE[idx % len(PALETTE)],
            alpha=0.85,
            edgecolors="#0f172a",
            linewidth=0.7,
            s=65
        )
    ax.set_title("Petal Length vs Petal Width (Key Linear Separability Feature)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(FEATURE_LABELS["petal_length"], fontsize=11, labelpad=8)
    ax.set_ylabel(FEATURE_LABELS["petal_width"], fontsize=11, labelpad=8)
    ax.legend(title="Species", frameon=True)
    save_chart(fig, "feature_comparison.png", target_dirs)
    generated_files.append("feature_comparison.png")

    # 8. Pairwise Feature Comparison Grid
    pairgrid = sns.pairplot(
        df[FEATURE_NAMES + ["species_name"]],
        hue="species_name",
        palette=PALETTE,
        diag_kind="kde",
        corner=False,
        plot_kws={"alpha": 0.8, "s": 35, "edgecolor": "none"}
    )
    pairgrid.figure.suptitle("Multivariate Feature Pairwise Relationships Across Species", y=1.02, fontsize=14, fontweight="bold")
    pairgrid.savefig(FIGURES_DIR / "pairplot_comparison.png", dpi=200, bbox_inches="tight")
    pairgrid.savefig(static_fig_dir / "pairplot_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(pairgrid.figure)
    generated_files.append("pairplot_comparison.png")

    logger.info(f"Successfully generated all {len(generated_files)} EDA figures.")
    return generated_files


if __name__ == "__main__":
    generate_eda_figures()
