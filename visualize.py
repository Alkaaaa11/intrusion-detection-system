"""
visualize.py
------------
Beginner-friendly visualization script for the Intrusion Detection System project.

This script creates and saves:
1) Attack distribution chart
2) Confusion matrix heatmap
3) Model accuracy comparison chart
4) Normal vs attack traffic chart
"""

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def ensure_charts_dir(charts_dir: Path) -> None:
    """
    Create charts directory if it does not exist.
    """
    charts_dir.mkdir(parents=True, exist_ok=True)


def plot_attack_distribution(data: pd.DataFrame, label_column: str, charts_dir: Path) -> None:
    """
    Create a bar chart showing count of each attack category.
    """
    if label_column not in data.columns:
        raise ValueError(f"Column '{label_column}' not found in DataFrame.")

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    order = data[label_column].value_counts().index
    ax = sns.countplot(data=data, x=label_column, order=order, hue=label_column, palette="Set2")

    plt.title("Attack Distribution", fontsize=14)
    plt.xlabel("Attack Category", fontsize=12)
    plt.ylabel("Number of Records", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    ax.legend(title="Category", loc="upper right")
    plt.tight_layout()

    output_path = charts_dir / "attack_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved chart: {output_path}")


def plot_confusion_matrix_heatmap(
    y_true: Iterable,
    y_pred: Iterable,
    class_names: Iterable[str],
    charts_dir: Path,
) -> None:
    """
    Create and save confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title("Confusion Matrix Heatmap", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()

    output_path = charts_dir / "confusion_matrix_heatmap.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved chart: {output_path}")


def plot_model_accuracy_comparison(model_scores: Dict[str, float], charts_dir: Path) -> None:
    """
    Create and save a bar chart comparing model accuracy scores.
    """
    if not model_scores:
        raise ValueError("model_scores dictionary is empty.")

    score_df = pd.DataFrame(
        {
            "Model": list(model_scores.keys()),
            "Accuracy": list(model_scores.values()),
        }
    ).sort_values(by="Accuracy", ascending=False)

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=score_df, x="Model", y="Accuracy", hue="Model", palette="viridis")

    plt.title("Model Accuracy Comparison", fontsize=14)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Accuracy Score", fontsize=12)
    plt.ylim(0, 1.0)
    ax.legend(title="Model", loc="lower right")

    # Add numeric labels on top of bars for readability.
    for index, value in enumerate(score_df["Accuracy"]):
        plt.text(index, value + 0.01, f"{value:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    output_path = charts_dir / "model_accuracy_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved chart: {output_path}")


def plot_normal_vs_attack(data: pd.DataFrame, label_column: str, charts_dir: Path) -> None:
    """
    Create and save graph comparing normal traffic vs attack traffic.
    """
    if label_column not in data.columns:
        raise ValueError(f"Column '{label_column}' not found in DataFrame.")

    # Create two groups: normal and attack.
    traffic_type = data[label_column].astype(str).apply(
        lambda x: "normal" if x.lower() == "normal" else "attack"
    )
    summary = traffic_type.value_counts().reindex(["normal", "attack"], fill_value=0)

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(x=summary.index, y=summary.values, hue=summary.index, palette=["#4CAF50", "#F44336"])

    plt.title("Normal vs Attack Traffic", fontsize=14)
    plt.xlabel("Traffic Type", fontsize=12)
    plt.ylabel("Number of Records", fontsize=12)
    ax.legend(title="Traffic", loc="upper right")

    for index, value in enumerate(summary.values):
        plt.text(index, value + max(1, int(0.01 * max(summary.values))), str(value), ha="center", fontsize=10)

    plt.tight_layout()
    output_path = charts_dir / "normal_vs_attack_traffic.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved chart: {output_path}")


def run_visualization_pipeline() -> None:
    """
    Example pipeline that reads processed test data and generates charts.
    """
    try:
        data_dir = Path("data")
        charts_dir = Path("charts")
        processed_test_path = data_dir / "processed_test.csv"

        ensure_charts_dir(charts_dir)

        if not processed_test_path.exists():
            raise FileNotFoundError(
                f"Missing file: {processed_test_path}. Run preprocessing.py first."
            )

        print("[INFO] Loading processed test data for visualizations...")
        test_df = pd.read_csv(processed_test_path)
        print(f"[INFO] Loaded test data shape: {test_df.shape}")

        # Use attack_category if available, otherwise fall back to label.
        label_column = "attack_category" if "attack_category" in test_df.columns else "label"
        print(f"[INFO] Using label column: {label_column}")

        plot_attack_distribution(test_df, label_column, charts_dir)
        plot_normal_vs_attack(test_df, label_column, charts_dir)

        # Demo accuracy comparison values (replace with real results if available).
        demo_scores = {
            "RandomForest": 0.92,
            "DecisionTree": 0.88,
            "LogisticRegression": 0.84,
        }
        plot_model_accuracy_comparison(demo_scores, charts_dir)

        # Demo confusion matrix generated from example predictions.
        # This keeps the script runnable even before full prediction integration.
        y_true_demo = test_df[label_column].head(500)
        y_pred_demo = y_true_demo.copy()  # perfect demo prediction
        class_names = sorted(y_true_demo.astype(str).unique())
        plot_confusion_matrix_heatmap(y_true_demo, y_pred_demo, class_names, charts_dir)

        print("[INFO] All visualization charts generated successfully.")

    except FileNotFoundError as file_error:
        print(f"[ERROR] {file_error}")
    except ValueError as value_error:
        print(f"[ERROR] {value_error}")
    except Exception as error:
        print(f"[ERROR] Unexpected visualization error: {error}")


if __name__ == "__main__":
    run_visualization_pipeline()
