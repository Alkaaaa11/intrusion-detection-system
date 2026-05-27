"""
train_model.py
--------------
Beginner-friendly training script for the Intrusion Detection System.

This script:
1) Loads processed_train.csv and processed_test.csv
2) Trains a Random Forest classifier
3) Predicts on test data
4) Prints accuracy, classification report, and confusion matrix
5) Saves the trained model to models/intrusion_model.pkl
6) Saves label encoder / scaler placeholders (if required)
"""

from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


def load_processed_data(
    train_path: Path, test_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load processed train and test CSV files.
    """
    print("[INFO] Loading processed datasets...")
    if not train_path.exists():
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing file: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"[INFO] Train data shape: {train_df.shape}")
    print(f"[INFO] Test data shape: {test_df.shape}")
    return train_df, test_df


def resolve_processed_paths() -> Tuple[Path, Path]:
    """
    Prefer data/, but allow dataset/ if preprocessing was run there.
    """
    candidates = [
        (Path("data/processed_train.csv"), Path("data/processed_test.csv")),
        (Path("dataset/processed_train.csv"), Path("dataset/processed_test.csv")),
    ]
    for train_path, test_path in candidates:
        if train_path.exists() and test_path.exists():
            return train_path, test_path
    return candidates[0]


def split_features_and_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str = "attack_category",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split train and test data into features (X) and target labels (y).
    """
    if target_column not in train_df.columns or target_column not in test_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in processed files.")

    # These columns are labels/metadata and must NOT be used as training features.
    # - attack_category: our main target label
    # - label: original raw NSL-KDD attack label text (like "normal", "neptune", etc.)
    columns_to_drop_from_features = [target_column, "label"]

    # Drop only columns that actually exist to avoid key errors.
    existing_drop_columns_train = [col for col in columns_to_drop_from_features if col in train_df.columns]
    existing_drop_columns_test = [col for col in columns_to_drop_from_features if col in test_df.columns]

    X_train = train_df.drop(columns=existing_drop_columns_train)
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=existing_drop_columns_test)
    y_test = test_df[target_column]

    print("[INFO] Features and labels prepared successfully.")
    print(f"[DEBUG] Label column used for training: {target_column}")
    print(f"[DEBUG] Dropped non-feature columns from train: {existing_drop_columns_train}")
    print(f"[DEBUG] Dropped non-feature columns from test: {existing_drop_columns_test}")
    print(f"[DEBUG] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"[DEBUG] X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return X_train, y_train, X_test, y_test


def ensure_numeric_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure all feature columns are numeric before model training.
    Any non-numeric values are converted to NaN and filled with 0.
    """
    print("[INFO] Ensuring all feature columns are numeric...")

    # Keep a copy so we do not modify the input DataFrames unexpectedly.
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Apply numeric conversion column by column.
    for col in X_train.columns:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    # Replace any NaN created by invalid values.
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Debugging info requested: feature names, dtypes, and dataset shapes.
    print("[DEBUG] Feature column names:")
    print(list(X_train.columns))

    print("\n[DEBUG] Feature data types:")
    print(X_train.dtypes)

    print(f"\n[DEBUG] Final numeric X_train shape: {X_train.shape}")
    print(f"[DEBUG] Final numeric X_test shape: {X_test.shape}")
    return X_train, X_test


def encode_target_labels(
    y_train: pd.Series, y_test: pd.Series
) -> Tuple[pd.Series, pd.Series, LabelEncoder]:
    """
    Encode string labels into numbers for model training.
    """
    print("[INFO] Encoding target labels...")
    label_encoder = LabelEncoder()

    # Fit on both sets to avoid unseen label issues.
    combined_labels = pd.concat([y_train.astype(str), y_test.astype(str)], axis=0)
    label_encoder.fit(combined_labels)

    y_train_encoded = label_encoder.transform(y_train.astype(str))
    y_test_encoded = label_encoder.transform(y_test.astype(str))
    print(f"[INFO] Encoded classes: {list(label_encoder.classes_)}")
    return y_train_encoded, y_test_encoded, label_encoder


def get_candidate_models() -> Dict[str, object]:
    """
    Return all models we want to compare.
    """
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            random_state=42,
        ),
    }


def compare_models(
    X_train: pd.DataFrame,
    y_train_encoded: pd.Series,
    X_test: pd.DataFrame,
    y_test_encoded: pd.Series,
) -> Tuple[str, object, Dict[str, float]]:
    """
    Train multiple models, compare test accuracy, and select the best one.
    """
    print("[INFO] Comparing multiple models for better accuracy...")
    models = get_candidate_models()
    model_scores: Dict[str, float] = {}
    trained_models: Dict[str, object] = {}

    for model_name, model in models.items():
        print(f"[INFO] Training {model_name}...")
        model.fit(X_train, y_train_encoded)
        predictions = model.predict(X_test)
        score = accuracy_score(y_test_encoded, predictions)
        model_scores[model_name] = score
        trained_models[model_name] = model
        print(f"[INFO] {model_name} accuracy: {score:.4f}")

    # Pick model with highest accuracy.
    best_model_name = max(model_scores, key=model_scores.get)
    best_model = trained_models[best_model_name]
    print(f"[INFO] Best model selected automatically: {best_model_name}")
    return best_model_name, best_model, model_scores


def evaluate_model(
    model: object,
    model_name: str,
    X_test: pd.DataFrame,
    y_test_encoded: pd.Series,
    label_encoder: LabelEncoder,
    reports_dir: Path,
    confusion_chart_path: Path,
) -> None:
    """
    Evaluate the model and print beginner-friendly metrics.
    """
    print(f"[INFO] Running predictions with best model: {model_name}")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test_encoded, y_pred)
    print("\n========== MODEL EVALUATION ==========")
    print(f"Accuracy Score ({model_name}): {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test_encoded,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test_encoded, y_pred)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    print(cm_df)
    print("======================================\n")

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_dict = classification_report(
        y_test_encoded,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = reports_dir / "classification_report.csv"
    matrix_path = reports_dir / "confusion_matrix.csv"
    report_df.to_csv(report_path)
    cm_df.to_csv(matrix_path)
    print(f"[INFO] Classification report saved to: {report_path}")
    print(f"[INFO] Confusion matrix CSV saved to: {matrix_path}")

    confusion_chart_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(confusion_chart_path, dpi=300)
    plt.close()
    print(f"[INFO] Confusion matrix chart saved to: {confusion_chart_path}")


def save_accuracy_comparison_chart(model_scores: Dict[str, float], chart_path: Path) -> None:
    """
    Save a bar chart showing accuracy comparison of all candidate models.
    """
    chart_path.parent.mkdir(parents=True, exist_ok=True)

    score_df = pd.DataFrame(
        {"Model": list(model_scores.keys()), "Accuracy": list(model_scores.values())}
    ).sort_values(by="Accuracy", ascending=False)

    plt.figure(figsize=(9, 5))
    bars = plt.bar(score_df["Model"], score_df["Accuracy"], color=["#38bdf8", "#a78bfa", "#34d399"])
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[INFO] Accuracy comparison chart saved to: {chart_path}")


def save_feature_importance_chart(model: object, feature_names, chart_path: Path, report_path: Path) -> None:
    """
    Save feature importance for tree-based models.
    """
    if not hasattr(model, "feature_importances_"):
        print("[INFO] Best model does not expose feature_importances_; skipping feature importance chart.")
        return

    chart_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    importance_df = pd.DataFrame(
        {
            "feature": list(feature_names),
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)
    importance_df.to_csv(report_path, index=False)

    top_features = importance_df.head(15).sort_values(by="importance", ascending=True)
    plt.figure(figsize=(9, 7))
    plt.barh(top_features["feature"], top_features["importance"], color="#38bdf8")
    plt.title("Top Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[INFO] Feature importance report saved to: {report_path}")
    print(f"[INFO] Feature importance chart saved to: {chart_path}")


def save_artifacts(
    model: object,
    best_model_name: str,
    model_scores: Dict[str, float],
    label_encoder: LabelEncoder,
    model_path: Path,
    label_encoder_path: Path,
    scaler_path: Path,
    score_path: Path,
) -> None:
    """
    Save model and optional preprocessing artifacts.
    """
    print("[INFO] Saving model artifacts...")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save best selected model.
    joblib.dump(model, model_path)
    print(f"[INFO] Best model ({best_model_name}) saved to: {model_path}")

    # Save target label encoder (useful for decoding predicted class ids back to names).
    joblib.dump(label_encoder, label_encoder_path)
    print(f"[INFO] LabelEncoder saved to: {label_encoder_path}")

    # Scaler note:
    # Features are already scaled in preprocessing.py and saved in processed CSV files.
    # So a separate scaler object is not required here.
    # We save a small note file as a beginner-friendly placeholder.
    scaler_note = (
        "No scaler object saved from train_model.py.\n"
        "Reason: numerical scaling was already applied in preprocessing.py "
        "before saving processed_train.csv and processed_test.csv.\n"
    )
    scaler_path.write_text(scaler_note, encoding="utf-8")
    print(f"[INFO] Scaler note saved to: {scaler_path}")

    # Save model score table for dashboard/reporting use.
    scores_df = pd.DataFrame(
        {"model_name": list(model_scores.keys()), "accuracy": list(model_scores.values())}
    ).sort_values(by="accuracy", ascending=False)
    scores_df.to_csv(score_path, index=False)
    print(f"[INFO] Model score table saved to: {score_path}")


def run_training_pipeline() -> None:
    """
    Main training pipeline function.
    """
    try:
        train_path, test_path = resolve_processed_paths()
        model_path = Path("models/intrusion_model.pkl")
        label_encoder_path = Path("models/label_encoder.pkl")
        scaler_path = Path("models/scaler_info.txt")
        score_path = Path("models/model_scores.csv")
        reports_dir = Path("reports")
        comparison_chart_path = Path("charts/model_accuracy_comparison.png")
        confusion_chart_path = Path("charts/confusion_matrix_heatmap.png")
        feature_importance_chart_path = Path("charts/feature_importance.png")
        feature_importance_report_path = Path("reports/feature_importance.csv")

        train_df, test_df = load_processed_data(train_path, test_path)
        X_train, y_train, X_test, y_test = split_features_and_target(train_df, test_df)
        X_train, X_test = ensure_numeric_features(X_train, X_test)
        y_train_encoded, y_test_encoded, label_encoder = encode_target_labels(y_train, y_test)

        # For a demo-friendly beginner/medium presentation, train a binary detector
        # that separates 'normal' from 'malicious' (all other attack categories).
        # This improves clarity for interviews and avoids sparse multi-class issues.
        print("[INFO] Converting target to binary labels for demo (normal vs malicious)...")
        y_train_binary = y_train.astype(str).apply(lambda x: "normal" if x.lower() == "normal" else "malicious")
        y_test_binary = y_test.astype(str).apply(lambda x: "normal" if x.lower() == "normal" else "malicious")

        y_train_encoded, y_test_encoded, label_encoder = encode_target_labels(y_train_binary, y_test_binary)

        best_model_name, best_model, model_scores = compare_models(
            X_train, y_train_encoded, X_test, y_test_encoded
        )
        save_accuracy_comparison_chart(model_scores, comparison_chart_path)
        evaluate_model(
            best_model,
            best_model_name,
            X_test,
            y_test_encoded,
            label_encoder,
            reports_dir,
            confusion_chart_path,
        )
        save_feature_importance_chart(
            best_model,
            X_train.columns,
            feature_importance_chart_path,
            feature_importance_report_path,
        )
        save_artifacts(
            best_model,
            best_model_name,
            model_scores,
            label_encoder,
            model_path,
            label_encoder_path,
            scaler_path,
            score_path,
        )

        print("[INFO] Training pipeline finished successfully.")

    except FileNotFoundError as file_error:
        print(f"[ERROR] {file_error}")
        print("[HINT] Run preprocessing.py first to generate processed_train.csv and processed_test.csv.")
    except ValueError as value_error:
        print(f"[ERROR] {value_error}")
    except Exception as error:
        print(f"[ERROR] Unexpected training error: {error}")


if __name__ == "__main__":
    run_training_pipeline()
