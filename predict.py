"""
predict.py
----------
Beginner-friendly prediction script for intrusion detection.

This script:
1) Loads trained model from models/
2) Loads sample network traffic data
3) Predicts normal traffic or malicious attack
4) Shows attack type and confidence score (if model supports probabilities)
"""

from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd


def load_model_and_encoder(
    model_path: Path, encoder_path: Path
) -> Tuple[object, Optional[object]]:
    """
    Load trained model and optional label encoder from disk.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"[INFO] Loading model from: {model_path}")
    model = joblib.load(model_path)

    label_encoder = None
    if encoder_path.exists():
        print(f"[INFO] Loading label encoder from: {encoder_path}")
        label_encoder = joblib.load(encoder_path)
    else:
        print("[WARN] Label encoder not found. Predictions will stay numeric if encoded.")

    return model, label_encoder


def load_sample_data(sample_data_path: Path, sample_size: int = 5) -> pd.DataFrame:
    """
    Load sample rows from processed test data for quick prediction demo.
    """
    if not sample_data_path.exists():
        raise FileNotFoundError(f"Sample data file not found: {sample_data_path}")

    print(f"[INFO] Loading sample data from: {sample_data_path}")
    data = pd.read_csv(sample_data_path)
    print(f"[INFO] Total rows available: {len(data)}")

    # Remove label columns from features, because model expects only feature columns.
    feature_df = data.drop(columns=["attack_category", "label"], errors="ignore")
    sample_df = feature_df.head(sample_size).copy()

    # Safety step: ensure features are numeric for model input.
    for col in sample_df.columns:
        sample_df[col] = pd.to_numeric(sample_df[col], errors="coerce")
    sample_df = sample_df.fillna(0)

    print(f"[INFO] Using {len(sample_df)} sample rows for prediction.")
    return sample_df


def resolve_sample_data_path() -> Path:
    """
    Prefer data/, but allow dataset/ if preprocessing was run there.
    """
    candidates = [Path("data/processed_test.csv"), Path("dataset/processed_test.csv")]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def decode_attack_types(predictions, label_encoder: Optional[object]):
    """
    Convert numeric class ids to readable attack type names (if encoder is available).
    """
    if label_encoder is None:
        return predictions
    return label_encoder.inverse_transform(predictions)


def predict_with_confidence(model, sample_df: pd.DataFrame) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    Predict attack class and confidence score.
    """
    # Predict class IDs or class labels.
    predictions = model.predict(sample_df)

    # Confidence is available when model supports predict_proba.
    confidence_scores = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample_df)
        confidence_scores = probabilities.max(axis=1)

    return predictions, confidence_scores


def build_prediction_output(
    sample_df: pd.DataFrame,
    attack_types,
    confidence_scores: Optional[pd.Series],
) -> pd.DataFrame:
    """
    Build a clean, readable output table.
    """
    output = sample_df.copy()
    output["attack_type"] = attack_types

    # Convert to simple binary traffic category.
    output["prediction_result"] = output["attack_type"].apply(
        lambda x: "Normal Traffic" if str(x).lower() == "normal" else "Malicious Attack"
    )

    if confidence_scores is not None:
        output["confidence"] = [f"{score * 100:.2f}%" for score in confidence_scores]
    else:
        output["confidence"] = "N/A"

    return output


def run_prediction_pipeline() -> None:
    """
    Main function to run full prediction workflow.
    """
    try:
        model_path = Path("models/intrusion_model.pkl")
        encoder_path = Path("models/label_encoder.pkl")
        sample_data_path = resolve_sample_data_path()

        model, label_encoder = load_model_and_encoder(model_path, encoder_path)
        sample_df = load_sample_data(sample_data_path, sample_size=5)

        raw_predictions, confidence_scores = predict_with_confidence(model, sample_df)
        attack_types = decode_attack_types(raw_predictions, label_encoder)

        result_df = build_prediction_output(sample_df, attack_types, confidence_scores)

        print("\n========== PREDICTION RESULTS ==========")
        print(result_df[["prediction_result", "attack_type", "confidence"]])
        print("========================================\n")

    except FileNotFoundError as file_error:
        print(f"[ERROR] {file_error}")
        print("[HINT] Run preprocessing.py and train_model.py first.")
    except Exception as error:
        print(f"[ERROR] Prediction failed: {error}")


if __name__ == "__main__":
    run_prediction_pipeline()
