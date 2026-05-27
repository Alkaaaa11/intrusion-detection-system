"""
app.py
------
Modern cybersecurity-themed Streamlit dashboard for the Intrusion Detection System.

This file focuses on UX improvements:
- Dark SOC-style theme with subtle CSS
- Hero section + sidebar navigation
- Professional metric cards
- Color-coded threat severity system (Low/Medium/High/Critical)
- CSV upload with drag-and-drop styling
- Loading states (spinner + progress + status messages)
- Premium charts (matplotlib/seaborn) rendered inside Streamlit
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

MODEL_PATH = Path("models/intrusion_model.pkl")
LABEL_ENCODER_PATH = Path("models/label_encoder.pkl")
MODEL_SCORES_PATH = Path("models/model_scores.csv")


def apply_theme_css() -> None:
    """
    Apply a dark cybersecurity theme using custom CSS.
    """
    st.markdown(
        """
        <style>
        /* Base page styling */
        .stApp { background-color: #070b18; color: #e5e7eb; }
        html, body { background-color: #070b18; }
        h1, h2, h3 { color: #7dd3fc !important; }
        p { color: #cbd5e1; }

        /* Card styling */
        .soc-card {
            background: linear-gradient(180deg, rgba(18, 26, 51, 0.95), rgba(10, 16, 36, 0.95));
            border: 1px solid rgba(125, 211, 252, 0.18);
            border-radius: 16px;
            padding: 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
            overflow: hidden;
        }
        .soc-card:hover {
            border-color: rgba(125, 211, 252, 0.28);
            transition: 180ms ease;
        }

        /* Metric card */
        .metric-card {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(125, 211, 252, 0.16);
            border-radius: 16px;
            padding: 14px;
            min-height: 92px;
        }
        .metric-top {
            display: flex;
            align-items: center;
            gap: 8px;
            color: rgba(226,232,240,0.95);
            font-weight: 600;
        }
        .metric-value {
            font-size: 28px;
            margin-top: 10px;
            color: #e2e8f0;
            font-weight: 800;
            letter-spacing: 0.2px;
        }
        .metric-sub {
            color: rgba(148,163,184,0.95);
            font-size: 12px;
            margin-top: 2px;
            font-weight: 600;
        }

        /* Upload box */
        .upload-box {
            border: 1px dashed rgba(125, 211, 252, 0.35);
            background: rgba(2, 6, 23, 0.4);
            border-radius: 18px;
            padding: 18px;
        }

        /* Buttons */
        button[kind="primary"] {
            border-radius: 12px;
        }
        .stButton>button {
            border-radius: 12px;
        }

        /* Badge colors */
        .badge {
            display: inline-block;
            border-radius: 999px;
            padding: 6px 10px;
            font-weight: 800;
            font-size: 12px;
            letter-spacing: 0.2px;
        }
        .badge-low {
            background: rgba(34, 197, 94, 0.14);
            border: 1px solid rgba(34, 197, 94, 0.40);
            color: #b7f7c9;
        }
        .badge-medium {
            background: rgba(234, 179, 8, 0.14);
            border: 1px solid rgba(234, 179, 8, 0.42);
            color: #fde68a;
        }
        .badge-high {
            background: rgba(239, 68, 68, 0.16);
            border: 1px solid rgba(239, 68, 68, 0.45);
            color: #fecaca;
        }
        .badge-critical {
            background: rgba(185, 28, 28, 0.20);
            border: 1px solid rgba(185, 28, 28, 0.55);
            color: #ffb3b3;
        }

        /* Alert cards */
        .alert-card {
            border-radius: 16px;
            border: 1px solid rgba(125, 211, 252, 0.18);
            background: rgba(2, 6, 23, 0.40);
            padding: 14px;
            margin-bottom: 10px;
        }
        .alert-row {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model_artifacts(
    model_path: Path = MODEL_PATH, encoder_path: Path = LABEL_ENCODER_PATH
) -> Tuple[Optional[object], Optional[object]]:
    """
    Load trained model and label encoder (optional).
    """
    if not model_path.exists():
        return None, None

    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path) if encoder_path.exists() else None
    return model, label_encoder


def load_detection_accuracy() -> Optional[float]:
    """
    Try to read best accuracy from models/model_scores.csv.
    """
    if not MODEL_SCORES_PATH.exists():
        return None
    try:
        scores_df = pd.read_csv(MODEL_SCORES_PATH)
        if "accuracy" not in scores_df.columns:
            return None
        return float(scores_df["accuracy"].max())
    except Exception:
        return None


def compute_severity(malicious_ratio: float) -> Tuple[str, str, str]:
    """
    Decide severity level based on malicious traffic ratio.

    Returns:
        (severity_name, badge_css_class, friendly_message)
    """
    # Thresholds are simple and easy for beginners:
    # - Low <= 10%
    # - Medium <= 30%
    # - High <= 50%
    # - Critical > 50%
    if malicious_ratio <= 0.10:
        return (
            "Low",
            "badge-low",
            "System posture looks stable. Continue routine monitoring.",
        )
    if malicious_ratio <= 0.30:
        return (
            "Medium",
            "badge-medium",
            "Suspicious activity is present. Investigate quickly if it persists.",
        )
    if malicious_ratio <= 0.50:
        return (
            "High",
            "badge-high",
            "High malicious activity detected. Prioritize incident response actions.",
        )
    return (
        "Critical",
        "badge-critical",
        "Critical threat activity detected. Immediate response is recommended.",
    )


def format_timestamp(ts: Optional[datetime] = None) -> str:
    """
    Format timestamps for alert cards.
    """
    if ts is None:
        ts = datetime.now()
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def align_and_clean_uploaded_df(uploaded_df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Prepare uploaded data to match the trained model's expected numeric features.

    Important note for beginners:
    - This dashboard expects *feature-like* CSV data (ideally from preprocessing).
    - If your file contains non-numeric strings in feature columns, they will become 0.
    """
    df = uploaded_df.copy()

    # Remove possible label columns if the user uploads a labeled dataset.
    df = df.drop(columns=["attack_category", "label"], errors="ignore")

    # Align column order to the model if the model has stored feature names.
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        df = df.reindex(columns=expected, fill_value=0)

    # Convert every column to numeric safely.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace NaN values created by coercion.
    df = df.fillna(0)
    return df


def run_predictions(prepared_df: pd.DataFrame, model, label_encoder: Optional[object]) -> pd.DataFrame:
    """
    Predict normal vs malicious + attack type, and provide confidence when possible.
    """
    pred_ids = model.predict(prepared_df)

    if label_encoder is not None:
        attack_types = label_encoder.inverse_transform(pred_ids)
    else:
        attack_types = pred_ids

    # Binary classification for SOC readability.
    prediction_result = ["normal" if str(x).lower() == "normal" else "malicious" for x in attack_types]

    # Confidence for probabilistic models.
    confidence: List[Optional[float]]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(prepared_df)
        confidence = proba.max(axis=1).astype(float).tolist()
    else:
        confidence = [None] * len(prepared_df)

    return pd.DataFrame(
        {
            "prediction_result": prediction_result,
            "attack_type": attack_types,
            "confidence": confidence,
        }
    )


def compute_metrics(result_df: pd.DataFrame) -> Dict[str, object]:
    """
    Compute dashboard KPIs from prediction results.
    """
    total = len(result_df)
    malicious_count = int((result_df["prediction_result"] == "malicious").sum())
    normal_count = int((result_df["prediction_result"] == "normal").sum())
    attack_types_detected = int(
        result_df.loc[result_df["prediction_result"] == "malicious", "attack_type"].astype(str).nunique()
    )

    malicious_ratio = (malicious_count / total) if total else 0.0
    severity_name, _, _ = compute_severity(malicious_ratio)

    return {
        "total_traffic": total,
        "malicious_count": malicious_count,
        "normal_count": normal_count,
        "attack_types_detected": attack_types_detected,
        "malicious_ratio": malicious_ratio,
        "threat_severity": severity_name,
    }


def styled_metrics_cards(metrics: Dict[str, object], detection_accuracy: Optional[float]) -> None:
    """
    Render modern metric cards (SOC-like).
    """
    sev = str(metrics["threat_severity"])
    sev_icon = "🛡️" if sev == "Low" else ("🟠" if sev == "Medium" else ("🔴" if sev == "High" else "🚨"))

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-top">📡 Total Traffic</div>
          <div class="metric-value">{metrics['total_traffic']}</div>
          <div class="metric-sub">Records analyzed</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-top">🧨 Malicious Traffic</div>
          <div class="metric-value" style="color:#fecaca;">{metrics['malicious_count']}</div>
          <div class="metric-sub">Detected attacks</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col3.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-top">✅ Normal Traffic</div>
          <div class="metric-value" style="color:#bbf7d0;">{metrics['normal_count']}</div>
          <div class="metric-sub">Benign records</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col4.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-top">{sev_icon} Threat Severity</div>
          <div class="metric-value">{sev}</div>
          <div class="metric-sub">Auto-calculated from attack %</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Second row: attack types + detection accuracy
    col5, col6, col7 = st.columns([2, 1, 1])
    col5.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-top">🧭 Attack Types Detected</div>
          <div class="metric-value">{metrics['attack_types_detected']}</div>
          <div class="metric-sub">Unique attack families</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    accuracy_text = "N/A" if detection_accuracy is None else f"{detection_accuracy * 100:.2f}%"
    col6.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-top">🎯 Detection Accuracy</div>
          <div class="metric-value">{accuracy_text}</div>
          <div class="metric-sub">Best model test score</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col7.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-top">📈 Malicious Ratio</div>
          <div class="metric-value">{metrics['malicious_ratio']*100:.1f}%</div>
          <div class="metric-sub">Based on uploaded file</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_realtime_alert_banner(metrics: Dict[str, object]) -> None:
    """
    Show a prominent alert banner based on severity.
    """
    ratio = float(metrics["malicious_ratio"])
    severity_name, badge_class, message = compute_severity(ratio)

    # Show a warning-style banner when attack is detected (requirement).
    if metrics["malicious_count"] == 0:
        st.success("⚡ System Secure: no malicious traffic detected.")
    elif severity_name in ("Low", "Medium"):
        st.warning(f"⚠ Threat Detected: {severity_name} severity.")
    elif severity_name == "High":
        st.error(f"🔴 High Threat Activity detected.")
    else:
        st.error("🚨 Critical Threat Activity detected!")

    st.markdown(
        f"""
        <div class="soc-card" style="padding:14px;">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
            <div>
              <div class="badge {badge_class}">Threat Level: {severity_name}</div>
              <div style="margin-top:8px;font-weight:700;color:#e2e8f0;">{message}</div>
            </div>
            <div style="text-align:right;color:rgba(148,163,184,0.95);font-weight:700;">
              Malicious: {metrics['malicious_count']}/{metrics['total_traffic']}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_attack_summary_cards(result_df: pd.DataFrame) -> None:
    """
    Show attack summary cards for the top detected malicious attack types.
    """
    malicious_df = result_df[result_df["prediction_result"] == "malicious"]
    if malicious_df.empty:
        st.info("No malicious attack types detected in this upload.")
        return

    top_counts = malicious_df["attack_type"].astype(str).value_counts().head(4)
    st.subheader("Attack Summary")

    cols = st.columns(len(top_counts))
    for i, (attack_name, count) in enumerate(top_counts.items()):
        with cols[i]:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-top">🧨 {attack_name}</div>
                  <div class="metric-value" style="color:#fecaca;">{count}</div>
                  <div class="metric-sub">Detected records</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def show_threat_summary(metrics: Dict[str, object], result_df: pd.DataFrame) -> None:
    """
    Show a short threat summary (easy for beginners).
    """
    st.subheader("Threat Summary")
    top_attack = "N/A"

    malicious_df = result_df[result_df["prediction_result"] == "malicious"]
    if not malicious_df.empty:
        top_attack = str(malicious_df["attack_type"].astype(str).value_counts().idxmax())

    st.markdown(
        f"""
        <div class="soc-card">
          <div style="display:flex;gap:18px;flex-wrap:wrap;">
            <div style="min-width:220px;">
              <div style="font-weight:900;color:#e2e8f0;">Threat Level</div>
              <div style="margin-top:8px;font-size:20px;font-weight:900;color:#7dd3fc;">{metrics['threat_severity']}</div>
              <div style="margin-top:6px;color:rgba(148,163,184,0.95);font-weight:700;">Auto-calculated from malicious %</div>
            </div>
            <div style="min-width:220px;">
              <div style="font-weight:900;color:#e2e8f0;">Most Frequent Attack</div>
              <div style="margin-top:8px;font-size:18px;font-weight:900;color:#e2e8f0;">{top_attack}</div>
              <div style="margin-top:6px;color:rgba(148,163,184,0.95);font-weight:700;">Among malicious records</div>
            </div>
            <div style="min-width:220px;">
              <div style="font-weight:900;color:#e2e8f0;">Traffic Breakdown</div>
              <div style="margin-top:8px;font-weight:900;color:#e2e8f0;">
                Malicious: {metrics['malicious_count']} • Normal: {metrics['normal_count']}
              </div>
              <div style="margin-top:6px;color:rgba(148,163,184,0.95);font-weight:700;">
                Total: {metrics['total_traffic']}
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_confidence_summary(result_df: pd.DataFrame) -> None:
    """
    Show a small confidence summary near the prediction table.
    """
    st.subheader("Confidence Summary")
    if "confidence" not in result_df.columns:
        st.info("Confidence is not available for this model.")
        return

    conf = pd.to_numeric(result_df["confidence"], errors="coerce")
    conf = conf.dropna()
    if conf.empty:
        st.info("Confidence scores are not available for this upload.")
        return

    malicious_conf = conf[result_df["prediction_result"] == "malicious"] if "prediction_result" in result_df.columns else conf
    if malicious_conf.empty:
        st.info("No malicious rows to summarize confidence for.")
        return

    avg_conf = float(malicious_conf.mean())
    max_conf = float(malicious_conf.max())

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Average Malicious Confidence", f"{avg_conf * 100:.2f}%")
    with c2:
        st.metric("Max Malicious Confidence", f"{max_conf * 100:.2f}%")

    top_conf = (
        result_df[result_df["prediction_result"] == "malicious"][["attack_type", "confidence"]]
        .copy()
    )
    top_conf["confidence"] = pd.to_numeric(top_conf["confidence"], errors="coerce")
    top_conf = top_conf.dropna().sort_values("confidence", ascending=False).head(10)

    if not top_conf.empty:
        st.write("Top confident malicious predictions (top 10):")
        st.dataframe(top_conf, use_container_width=True)


def make_styled_predictions_table(result_df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Highlight malicious rows in the predictions table.
    """
    def highlight_row(row):
        if row["prediction_result"] == "malicious":
            return ["background-color: rgba(239, 68, 68, 0.22); color:#fecaca;"] * len(row)
        return [""] * len(row)

    # Confidence is numeric-like (0..1). Display nicely.
    df = result_df.copy()
    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    styler = df.style.apply(highlight_row, axis=1)
    return styler


def filter_predictions(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Provide basic filtering/search to make results easier to explore.
    """
    st.subheader("Predictions Explorer")
    show_only_malicious = st.checkbox("Show only malicious rows", value=False)
    search_attack = st.text_input("Filter by attack type (e.g., dos, probe)", value="")

    df = result_df.copy()
    if show_only_malicious:
        df = df[df["prediction_result"] == "malicious"]

    if search_attack.strip():
        term = search_attack.strip().lower()
        df = df[df["attack_type"].astype(str).str.lower().str.contains(term, na=False)]

    return df


def set_matplotlib_dark_style() -> None:
    """
    Apply a dark theme to matplotlib charts so they match the Streamlit UI.
    """
    plt.style.use("dark_background")
    sns.set_theme(style="darkgrid")


def plot_donut_attack_distribution(result_df: pd.DataFrame) -> plt.Figure:
    """
    Donut chart: normal vs malicious distribution.
    """
    set_matplotlib_dark_style()
    counts = result_df["prediction_result"].value_counts()
    labels = ["Normal", "Malicious"]
    values = [int(counts.get("normal", 0)), int(counts.get("malicious", 0))]

    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor="#070b18")
    colors = ["#22c55e", "#ef4444"]
    ax.pie(
        values,
        labels=labels,
        autopct="%1.0f%%",
        startangle=90,
        colors=colors,
        wedgeprops={"width": 0.35, "edgecolor": "#070b18"},
        textprops={"color": "white"},
    )
    ax.set_title("Attack Distribution (Normal vs Malicious)", color="#e2e8f0", pad=14)
    return fig


def plot_attack_type_bar(result_df: pd.DataFrame) -> plt.Figure:
    """
    Bar chart of detected attack categories (malicious rows only).
    """
    set_matplotlib_dark_style()
    malicious_df = result_df[result_df["prediction_result"] == "malicious"]
    counts = malicious_df["attack_type"].astype(str).value_counts().head(12)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor="#070b18")
    if counts.empty:
        ax.text(0.5, 0.5, "No attacks detected", ha="center", va="center", color="#cbd5e1")
        ax.set_axis_off()
        return fig

    sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="Reds")
    ax.set_title("Detected Attack Types (Top Categories)", color="#e2e8f0", pad=14)
    ax.set_xlabel("Number of Records", color="#cbd5e1")
    ax.set_ylabel("Attack Type", color="#cbd5e1")
    ax.legend([], [], frameon=False)
    return fig


def plot_threat_severity_graph(metrics: Dict[str, object]) -> plt.Figure:
    """
    Threat severity graph: visualize malicious ratio relative to thresholds.
    """
    set_matplotlib_dark_style()
    ratio = float(metrics["malicious_ratio"])
    severity_name, badge_class, _ = compute_severity(ratio)

    # Thresholds and colors.
    thresholds = [0.10, 0.30, 0.50, 1.0]
    colors = ["#22c55e", "#eab308", "#ef4444", "#b91c1c"]
    labels = ["Low", "Medium", "High", "Critical"]

    fig, ax = plt.subplots(figsize=(7.5, 2.6), facecolor="#070b18")
    ax.set_title("Threat Severity Level (Auto-Calculated)", color="#e2e8f0", pad=12)
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Malicious Traffic Ratio", color="#cbd5e1")

    # Draw colored segments.
    start = 0.0
    for idx, end in enumerate(thresholds):
        ax.barh(
            y=0,
            width=end - start,
            left=start,
            height=0.22,
            color=colors[idx],
            alpha=0.8,
        )
        start = end

    # Marker for current ratio.
    ax.scatter([ratio], [0], color="white", s=120, zorder=3, marker="D")
    ax.text(ratio, 0.32, f"{severity_name} ({ratio*100:.1f}%)", ha="center", color="#e2e8f0", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def plot_confidence_histogram(result_df: pd.DataFrame) -> plt.Figure:
    """
    Confidence histogram for model predictions.
    """
    set_matplotlib_dark_style()
    conf = pd.to_numeric(result_df["confidence"], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor="#070b18")
    if conf.empty:
        ax.text(0.5, 0.5, "Confidence not available for this model", ha="center", va="center", color="#cbd5e1")
        ax.set_axis_off()
        return fig

    sns.histplot(conf, bins=20, kde=False, ax=ax, color="#7dd3fc")
    ax.set_title("Confidence Score Histogram", color="#e2e8f0", pad=14)
    ax.set_xlabel("Confidence (0 to 1)", color="#cbd5e1")
    ax.set_ylabel("Count", color="#cbd5e1")
    return fig


def plot_traffic_activity(result_df: pd.DataFrame) -> plt.Figure:
    """
    Traffic activity visualization: cumulative malicious ratio over rows.
    """
    set_matplotlib_dark_style()
    malicious_flags = (result_df["prediction_result"] == "malicious").astype(int).values
    cumulative = np.cumsum(malicious_flags) / np.arange(1, len(malicious_flags) + 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor="#070b18")
    ax.plot(cumulative, color="#f59e0b", linewidth=2, label="Cumulative Malicious Ratio")
    ax.axhline(0.1, color="#22c55e", linestyle="--", linewidth=1, alpha=0.8, label="Low threshold (10%)")
    ax.axhline(0.3, color="#eab308", linestyle="--", linewidth=1, alpha=0.8, label="Medium threshold (30%)")
    ax.axhline(0.5, color="#ef4444", linestyle="--", linewidth=1, alpha=0.8, label="High threshold (50%)")

    ax.set_title("Traffic Activity (Cumulative Malicious Ratio)", color="#e2e8f0", pad=14)
    ax.set_xlabel("Row Index", color="#cbd5e1")
    ax.set_ylabel("Cumulative Malicious Ratio", color="#cbd5e1")
    ax.legend(loc="best", frameon=False)
    return fig


def show_charts(result_df: pd.DataFrame, metrics: Dict[str, object]) -> None:
    """
    Render all required upgraded charts in a clean responsive layout.
    """
    st.subheader("Cyber Attack Analytics")

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_donut_attack_distribution(result_df), use_container_width=True)
    with c2:
        st.pyplot(plot_attack_type_bar(result_df), use_container_width=True)

    st.pyplot(plot_threat_severity_graph(metrics), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.pyplot(plot_confidence_histogram(result_df), use_container_width=True)
    with c4:
        st.pyplot(plot_traffic_activity(result_df), use_container_width=True)


def build_realtime_alert_center(
    result_df: pd.DataFrame, metrics: Dict[str, object], max_alerts: int = 6
) -> List[Dict[str, str]]:
    """
    Build a list of alert cards with timestamps.
    """
    sev = compute_severity(float(metrics["malicious_ratio"]))[0]
    ts = format_timestamp(datetime.now())

    malicious_df = result_df[result_df["prediction_result"] == "malicious"]
    if malicious_df.empty:
        return [
            {
                "severity": "Low",
                "title": "System Secure",
                "message": "No malicious records detected in the uploaded file.",
                "timestamp": ts,
            }
        ]

    top_rows = malicious_df.head(max_alerts)
    alerts: List[Dict[str, str]] = []
    for _, row in top_rows.iterrows():
        attack_name = str(row["attack_type"])
        conf = row["confidence"]
        conf_text = "N/A" if pd.isna(conf) else f"{float(conf)*100:.1f}%"
        alerts.append(
            {
                "severity": sev,
                "title": f"Threat Detected: {attack_name}",
                "message": f"Malicious traffic signature matched. Confidence: {conf_text}.",
                "timestamp": ts,
            }
        )
    return alerts


def show_realtime_alert_cards(alerts: List[Dict[str, str]]) -> None:
    """
    Render alert cards (color-coded via badge CSS).
    """
    st.subheader("Realtime Alert Center")

    for alert in alerts:
        sev = alert["severity"]
        if sev == "Low":
            badge_class = "badge-low"
        elif sev == "Medium":
            badge_class = "badge-medium"
        elif sev == "High":
            badge_class = "badge-high"
        else:
            badge_class = "badge-critical"

        # Animate appearance a bit (small delay). Keep it short for performance.
        time.sleep(0.05)
        st.markdown(
            f"""
            <div class="alert-card">
              <div class="alert-row">
                <div>
                  <span class="badge {badge_class}">{sev}</span>
                  <div style="margin-top:8px;font-weight:900;color:#e2e8f0;">{alert['title']}</div>
                  <div style="margin-top:6px;color:#cbd5e1;">{alert['message']}</div>
                </div>
                <div style="text-align:right;color:rgba(148,163,184,0.95);font-weight:700;">
                  {alert['timestamp']}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_hero_section() -> None:
    """
    Render the top hero section (title + tagline + live monitoring badge).
    """
    last_metrics = st.session_state.get("last_metrics")
    if not last_metrics:
        threat_status = "Awaiting Upload"
        threat_ratio_text = ""
        threat_badge_class = "badge-low"
    else:
        threat_sev = str(last_metrics.get("threat_severity", "Low"))
        threat_status = f"{threat_sev} Threat"
        threat_ratio = float(last_metrics.get("malicious_ratio", 0.0)) * 100.0
        threat_ratio_text = f" • Malicious: {threat_ratio:.1f}%"
        if threat_sev == "Low":
            threat_badge_class = "badge-low"
        elif threat_sev == "Medium":
            threat_badge_class = "badge-medium"
        elif threat_sev == "High":
            threat_badge_class = "badge-high"
        else:
            threat_badge_class = "badge-critical"

    st.markdown(
        f"""
        <div class="soc-card" style="padding:20px;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:18px;flex-wrap:wrap;">
            <div>
              <h1 style="margin:0;color:#7dd3fc;font-size:30px;">AI-Powered Intrusion Detection</h1>
              <p style="margin:10px 0 0 0;font-size:15px;">
                Real-Time Intrusion Monitoring • Threat Classification • Security Analytics
              </p>
              <p style="margin:10px 0 0 0;color:rgba(148,163,184,0.95);font-weight:600;">
                AI detects malicious network traffic patterns and helps analysts respond faster.
              </p>
            </div>
            <div style="min-width:260px;">
              <span class="badge badge-low">● Live Monitoring</span>
              <div style="margin-top:12px;color:#e2e8f0;font-weight:800;">
                SOC Dashboard Mode
              </div>
              <div style="margin-top:6px;color:rgba(148,163,184,0.95);font-weight:700;">
                Powered by Machine Learning
              </div>
              <div style="margin-top:12px;">
                <span class="badge {threat_badge_class}">Threat Status: {threat_status}{threat_ratio_text}</span>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_section() -> Optional[pd.DataFrame]:
    """
    Render the file upload UI and return loaded DataFrame.
    """
    st.subheader("Upload Network Traffic (CSV)")

    st.markdown(
        """
        <div class="upload-box">
          <div style="font-weight:900;color:#e2e8f0;font-size:16px;">
            📥 Drag-and-drop your CSV file
          </div>
          <div style="margin-top:6px;color:#cbd5e1;">
            Tip: For best results, upload the <b>processed</b> CSV generated by <b>preprocessing.py</b>.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload traffic data for AI threat detection.",
    )
    if uploaded_file is None:
        return None

    try:
        # Try standard decoding first, then safe fallback.
        try:
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding="latin1", engine="python", on_bad_lines="skip")

        if df.empty:
            st.error("Uploaded CSV is empty. Please upload a valid file.")
            return None

        st.write("Uploaded file preview:")
        st.dataframe(df.head(10), use_container_width=True)
        return df
    except Exception as e:
        st.error(f"Could not load uploaded CSV: {e}")
        return None


def show_loading_and_predict(uploaded_df: pd.DataFrame, model, label_encoder: Optional[object]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Show loading states and run the prediction pipeline.
    """
    status_box = st.empty()
    progress_bar = st.progress(0)

    # Step 1
    status_box.info("Analyzing Network Traffic…")
    progress_bar.progress(20)
    time.sleep(0.2)

    # Step 2
    status_box.info("Scanning for Threats…")
    progress_bar.progress(45)
    time.sleep(0.2)

    # Step 3
    status_box.info("Generating Threat Intelligence…")
    progress_bar.progress(70)
    time.sleep(0.2)

    # Run actual prediction
    prepared_df = align_and_clean_uploaded_df(uploaded_df, model)
    result_df = run_predictions(prepared_df, model, label_encoder)
    metrics = compute_metrics(result_df)

    progress_bar.progress(100)
    status_box.success("Threat analysis completed.")
    # Store latest metrics so the hero section can show live threat status.
    st.session_state["last_metrics"] = metrics
    return result_df, metrics


def show_footer() -> None:
    """
    Render a clean professional footer.
    """
    st.markdown(
        """
        <div style="margin-top:20px;padding:16px;border-top:1px solid rgba(125,211,252,0.18);">
          <div style="font-weight:900;color:#e2e8f0;">Intrusion Detection System</div>
          <div style="color:rgba(148,163,184,0.95);margin-top:6px;font-weight:700;">
            Cybersecurity ML • Powered by Machine Learning • Built with Python + Scikit-learn + Streamlit
          </div>
          <div style="color:rgba(148,163,184,0.95);margin-top:6px;">
            Creator: Your Name Here
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """
    Main dashboard entry point.
    """
    st.set_page_config(page_title="Cyber IDS Dashboard", layout="wide")
    apply_theme_css()
    sns.set_theme()

    show_hero_section()

    # Sidebar navigation with icons (minimal and professional).
    st.sidebar.title("Navigation")
    page_display_to_key = {
        "🛡️ Dashboard": "Dashboard",
        "🔎 Threat Analysis": "Threat Analysis",
        "📄 Predictions": "Predictions",
        "📊 Visualizations": "Visualizations",
        "🧾 Reports": "Reports",
        "ℹ️ About Project": "About Project",
    }
    selected_page = st.sidebar.radio("Select Section", list(page_display_to_key.keys()), index=0)
    page = page_display_to_key[selected_page]

    if page == "About Project":
        st.subheader("About Project")
        st.write(
            "This project demonstrates how machine learning can support an intrusion detection workflow "
            "using the NSL-KDD dataset. The dashboard provides alerts, severity scoring, and attack analytics "
            "on uploaded traffic CSV files."
        )
        st.info("For best accuracy, upload processed CSV files generated by this project’s preprocessing step.")
        show_footer()
        return

    if page == "Dashboard":
        model, label_encoder = load_model_artifacts()
        if model is None:
            st.error("Model not found. Run `train_model.py` first.")
            return

        detection_accuracy = load_detection_accuracy()

        uploaded_df = render_upload_section()
        if uploaded_df is None:
            show_footer()
            return

        if st.button("Run Intrusion Detection", type="primary"):
            with st.spinner("Running AI analysis..."):
                result_df, metrics = show_loading_and_predict(uploaded_df, model, label_encoder)

            # Metrics + alerts + summaries
            st.subheader("Security Metrics")
            styled_metrics_cards(metrics, detection_accuracy)
            show_realtime_alert_banner(metrics)
            build_attack_summary_cards(result_df)
            show_threat_summary(metrics, result_df)

            # Predictions table
            st.subheader("Prediction Results")
            filtered_df = result_df.copy()
            # Basic highlight in table (malicious rows in red)
            try:
                st.dataframe(make_styled_predictions_table(filtered_df), use_container_width=True)
            except Exception:
                st.dataframe(filtered_df, use_container_width=True)

            show_confidence_summary(result_df)

            show_charts(result_df, metrics)
            alerts = build_realtime_alert_center(result_df, metrics)
            show_realtime_alert_cards(alerts)

    elif page == "Threat Analysis":
        # A dedicated view that focuses on severity + alerts + high-level stats.
        model, label_encoder = load_model_artifacts()
        if model is None:
            st.error("Model not found. Run `train_model.py` first.")
            return

        detection_accuracy = load_detection_accuracy()
        uploaded_df = render_upload_section()
        if uploaded_df is None:
            show_footer()
            return

        if st.button("Run Threat Analysis", type="primary"):
            result_df, metrics = show_loading_and_predict(uploaded_df, model, label_encoder)
            styled_metrics_cards(metrics, detection_accuracy)
            show_realtime_alert_banner(metrics)
            show_charts(result_df, metrics)
            alerts = build_realtime_alert_center(result_df, metrics)
            show_realtime_alert_cards(alerts)

    elif page == "Predictions":
        model, label_encoder = load_model_artifacts()
        if model is None:
            st.error("Model not found. Run `train_model.py` first.")
            return

        uploaded_df = render_upload_section()
        if uploaded_df is None:
            show_footer()
            return

        if st.button("Predict Attacks", type="primary"):
            result_df, metrics = show_loading_and_predict(uploaded_df, model, label_encoder)

            # Search/filter UI
            filtered_df = filter_predictions(result_df)
            st.subheader("Filtered Predictions")
            try:
                st.dataframe(make_styled_predictions_table(filtered_df), use_container_width=True)
            except Exception:
                st.dataframe(filtered_df, use_container_width=True)

            st.subheader("Threat Summary")
            st.write(f"Total analyzed records: {metrics['total_traffic']}")
            st.write(f"Malicious records: {metrics['malicious_count']}")
            st.write(f"Threat severity: {metrics['threat_severity']}")

    elif page == "Visualizations":
        model, label_encoder = load_model_artifacts()
        if model is None:
            st.error("Model not found. Run `train_model.py` first.")
            return

        uploaded_df = render_upload_section()
        if uploaded_df is None:
            show_footer()
            return

        if st.button("Generate Analytics", type="primary"):
            result_df, metrics = show_loading_and_predict(uploaded_df, model, label_encoder)
            show_charts(result_df, metrics)

    elif page == "Reports":
        # Quick report-style summary suitable for screenshots/demos.
        model, label_encoder = load_model_artifacts()
        if model is None:
            st.error("Model not found. Run `train_model.py` first.")
            return

        uploaded_df = render_upload_section()
        if uploaded_df is None:
            show_footer()
            return

        if st.button("Create Threat Report", type="primary"):
            result_df, metrics = show_loading_and_predict(uploaded_df, model, label_encoder)
            st.subheader("Threat Report (Summary)")
            styled_metrics_cards(metrics, load_detection_accuracy())
            show_realtime_alert_banner(metrics)

            malicious_df = result_df[result_df["prediction_result"] == "malicious"]
            if malicious_df.empty:
                st.success("Report Outcome: No malicious threats detected.")
            else:
                st.warning("Report Outcome: Malicious threats detected.")
                st.write("Top detected attack types:")
                top_attack_counts = (
                    malicious_df["attack_type"].astype(str).value_counts().head(8).reset_index()
                )
                top_attack_counts.columns = ["attack_type", "count"]
                st.dataframe(top_attack_counts, use_container_width=True)

            alerts = build_realtime_alert_center(result_df, metrics)
            show_realtime_alert_cards(alerts)

            st.subheader("Recommendations (Beginner Friendly)")
            st.write(
                "1. If severity is Medium/High/Critical, check recent logs from firewalls and IDS sensors.\n"
                "2. Validate the source IPs for unusual scanning or login attempts.\n"
                "3. If repeated detections occur, update detection rules and consider blocking suspicious traffic."
            )

    # Footer always visible when user is done
    show_footer()


if __name__ == "__main__":
    main()
