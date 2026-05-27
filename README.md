# Intrusion Detection System Using Machine Learning

A beginner-to-medium cybersecurity machine learning project that uses the NSL-KDD dataset to classify network traffic as `normal`, `dos`, `probe`, `r2l`, or `u2r`, then displays predictions in a Streamlit SOC-style dashboard.

## What This Project Includes

- NSL-KDD preprocessing with categorical encoding and numerical scaling
- Reusable saved preprocessing artifacts
- Multi-model training with Random Forest, Decision Tree, and Logistic Regression
- Automatic best-model selection by test accuracy
- Accuracy, precision, recall, F1-score, and confusion matrix reports
- Feature importance report for tree-based models
- CSV upload prediction from the Streamlit dashboard
- Manual single-record prediction from the dashboard
- Threat severity scoring and color-coded alerts
- Local prediction history logging
- Dashboard charts for traffic breakdown, attack types, confidence, and threat activity

## Project Structure

```text
intrusion-detection-system/
|-- app.py
|-- download_dataset.py
|-- preprocessing.py
|-- train_model.py
|-- predict.py
|-- visualize.py
|-- requirements.txt
|-- README.md
|-- charts/
|-- data/
|-- dataset/
|-- logs/
|-- models/
|-- reports/
`-- utils/
```

`data/`, `dataset/`, `models/`, `charts/`, `logs/`, and `reports/` contain `.gitkeep` placeholders. Large datasets, trained models, generated charts, and logs are created locally and are not committed.

## Dataset

This project expects the NSL-KDD files:

```text
KDDTrain+.txt
KDDTest+.txt
```

Place them in either:

```text
data/
```

or:

```text
dataset/
```

The scripts prefer `data/`, but `preprocessing.py` can also detect raw files in `dataset/`.

You can also try the included downloader:

```bash
python download_dataset.py
```

It attempts Zenodo first and then a GitHub mirror. If your network is slow, rerun the same command; it resumes partial downloads.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## How To Run

Run these commands from the project root.

### 1. Download Or Add Dataset

```bash
python download_dataset.py
```

If the downloader fails on your network, manually download `KDDTrain+.txt` and `KDDTest+.txt` and place them in `data/`.

### 2. Preprocess Data

```bash
python preprocessing.py
```

Generated outputs:

- `data/processed_train.csv` or `dataset/processed_train.csv`
- `data/processed_test.csv` or `dataset/processed_test.csv`
- `models/preprocessing_artifacts.pkl`

### 3. Train Models

```bash
python train_model.py
```

Generated outputs:

- `models/intrusion_model.pkl`
- `models/label_encoder.pkl`
- `models/model_scores.csv`
- `reports/classification_report.csv`
- `reports/confusion_matrix.csv`
- `reports/feature_importance.csv` when supported
- `charts/model_accuracy_comparison.png`
- `charts/confusion_matrix_heatmap.png`
- `charts/feature_importance.png` when supported

### 4. Run CLI Prediction

```bash
python predict.py
```

This predicts a few rows from the processed test dataset.

### 5. Generate Static Charts

```bash
python visualize.py
```

This creates dataset and evaluation charts when processed data and reports are available.

### 6. Launch Dashboard

```bash
streamlit run app.py
```

Dashboard features:

- Sidebar navigation
- CSV upload prediction
- Manual traffic prediction
- Threat severity alerts
- Confidence scores when supported by the model
- Attack distribution charts
- Local logs in `logs/attack_history.csv`
- Latest summary report in `reports/latest_threat_report.csv`

## Attack Categories

| Category | Meaning | Examples |
|---|---|---|
| `normal` | Legitimate traffic | Standard web or file traffic |
| `dos` | Denial of Service | neptune, smurf, teardrop |
| `probe` | Reconnaissance/scanning | nmap, ipsweep, portsweep |
| `r2l` | Remote to Local attack | guess_passwd, ftp_write |
| `u2r` | User to Root escalation | buffer_overflow, rootkit |

## Current Limitations

- This is a learning project, not a production IDS.
- It does not capture live packets from an interface.
- It does not include authentication or role-based access control.
- SHAP/LIME explanations are not implemented yet.
- Manual input is simplified and uses defaults for unshown NSL-KDD features.

## Suggested Future Improvements

- Add SHAP explanations for individual predictions
- Add Scapy or PyShark for real packet capture
- Add alert export to PDF
- Add authentication to the dashboard
- Add XGBoost, SVM, and class-imbalance handling
- Add unit tests for preprocessing and prediction
- Deploy the dashboard on Streamlit Cloud

