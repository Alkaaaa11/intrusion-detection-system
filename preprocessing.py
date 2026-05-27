"""
preprocessing.py
----------------
Beginner-friendly preprocessing script for the NSL-KDD intrusion detection dataset.

What this script does:
1) Loads KDDTrain+.txt and KDDTest+.txt from data/
2) Adds official NSL-KDD column names
3) Cleans and preprocesses data
4) Encodes categorical columns using LabelEncoder
5) Normalizes numerical columns using StandardScaler
6) Maps fine-grained attack labels to attack categories
7) Splits features (X) and target (y)
8) Saves processed files to data/processed_train.csv and data/processed_test.csv
"""

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Official NSL-KDD columns (41 features + label + difficulty_level)
NSL_KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty_level",
]

# Attack mapping: raw attack name -> broad category
ATTACK_CATEGORY_MAP = {
    # Normal
    "normal": "normal",
    # DoS
    "back": "dos",
    "land": "dos",
    "neptune": "dos",
    "pod": "dos",
    "smurf": "dos",
    "teardrop": "dos",
    "mailbomb": "dos",
    "apache2": "dos",
    "processtable": "dos",
    "udpstorm": "dos",
    # Probe
    "satan": "probe",
    "ipsweep": "probe",
    "nmap": "probe",
    "portsweep": "probe",
    "mscan": "probe",
    "saint": "probe",
    # R2L
    "guess_passwd": "r2l",
    "ftp_write": "r2l",
    "imap": "r2l",
    "phf": "r2l",
    "multihop": "r2l",
    "warezmaster": "r2l",
    "warezclient": "r2l",
    "spy": "r2l",
    "xlock": "r2l",
    "xsnoop": "r2l",
    "snmpguess": "r2l",
    "snmpgetattack": "r2l",
    "httptunnel": "r2l",
    "sendmail": "r2l",
    "named": "r2l",
    # U2R
    "buffer_overflow": "u2r",
    "loadmodule": "u2r",
    "rootkit": "u2r",
    "perl": "u2r",
    "sqlattack": "u2r",
    "xterm": "u2r",
    "ps": "u2r",
}


def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load NSL-KDD file and apply proper column names.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    print(f"[INFO] Loading dataset: {file_path}")
    print("[INFO] Using safe settings: encoding='latin1', engine='python', on_bad_lines='skip'")

    try:
        # NSL-KDD is comma-separated text. We use safer read settings to handle:
        # - encoding issues
        # - malformed rows
        # - tokenizing errors
        # NSL-KDD can be comma-separated or tab-separated depending on the mirror.
        # Use a regex separator so both formats are parsed correctly.
        data = pd.read_csv(
            file_path,
            header=None,
            names=NSL_KDD_COLUMNS,
            sep=r"[\t,]+",
            encoding="latin1",
            engine="python",
            on_bad_lines="skip",
            skipinitialspace=True,
        )

        if data.empty:
            raise pd.errors.EmptyDataError(f"No valid rows found in file: {file_path}")

        print(f"[INFO] Loaded {len(data)} valid rows and {len(data.columns)} columns from {file_path.name}")
        return data

    except UnicodeDecodeError as decode_error:
        print(f"[ERROR] Encoding issue while reading {file_path.name}: {decode_error}")
        raise ValueError(
            f"Could not decode {file_path.name}. Try re-downloading the dataset file."
        ) from decode_error
    except pd.errors.ParserError as parser_error:
        print(f"[ERROR] Parser/tokenizing issue in {file_path.name}: {parser_error}")
        raise ValueError(
            f"Parsing failed for {file_path.name}. File may be heavily corrupted."
        ) from parser_error
    except pd.errors.EmptyDataError as empty_error:
        print(f"[ERROR] Empty or unreadable data in {file_path.name}: {empty_error}")
        raise
    except Exception as error:
        print(f"[ERROR] Unexpected loading error for {file_path.name}: {error}")
        raise


def map_attack_labels_to_categories(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw attack labels to broad categories (normal, dos, probe, r2l, u2r).
    """
    print("[INFO] Mapping raw attack labels to attack categories...")
    data = data.copy()
    data["attack_category"] = data["label"].map(ATTACK_CATEGORY_MAP).fillna("unknown")
    return data


def remove_unnecessary_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that are usually not needed for model training.
    """
    print("[INFO] Removing unnecessary columns (if present)...")
    data = data.copy()

    # difficulty_level is often metadata and not used as a feature.
    if "difficulty_level" in data.columns:
        data.drop(columns=["difficulty_level"], inplace=True)
        print("[INFO] Dropped column: difficulty_level")

    return data


def encode_categorical_columns(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    categorical_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Label-encode categorical feature columns using shared fit across train+test values.
    """
    print("[INFO] Encoding categorical columns using LabelEncoder...")
    train_data = train_data.copy()
    test_data = test_data.copy()
    encoders: Dict[str, LabelEncoder] = {}

    for col in categorical_columns:
        encoder = LabelEncoder()

        # Fit on combined values so test-only categories do not fail.
        combined_values = pd.concat(
            [train_data[col].astype(str), test_data[col].astype(str)], axis=0
        )
        encoder.fit(combined_values)

        train_data[col] = encoder.transform(train_data[col].astype(str))
        test_data[col] = encoder.transform(test_data[col].astype(str))
        encoders[col] = encoder
        print(f"[INFO] Encoded column: {col}")

    return train_data, test_data, encoders


def scale_numerical_columns(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    numerical_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Standardize numerical columns using StandardScaler (fit on train, transform both).
    """
    print("[INFO] Normalizing numerical columns using StandardScaler...")
    train_data = train_data.copy()
    test_data = test_data.copy()

    scaler = StandardScaler()
    train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])
    test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])

    print(f"[INFO] Scaled {len(numerical_columns)} numerical columns.")
    return train_data, test_data, scaler


def split_features_and_target(
    data: pd.DataFrame, target_column: str = "attack_category"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features X and target y.
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")

    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def save_processed_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_dir: Path) -> None:
    """
    Save processed train and test datasets as CSV files.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    processed_train_path = data_dir / "processed_train.csv"
    processed_test_path = data_dir / "processed_test.csv"

    train_data.to_csv(processed_train_path, index=False)
    test_data.to_csv(processed_test_path, index=False)

    print(f"[INFO] Saved processed train data to: {processed_train_path}")
    print(f"[INFO] Saved processed test data to: {processed_test_path}")


def save_preprocessing_artifacts(
    encoders: Dict[str, LabelEncoder],
    scaler: StandardScaler,
    feature_columns: List[str],
    categorical_columns: List[str],
    numerical_columns: List[str],
    models_dir: Path = Path("models"),
) -> None:
    """
    Save preprocessing objects so prediction uses the same transformations as training.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "categorical_encoders": encoders,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "target_column": "attack_category",
        "label_column": "label",
    }
    artifact_path = models_dir / "preprocessing_artifacts.pkl"
    joblib.dump(artifact, artifact_path)
    print(f"[INFO] Preprocessing artifacts saved to: {artifact_path}")


def resolve_data_dir(preferred_dir: Path = Path("data")) -> Path:
    """
    Prefer data/, but allow dataset/ for checklist-friendly project layouts.
    """
    train_name = "KDDTrain+.txt"
    test_name = "KDDTest+.txt"

    if (preferred_dir / train_name).exists() and (preferred_dir / test_name).exists():
        return preferred_dir

    alternate_dir = Path("dataset")
    if (alternate_dir / train_name).exists() and (alternate_dir / test_name).exists():
        print("[INFO] Using dataset/ because raw NSL-KDD files were found there.")
        return alternate_dir

    return preferred_dir


def preprocess_nsl_kdd(data_dir: Path = Path("data")) -> None:
    """
    Main preprocessing workflow for NSL-KDD.
    """
    try:
        print("[INFO] Starting NSL-KDD preprocessing...")
        data_dir = resolve_data_dir(data_dir)

        train_path = data_dir / "KDDTrain+.txt"
        test_path = data_dir / "KDDTest+.txt"

        train_df = load_dataset(train_path)
        test_df = load_dataset(test_path)

        train_df = map_attack_labels_to_categories(train_df)
        test_df = map_attack_labels_to_categories(test_df)

        train_df = remove_unnecessary_columns(train_df)
        test_df = remove_unnecessary_columns(test_df)

        # Keep raw label as useful reference, but target for ML will be attack_category.
        # Categorical features (not targets).
        categorical_columns = ["protocol_type", "service", "flag"]

        train_df, test_df, encoders = encode_categorical_columns(
            train_df, test_df, categorical_columns
        )

        # Numerical columns are all columns except non-feature targets.
        excluded_columns = {"label", "attack_category"}
        numerical_columns = [
            col for col in train_df.columns if col not in excluded_columns and col not in categorical_columns
        ]

        train_df, test_df, scaler = scale_numerical_columns(
            train_df, test_df, numerical_columns
        )

        # Optional split preview for users (for understanding pipeline).
        X_train, y_train = split_features_and_target(train_df, target_column="attack_category")
        X_test, y_test = split_features_and_target(test_df, target_column="attack_category")

        print(f"[INFO] Train features shape: {X_train.shape}, Train labels shape: {y_train.shape}")
        print(f"[INFO] Test features shape: {X_test.shape}, Test labels shape: {y_test.shape}")

        save_processed_data(train_df, test_df, data_dir)
        save_preprocessing_artifacts(
            encoders=encoders,
            scaler=scaler,
            feature_columns=list(X_train.drop(columns=["label"], errors="ignore").columns),
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
        )
        print("[INFO] Preprocessing completed successfully.")

    except FileNotFoundError as file_error:
        print(f"[ERROR] {file_error}")
        print("[HINT] Make sure KDDTrain+.txt and KDDTest+.txt are inside the data/ folder.")
    except pd.errors.EmptyDataError:
        print("[ERROR] One of the dataset files is empty.")
    except Exception as error:
        print(f"[ERROR] Unexpected error while preprocessing: {error}")


if __name__ == "__main__":
    preprocess_nsl_kdd()
