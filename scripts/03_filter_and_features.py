import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "aligned_sensor_data.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "windowed_dataset.npz"
DEFAULT_META = ROOT / "data" / "processed" / "windowed_dataset_meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter sensor streams and build fixed-size ML windows.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to aligned sensor CSV file.",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to save the generated train/val/test arrays.",
    )
    parser.add_argument(
        "--meta-json",
        type=Path,
        default=DEFAULT_META,
        help="Where to save metadata for feature and label names.",
    )
    parser.add_argument("--window-size", type=int, default=64, help="Number of samples per window.")
    parser.add_argument("--step-size", type=int, default=32, help="Stride between windows.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--val-size", type=float, default=0.25, help="Validation split fraction for the remaining train set.")
    return parser.parse_args()


def low_pass_filter(signal: np.ndarray, fs: float, cutoff_hz: float = 5.0, order: int = 4) -> np.ndarray:
    if len(signal) < 3:
        return signal.copy()
    nyquist = 0.5 * fs
    if cutoff_hz >= nyquist:
        cutoff_hz = max(0.1, nyquist * 0.8)
    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    return filtfilt(b, a, signal)


def prepare_sensor_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df = df.sort_values("t").reset_index(drop=True)

    numeric_cols = [
        "a_x", "a_y", "a_z", "gs_x", "gs_y", "gs_z",
        "m_x", "m_y", "m_z", "baro_hpa",
        "lat", "lon", "height_m", "velocity_mps", "direction_deg",
        "h_acc_m", "v_acc_deg"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].interpolate(method="linear", limit_direction="both")

    df["acc_mag"] = np.sqrt(df["a_x"] ** 2 + df["a_y"] ** 2 + df["a_z"] ** 2)
    df["gyro_mag"] = np.sqrt(df["gs_x"] ** 2 + df["gs_y"] ** 2 + df["gs_z"] ** 2)
    df["mag_mag"] = np.sqrt(df["m_x"] ** 2 + df["m_y"] ** 2 + df["m_z"] ** 2)
    return df


def build_feature_columns() -> list[str]:
    return [
        "a_x", "a_y", "a_z", "acc_mag",
        "gs_x", "gs_y", "gs_z", "gyro_mag",
        "m_x", "m_y", "m_z", "mag_mag",
        "baro_hpa",
        "lat", "lon", "height_m",
        "velocity_mps", "direction_deg",
        "h_acc_m", "v_acc_deg"
    ]


def compute_filtered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = prepare_sensor_dataframe(df)
    fs = 1.0 / np.median(np.diff(df["t"].dropna().to_numpy())) if len(df) > 1 else 50.0
    fs = max(fs, 1.0)

    feature_cols = build_feature_columns()
    filtered = df[feature_cols].copy()

    for col in feature_cols:
        if col in filtered.columns:
            filtered[col] = low_pass_filter(filtered[col].to_numpy(), fs=fs, cutoff_hz=5.0)

    return filtered


def create_windows(features_df: pd.DataFrame, window_size: int, step_size: int, feature_names: list[str]):
    if len(features_df) < window_size:
        raise ValueError(f"Dataset too small for window_size={window_size}. Size={len(features_df)}")

    feature_array = features_df[feature_names].to_numpy(dtype=np.float64)
    if np.isnan(feature_array).any():
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)

    X_windows = []
    y_windows = []

    vel_ix = feature_names.index("velocity_mps")
    dir_ix = feature_names.index("direction_deg")

    for start in range(0, len(feature_array) - window_size + 1, step_size):
        window = feature_array[start:start + window_size]
        if np.isnan(window).any():
            continue

        X_windows.append(window)
        velocity = float(np.mean(window[:, vel_ix]))
        heading = float(np.mean(window[:, dir_ix]))
        y_windows.append([velocity, heading])

    if len(X_windows) == 0:
        raise ValueError("No valid windows were created from the dataset.")

    return np.asarray(X_windows, dtype=np.float64), np.asarray(y_windows, dtype=np.float64)


def split_windows(X: np.ndarray, y: np.ndarray, test_size: float, val_size: float):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42, shuffle=False
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_arrays(output_npz: Path, X_train, X_val, X_test, y_train, y_val, y_test, feature_names, target_names):
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    metadata = {
        "feature_names": feature_names,
        "target_names": target_names,
        "window_shape": list(X_train.shape[1:]),
        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
    }
    return metadata


if __name__ == "__main__":
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    feature_df = compute_filtered_features(df)
    feature_columns = list(feature_df.columns)

    X, y = create_windows(feature_df, args.window_size, args.step_size, feature_columns)
    X_train, X_val, X_test, y_train, y_val, y_test = split_windows(X, y, args.test_size, args.val_size)

    metadata = save_arrays(
        args.output_npz,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        feature_columns,
        ["avg_velocity_mps", "avg_direction_deg"],
    )

    with open(args.meta_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Input dataset: {args.input_csv}")
    print(f"Saved arrays to: {args.output_npz}")
    print(f"Saved metadata to: {args.meta_json}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Window shape: {X_train.shape}")
    print(f"Train/X: {X_train.shape}, Train/y: {y_train.shape}")
    print(f"Val/X: {X_val.shape}, Val/y: {y_val.shape}")
    print(f"Test/X: {X_test.shape}, Test/y: {y_test.shape}")
