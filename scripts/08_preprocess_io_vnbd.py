"""
IO-VNBD Dataset Preprocessing Pipeline
=======================================
Preprocesses Inertial and Odometry Benchmark Dataset for ground vehicle positioning.

The IO-VNBD dataset contains vehicle telemetry (OBD + GPS):
- Vehicle (V) data: 40 hours, 1,300 km
- Smartphone (S) data: 58 hours, 4,400 km
- Multiple drivers, locations, road types

Expected file structure (after Git LFS download):
    IO-VNBD-master/
    └── Synchronised V abd S datasets/
        └── Categorised IOVNB Dataset/
            └── [Route Name] (Driver)/
                ├── V-[Route].csv    (vehicle telemetry)
                └── S-[Route].csv    (smartphone IMU + GPS)

Actual columns in V-*.csv (Vehicle telemetry):
- Time Since Start of Day (seconds)
- Latitude (degrees), Longitude (degrees)
- Velocity (km/hr), Heading (degrees)  ← GROUND TRUTH
- Height (km)
- Wheel Speed Front/Rear Left/Right (rad/sec)
- Yaw Rate (deg/sec)
- Indicated Longitudinal/Lateral Acceleration (g)
- Engine, Steering, Brake data
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "raw" / "IO-VNBD"
PROCESSED_DIR = ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess IO-VNBD dataset for ML training.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to IO-VNBD dataset root (contains V-*.csv and S-*.csv files).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Where to save processed data.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=64,
        help="Number of samples per window.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=32,
        help="Stride between windows (overlap = window_size - step_size).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.25,
        help="Validation split fraction from training set.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit total number of windows (for storage constraints). Default: use all.",
    )
    parser.add_argument(
        "--use-vehicle",
        action="store_true",
        default=True,
        help="Use vehicle (V) data.",
    )
    parser.add_argument(
        "--use-smartphone",
        action="store_true",
        help="Also use smartphone (S) data (merged with vehicle).",
    )
    return parser.parse_args()


def find_data_files(data_dir: Path, use_vehicle: bool = True, use_smartphone: bool = False) -> list:
    """Find all V-*.csv and S-*.csv files in the dataset directory."""
    files = []
    
    if use_vehicle:
        v_files = list(data_dir.rglob("V-*.csv"))
        files.extend([(f, "vehicle") for f in v_files])
        print(f"Found {len(v_files)} vehicle files (V-*.csv)")
    
    if use_smartphone:
        s_files = list(data_dir.rglob("S-*.csv"))
        files.extend([(f, "smartphone") for f in s_files])
        print(f"Found {len(s_files)} smartphone files (S-*.csv)")
    
    return files


def load_and_align_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """Load IO-VNBD CSV and extract telemetry features."""
    try:
        df = pd.read_csv(csv_path)
        
        # Strip leading/trailing whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Check for minimum required columns (Velocity and Heading at minimum)
        if "Velocity (km/hr)" not in df.columns or "Heading (degrees)" not in df.columns:
            print(f"Warning: {csv_path.name} missing Velocity or Heading. Skipping.")
            return None
        
        # Rename columns to match our pipeline format
        rename_map = {
            "Time Since Start of Day (seconds)": "t",
            "Latitude (degrees)": "lat",
            "Longitude (degrees)": "lon",
            "Height (km)": "height_km",
            "Velocity (km/hr)": "velocity_kmh",
            "Heading (degrees)": "heading_deg",
            "Wheel Speed Front Left (rad/sec)": "ws_fl",
            "Wheel Speed Front Right (rad/sec)": "ws_fr",
            "Wheel Speed Rear Left (rad/sec)": "ws_rl",
            "Wheel Speed Rear Right (rad/sec)": "ws_rr",
            "Yaw Rate (deg/sec)": "yaw_rate",
            "Indicated Longitudinal Acceleration (g)": "accel_long",
            "Indicated Lateral Acceleration (g)": "accel_lat",
            "Steering Angle (degrees)": "steering_angle",
            "Engine Speed (rev/min)": "engine_rpm",
            "Indicated Vehicle Speed (km/hr)": "indicated_speed_kmh",
        }
        
        # Rename available columns
        available_renames = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=available_renames)
        
        # Sort by time if available
        if "t" in df.columns:
            df = df.sort_values("t").reset_index(drop=True)
        
        return df
    
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None


def low_pass_filter(signal: np.ndarray, fs: float, cutoff_hz: float = 5.0, order: int = 4) -> np.ndarray:
    """Apply Butterworth low-pass filter to smooth sensor noise."""
    if len(signal) < 3:
        return signal.copy()
    
    nyquist = 0.5 * fs
    if cutoff_hz >= nyquist:
        cutoff_hz = max(0.1, nyquist * 0.8)
    
    try:
        b, a = butter(order, cutoff_hz / nyquist, btype="low")
        return filtfilt(b, a, signal)
    except:
        return signal.copy()


def prepare_sensor_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess telemetry data: convert types, interpolate, compute features."""
    df = df.copy()
    
    # Ensure time column is numeric
    if "t" in df.columns:
        df["t"] = pd.to_numeric(df["t"], errors="coerce")
        df = df.sort_values("t").reset_index(drop=True)
    
    # Numeric columns to interpolate (telemetry format)
    numeric_cols = [
        "lat", "lon", "height_km", "velocity_kmh", "heading_deg",
        "ws_fl", "ws_fr", "ws_rl", "ws_rr",
        "yaw_rate", "accel_long", "accel_lat", "steering_angle", "engine_rpm"
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
            df[col] = df[col].fillna(0.0)
    
    # Compute derived features
    if all(c in df.columns for c in ["ws_fl", "ws_fr", "ws_rl", "ws_rr"]):
        # Average wheel speed (rad/sec) → convert to velocity
        df["wheel_speed_avg"] = (df["ws_fl"] + df["ws_fr"] + df["ws_rl"] + df["ws_rr"]) / 4.0
        # Tire radius ~0.33m → velocity in m/s
        df["velocity_from_wheels_mps"] = df["wheel_speed_avg"] * 0.33
    
    # Convert velocity from km/h to m/s
    if "velocity_kmh" in df.columns:
        df["velocity_mps"] = df["velocity_kmh"] / 3.6
    
    return df


def build_feature_columns() -> list:
    """Define features for ML input — NO leakage of labels (heading/velocity/position)."""
    return [
        "ws_fl", "ws_fr", "ws_rl", "ws_rr",          # Wheel speeds (4) — IMU-available
        "wheel_speed_avg", "velocity_from_wheels_mps", # Derived wheel velocity (2)
        "yaw_rate",                                    # Yaw rotation rate
        "accel_long",                                  # Longitudinal acceleration
        "accel_lat",                                   # Lateral acceleration
        "steering_angle",                              # Steering input
        "engine_rpm",                                  # Engine speed
        "height_km",                                   # Altitude (barometer proxy)
    ]


def compute_filtered_features(df: pd.DataFrame):
    """Apply low-pass filtering and derive features. Returns (feature_df, labels_df)."""
    df = prepare_sensor_dataframe(df)

    if len(df) > 1 and "t" in df.columns:
        time_diff = np.diff(df["t"].dropna().to_numpy())
        fs = 1.0 / np.median(time_diff) if len(time_diff) > 0 else 50.0
    else:
        fs = 50.0
    fs = max(fs, 1.0)

    feature_cols = [c for c in build_feature_columns() if c in df.columns]
    filtered = df[feature_cols].copy()
    for col in feature_cols:
        try:
            filtered[col] = low_pass_filter(filtered[col].to_numpy(), fs=fs, cutoff_hz=5.0)
        except:
            pass

    # Labels: GPS velocity and heading — kept separate, never in X
    labels = df[["velocity_mps", "heading_deg"]].copy()
    return filtered, labels


def create_windows(features_df: pd.DataFrame, labels_df: pd.DataFrame, window_size: int, step_size: int, feature_names: list):
    """Create fixed-size windows for LSTM input. Labels come from separate GPS columns."""
    if len(features_df) < window_size:
        raise ValueError(f"Dataset too small for window_size={window_size}. Got {len(features_df)} samples.")

    feature_array = features_df[feature_names].to_numpy(dtype=np.float64)
    if np.isnan(feature_array).any():
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)

    vel_array = labels_df["velocity_mps"].to_numpy(dtype=np.float64)
    hdg_array = labels_df["heading_deg"].to_numpy(dtype=np.float64)

    X_windows, y_windows = [], []
    for start in range(0, len(feature_array) - window_size + 1, step_size):
        window = feature_array[start:start + window_size]
        if np.isnan(window).any():
            continue
        X_windows.append(window)
        y_windows.append([float(np.mean(vel_array[start:start + window_size])),
                          float(np.mean(hdg_array[start:start + window_size]))])

    if len(X_windows) == 0:
        raise ValueError("No valid windows created from dataset.")
    return np.asarray(X_windows, dtype=np.float64), np.asarray(y_windows, dtype=np.float64)


def split_windows(X: np.ndarray, y: np.ndarray, test_size: float, val_size: float, max_samples: Optional[int] = None):
    """Split into train/val/test sets with optional downsampling."""
    # Downsample if needed
    if max_samples is not None and len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        indices.sort()  # Keep temporal order
        X = X[indices]
        y = y[indices]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42, shuffle=False
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_arrays(output_dir: Path, X_train, X_val, X_test, y_train, y_val, y_test, feature_names, target_names, dataset_source: str = "IO-VNBD"):
    """Save windowed data as NPZ and metadata as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_npz = output_dir / f"windowed_dataset_{dataset_source}.npz"
    output_meta = output_dir / f"windowed_dataset_{dataset_source}_meta.json"
    
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
        "dataset": dataset_source,
        "feature_names": feature_names,
        "target_names": target_names,
        "window_shape": list(X_train.shape[1:]),
        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
        "splits": {
            "train_size": int(X_train.shape[0]),
            "val_size": int(X_val.shape[0]),
            "test_size": int(X_test.shape[0]),
        }
    }
    
    with open(output_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    return output_npz, output_meta


def main():
    args = parse_args()
    
    print("="*60)
    print("IO-VNBD Dataset Preprocessing")
    print("="*60)
    
    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")
    
    # Find all CSV files
    data_files = find_data_files(args.dataset_dir, args.use_vehicle, args.use_smartphone)
    
    if not data_files:
        raise FileNotFoundError(
            f"No CSV files found in {args.dataset_dir}.\n"
            "Expected: V-*.csv (vehicle) and/or S-*.csv (smartphone) files.\n"
            "Note: If using Git LFS, ensure data is downloaded with 'git lfs pull'."
        )
    
    # Load and merge all files
    print(f"\nLoading {len(data_files)} file(s)...")
    all_dfs = []
    
    for csv_path, data_type in data_files:
        df = load_and_align_csv(csv_path)
        if df is not None:
            print(f"  OK {csv_path.name} ({len(df)} rows)")
            all_dfs.append(df)
    
    if not all_dfs:
        raise ValueError("No valid CSV data could be loaded.")
    
    # Merge all dataframes
    if len(all_dfs) == 1:
        merged = all_dfs[0]
    else:
        # Concatenate and sort by time
        merged = pd.concat(all_dfs, ignore_index=True)
        if "t" in merged.columns:
            merged = merged.sort_values("t").reset_index(drop=True)
    
    print(f"\nMerged dataset: {merged.shape[0]} rows × {merged.shape[1]} columns")
    
    # Process features
    print("\nProcessing features (filtering, interpolation, magnitudes)...")
    feature_df, labels_df = compute_filtered_features(merged)
    feature_columns = list(feature_df.columns)

    # Create windows
    print(f"\nCreating windows (size={args.window_size}, step={args.step_size})...")
    X, y = create_windows(feature_df, labels_df, args.window_size, args.step_size, feature_columns)
    print(f"  Created {len(X)} windows")
    
    # Split
    print(f"\nSplitting data (test={args.test_size*100:.0f}%, val={args.val_size*100:.0f}%)...")
    if args.max_samples:
        print(f"  Downsampling to {args.max_samples} windows...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_windows(
        X, y, args.test_size, args.val_size, args.max_samples
    )
    
    # Save
    print("\nSaving processed data...")
    output_npz, output_meta = save_arrays(
        args.output_dir,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        feature_columns,
        ["avg_velocity_mps", "avg_direction_deg"],
        dataset_source="IO-VNBD"
    )
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Dataset: IO-VNBD")
    print(f"Input files: {len(data_files)}")
    print(f"Total samples: {len(X)}")
    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Window shape: {X_train.shape[1:]}")
    print(f"\nSaved NPZ: {output_npz}")
    print(f"Saved metadata: {output_meta}")
    print(f"\nReady for ML training (Person 2)!")


if __name__ == "__main__":
    main()
