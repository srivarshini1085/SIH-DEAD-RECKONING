"""
In-Vehicle Alignment & Calibration Engine
==========================================
Corrects for phone mounting orientation so accelerometer "forward" matches
the vehicle's forward direction.

Two modes:
  1. FULL mode  -- reads aligned_sensor_data.csv (output of 02_sync_sensors.py)
                   with raw phone IMU columns: a_x, a_y, a_z, gs_x, gs_y, gs_z
  2. IO-VNBD mode -- no raw phone IMU available; the IO-VNBD dataset uses
                     vehicle odometry (wheel speeds, yaw_rate, accel_long/lat).
                     Reconstructs a synthetic aligned CSV from the .npz so
                     downstream scripts have a consistent file, and reports
                     identity calibration (no phone tilt to correct).

Usage:
    py scripts/10_calibration.py
    py scripts/10_calibration.py --input path/to/aligned_sensor_data.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT  = ROOT / "data" / "processed" / "aligned_sensor_data.csv"
CALIBRATED_OUT = ROOT / "data" / "processed" / "calibrated_sensor_data.csv"
NPZ_PATH       = ROOT / "data" / "processed" / "windowed_dataset_IO-VNBD.npz"
META_PATH      = ROOT / "data" / "processed" / "windowed_dataset_IO-VNBD_meta.json"

STATIONARY_VAR_THRESHOLD  = 0.05
STATIONARY_WINDOW_SAMPLES = 50
MIN_STATIONARY_SAMPLES    = 30
MOTION_SPEED_THRESHOLD    = 1.5
MOTION_HEADING_STD_MAX    = 10.0
MOTION_WINDOW_SAMPLES     = 100
GRAVITY_MPS2              = 9.81


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phone orientation calibration for vehicle dead reckoning.")
    p.add_argument("--input",  type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=CALIBRATED_OUT)
    return p.parse_args()


# -- Rotation helpers ----------------------------------------------------------

def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if abs(c + 1.0) < 1e-6:
        perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
        v = np.cross(a, perp)
        v /= np.linalg.norm(v)
        return 2 * np.outer(v, v) - np.eye(3)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2 + 1e-12))


def rotation_matrix_yaw(yaw_rad: float) -> np.ndarray:
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def matrix_to_euler_deg(R: np.ndarray):
    pitch = np.degrees(np.arcsin(-R[2, 0]))
    roll  = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw   = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw


# -- IO-VNBD fallback: reconstruct synthetic CSV from .npz --------------------

def reconstruct_csv_from_npz(output_path: Path) -> pd.DataFrame:
    """
    IO-VNBD has no raw phone IMU. Reconstruct a flat CSV from the training
    windows so downstream scripts have a consistent file to read.
    Calibration is identity -- vehicle-mounted sensors are already in
    vehicle frame.
    """
    import json
    d = np.load(NPZ_PATH)
    X_train = d["X_train"]   # (N, 64, 12)
    y_train = d["y_train"]   # (N, 2)  [velocity, heading]

    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    feat_names = meta["feature_names"]   # 12 names

    # Take the first sample of each window (stride already applied in npz)
    rows = []
    for i in range(len(X_train)):
        row = {name: float(X_train[i, 0, j]) for j, name in enumerate(feat_names)}
        row["velocity_mps"]  = float(y_train[i, 0])
        row["direction_deg"] = float(y_train[i, 1])
        row["t"] = float(i) * 1.28
        rows.append(row)

    df = pd.DataFrame(rows)

    # Add synthetic IMU columns so 03_filter_and_features.py can find them
    df["a_x"]  = df["accel_long"] if "accel_long" in df.columns else 0.0
    df["a_y"]  = df["accel_lat"]  if "accel_lat"  in df.columns else 0.0
    df["a_z"]  = -GRAVITY_MPS2
    df["gs_x"] = 0.0
    df["gs_y"] = 0.0
    df["gs_z"] = df["yaw_rate"] if "yaw_rate" in df.columns else 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


# -- Full calibration (phone IMU CSV available) --------------------------------

def find_stationary_mask(df: pd.DataFrame) -> np.ndarray:
    acc_mag = np.sqrt(df["a_x"]**2 + df["a_y"]**2 + df["a_z"]**2).to_numpy()
    rolling_var = (pd.Series(acc_mag)
                   .rolling(STATIONARY_WINDOW_SAMPLES, center=True)
                   .var().fillna(999).to_numpy())
    return rolling_var < STATIONARY_VAR_THRESHOLD


def estimate_gravity_rotation(df: pd.DataFrame, stationary_mask: np.ndarray):
    idle_df = df[stationary_mask]
    if len(idle_df) < MIN_STATIONARY_SAMPLES:
        print("  WARNING: Too few stationary samples -- using identity for pitch/roll.")
        return np.eye(3), 0.0, 0.0, len(idle_df)
    g_measured = np.array([idle_df["a_x"].mean(), idle_df["a_y"].mean(), idle_df["a_z"].mean()])
    g_true = np.array([0.0, 0.0, -GRAVITY_MPS2])
    R_grav = rotation_matrix_from_vectors(g_measured, g_true)
    roll, pitch, _ = matrix_to_euler_deg(R_grav)
    return R_grav, roll, pitch, len(idle_df)


def estimate_yaw_rotation(df: pd.DataFrame, R_grav: np.ndarray):
    if "velocity_mps" not in df.columns or "direction_deg" not in df.columns:
        print("  WARNING: GPS velocity/heading missing -- skipping yaw calibration.")
        return np.eye(3), 0.0, 0

    speed   = df["velocity_mps"].fillna(0).to_numpy()
    heading = df["direction_deg"].fillna(np.nan).to_numpy()
    n = len(df)
    best_start = None
    for i in range(0, n - MOTION_WINDOW_SAMPLES, MOTION_WINDOW_SAMPLES // 2):
        seg_speed = speed[i: i + MOTION_WINDOW_SAMPLES]
        seg_hdg   = heading[i: i + MOTION_WINDOW_SAMPLES]
        if (np.nanmean(seg_speed) > MOTION_SPEED_THRESHOLD and
                np.nanstd(seg_hdg) < MOTION_HEADING_STD_MAX and
                not np.isnan(seg_hdg).any()):
            best_start = i
            break

    if best_start is None:
        print("  WARNING: No clear forward-motion window -- skipping yaw calibration.")
        return np.eye(3), 0.0, 0

    seg = df.iloc[best_start: best_start + MOTION_WINDOW_SAMPLES]
    gps_heading_deg = float(seg["direction_deg"].mean())
    acc = np.column_stack([seg["a_x"].to_numpy(), seg["a_y"].to_numpy(), seg["a_z"].to_numpy()])
    acc_corrected = (R_grav @ acc.T).T
    fwd_x = float(np.mean(acc_corrected[:, 0]))
    fwd_y = float(np.mean(acc_corrected[:, 1]))
    phone_heading_deg = float(np.degrees(np.arctan2(fwd_y, fwd_x)) % 360)
    yaw_offset_deg = (gps_heading_deg - phone_heading_deg + 180) % 360 - 180
    return rotation_matrix_yaw(np.radians(yaw_offset_deg)), yaw_offset_deg, MOTION_WINDOW_SAMPLES


def apply_rotation(df: pd.DataFrame, R: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    acc  = np.column_stack([df["a_x"].to_numpy(),  df["a_y"].to_numpy(),  df["a_z"].to_numpy()])
    gyro = np.column_stack([df["gs_x"].to_numpy(), df["gs_y"].to_numpy(), df["gs_z"].to_numpy()])
    acc_rot  = (R @ acc.T).T
    gyro_rot = (R @ gyro.T).T
    df["a_x"],  df["a_y"],  df["a_z"]  = acc_rot[:, 0],  acc_rot[:, 1],  acc_rot[:, 2]
    df["gs_x"], df["gs_y"], df["gs_z"] = gyro_rot[:, 0], gyro_rot[:, 1], gyro_rot[:, 2]
    return df


def calibrate_from_csv(input_path: Path, output_path: Path) -> pd.DataFrame:
    print("Loading: %s" % input_path)
    df = pd.read_csv(input_path)

    for col in ["a_x", "a_y", "a_z", "gs_x", "gs_y", "gs_z"]:
        if col not in df.columns:
            raise ValueError("Required column '%s' not found. Run 02_sync_sensors.py first." % col)
        df[col] = pd.to_numeric(df[col], errors="coerce").interpolate(limit_direction="both")

    print("\n-- Step 1: Detecting stationary periods --")
    mask = find_stationary_mask(df)
    print("  Stationary samples: %d / %d" % (mask.sum(), len(df)))

    print("\n-- Step 2: Estimating pitch/roll from gravity vector --")
    R_grav, roll_deg, pitch_deg, n_stat = estimate_gravity_rotation(df, mask)
    print("  Roll  correction: %+.2f deg" % roll_deg)
    print("  Pitch correction: %+.2f deg" % pitch_deg)

    print("\n-- Step 3: Estimating yaw from GPS heading --")
    R_yaw, yaw_deg, n_motion = estimate_yaw_rotation(df, R_grav)
    print("  Yaw correction: %+.2f deg" % yaw_deg)

    R_combined = R_yaw @ R_grav
    df_cal = apply_rotation(df, R_combined)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_cal.to_csv(output_path, index=False)

    print("\n[OK] Calibrated data saved -> %s" % output_path)
    print("\n" + "="*55)
    print("  CALIBRATION SUMMARY")
    print("="*55)
    print("  Roll  correction : %+.2f deg" % roll_deg)
    print("  Pitch correction : %+.2f deg" % pitch_deg)
    print("  Yaw   correction : %+.2f deg" % yaw_deg)
    print("  Stationary windows used : %d" % n_stat)
    print("  Motion windows used     : %d" % n_motion)
    print("  Output rows             : %d" % len(df_cal))
    print("="*55)
    return df_cal


# -- Main ----------------------------------------------------------------------

def main():
    args = parse_args()

    if args.input.exists():
        calibrate_from_csv(args.input, args.output)

    elif NPZ_PATH.exists():
        print("aligned_sensor_data.csv not found.")
        print("Detected IO-VNBD dataset at: %s" % NPZ_PATH)
        print("Running in IO-VNBD mode -- reconstructing synthetic CSV from .npz\n")
        print("NOTE: IO-VNBD uses vehicle-frame odometry sensors (wheel speeds,")
        print("      yaw_rate, accel_long/lat) -- already in vehicle frame.")
        print("      Calibration correction = identity (0 deg roll/pitch/yaw)\n")

        df = reconstruct_csv_from_npz(args.output)

        print("[OK] Synthetic calibrated CSV saved -> %s" % args.output)
        print("\n" + "="*55)
        print("  CALIBRATION SUMMARY  (IO-VNBD mode)")
        print("="*55)
        print("  Roll  correction : +0.00 deg (vehicle-frame sensors)")
        print("  Pitch correction : +0.00 deg (vehicle-frame sensors)")
        print("  Yaw   correction : +0.00 deg (vehicle-frame sensors)")
        print("  Windows reconstructed   : %d" % len(df))
        print("  Features                : %s" % [c for c in df.columns if c != "t"])
        print("="*55)

    else:
        raise FileNotFoundError(
            "Neither %s nor %s found.\n"
            "Run 02_sync_sensors.py (for phone IMU data) or ensure "
            "windowed_dataset_IO-VNBD.npz is in data/processed/." % (args.input, NPZ_PATH)
        )


if __name__ == "__main__":
    main()
