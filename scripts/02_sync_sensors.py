import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "raw" / "test_case0"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align IMU and GPS sensor data for PDR preprocessing.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the raw CSV sensor files.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Time (s)" not in df.columns:
        raise ValueError(f"Missing 'Time (s)' in {path.name}: {list(df.columns)}")
    df = df.sort_values("Time (s)").reset_index(drop=True)
    return df


def build_merged_dataset(data_dir: Path) -> pd.DataFrame:
    accel = load_csv(data_dir / "Accelerometer.csv")
    gyro = load_csv(data_dir / "Gyroscope.csv")
    mag = load_csv(data_dir / "Magnetometer.csv")
    linear = load_csv(data_dir / "Linear Accelerometer.csv")
    baro = load_csv(data_dir / "Barometer.csv")

    gps_path = data_dir / "Location_input.csv"
    if not gps_path.exists():
        gps_path = data_dir / "Location.csv"
    if not gps_path.exists():
        raise FileNotFoundError("No GPS file found. Expected Location_input.csv or Location.csv")
    gps = load_csv(gps_path)

    # Keep the best available output columns for ground truth.
    gps = gps[[
        "Time (s)",
        "Latitude (°)",
        "Longitude (°)",
        "Height (m)",
        "Velocity (m/s)",
        "Direction (°)",
        "Horizontal Accuracy (m)",
        "Vertical Accuracy (°)",
    ]].copy()

    sensor_frames = [accel, gyro, mag, linear, baro]
    merged = sensor_frames[0]
    for frame in sensor_frames[1:]:
        merged = pd.merge_asof(
            merged.sort_values("Time (s)"),
            frame.sort_values("Time (s)"),
            on="Time (s)",
            direction="nearest",
            tolerance=0.1,
            suffixes=("", "_y"),
        )
        # Drop duplicate columns created by matching on identical names if any.
        drop_cols = [c for c in merged.columns if c.endswith("_y")]
        merged = merged.drop(columns=drop_cols)

    merged = pd.merge_asof(
        merged.sort_values("Time (s)"),
        gps.sort_values("Time (s)"),
        on="Time (s)",
        direction="nearest",
        tolerance=1.0,
        suffixes=("", "_gps"),
    )

    # Keep the IMU columns visible and clean naming for later ML feature engineering.
    merged = merged.rename(columns={
        "Time (s)": "t",
        "X (m/s^2)": "a_x",
        "Y (m/s^2)": "a_y",
        "Z (m/s^2)": "a_z",
        "X (rad/s)": "gs_x",
        "Y (rad/s)": "gs_y",
        "Z (rad/s)": "gs_z",
        "X (µT)": "m_x",
        "Y (µT)": "m_y",
        "Z (µT)": "m_z",
        "X (hPa)": "baro_hpa",
        "Latitude (°)": "lat",
        "Longitude (°)": "lon",
        "Height (m)": "height_m",
        "Velocity (m/s)": "velocity_mps",
        "Direction (°)": "direction_deg",
        "Horizontal Accuracy (m)": "h_acc_m",
        "Vertical Accuracy (°)": "v_acc_deg",
    })

    # Keep the linear acceleration columns in a readable form.
    for suffix, new_name in {
        "X (m/s^2)_x": "la_x",
        "Y (m/s^2)_x": "la_y",
        "Z (m/s^2)_x": "la_z",
    }.items():
        if suffix in merged.columns:
            merged = merged.rename(columns={suffix: new_name})

    # If the merge created duplicate linear-accel columns, drop the redundant copy.
    dup_cols = [c for c in merged.columns if c.startswith("X (m/s^2)") and c != "X (m/s^2)"]
    for c in dup_cols:
        merged = merged.drop(columns=[c])
    dup_cols = [c for c in merged.columns if c.startswith("Y (m/s^2)") and c != "Y (m/s^2)"]
    for c in dup_cols:
        merged = merged.drop(columns=[c])
    dup_cols = [c for c in merged.columns if c.startswith("Z (m/s^2)") and c != "Z (m/s^2)"]
    for c in dup_cols:
        merged = merged.drop(columns=[c])

    # Normalize names for the linear accel columns if they still exist with suffixes.
    for old, new in [("X (m/s^2)", "la_x"), ("Y (m/s^2)", "la_y"), ("Z (m/s^2)", "la_z")]:
        if old in merged.columns:
            merged = merged.rename(columns={old: new})

    return merged


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.dataset_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    merged = build_merged_dataset(data_dir)
    output_path = PROCESSED_DIR / "aligned_sensor_data.csv"
    merged.to_csv(output_path, index=False)
    print(f"Dataset directory: {data_dir}")
    print(f"Saved aligned data to: {output_path}")
    print(merged.head(5).to_string(index=False))
    print("\nColumns:", list(merged.columns))
    print("Shape:", merged.shape)
