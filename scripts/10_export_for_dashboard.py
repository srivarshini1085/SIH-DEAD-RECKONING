"""
Export dashboard trajectories from the leakage-free 12-column dataset.

Reads feature names dynamically from windowed_dataset_IO-VNBD_meta.json
(lat/lon/GPS velocity/heading are no longer in X). Ground-truth GPS is
reconstructed by integrating the label velocity/heading. Naive dead
reckoning integrates raw IMU (accel + yaw). The AI path uses the trained
PyTorch LSTM (via EKF fusion when GPS is locked).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

DATA_PATH = ROOT / "data" / "processed" / "windowed_dataset_IO-VNBD.npz"
META_PATH = ROOT / "data" / "processed" / "windowed_dataset_IO-VNBD_meta.json"
MODEL_PATH = ROOT / "models" / "lstm_velocity_direction_io_vnbd.pt"
OUT_PATHS = [
    ROOT / "frontend" / "sample_data.csv",
    ROOT / "frontend_data" / "sample_data.csv",
]

# Fallback names if metadata is missing (post 11_fix_leakage.py order)
DEFAULT_FEATURE_NAMES = [
    "ws_fl", "ws_fr", "ws_rl", "ws_rr",
    "wheel_speed_avg", "velocity_from_wheels_mps",
    "yaw_rate", "accel_long", "accel_lat",
    "steering_angle", "engine_rpm", "height_km",
]

WINDOW_DT = 1.28  # 64 samples @ 50 Hz
EARTH_M_PER_DEG_LAT = 111320.0
SESSION_BREAK_HEADING_DEG = 120.0
DEFAULT_ORIGIN_LAT = 52.5184417671875  # prior dashboard origin (UK)
DEFAULT_ORIGIN_LON = -1.5083977265625
MAX_VEHICLE_SPEED_MPS = 50.0


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.minimum(1.0, a)))


def propagate(lat, lon, vel, heading_deg, dt):
    hdg = np.deg2rad(heading_deg)
    lat_rad = np.deg2rad(lat)
    dlat = (vel * np.cos(hdg) * dt) / EARTH_M_PER_DEG_LAT
    dlon = (vel * np.sin(hdg) * dt) / (EARTH_M_PER_DEG_LAT * max(np.cos(lat_rad), 1e-6))
    return lat + dlat, lon + dlon


def wrap_heading(deg):
    return float(np.mod(deg, 360.0))


def heading_delta_deg(a, b):
    return np.abs((b - a + 180.0) % 360.0 - 180.0)


def load_feature_names(n_features: int) -> list[str]:
    names = list(DEFAULT_FEATURE_NAMES)
    if META_PATH.exists():
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        meta_names = meta.get("feature_names") or []
        if meta_names:
            names = list(meta_names)
    if len(names) != n_features:
        if len(names) > n_features:
            names = names[:n_features]
        else:
            names = names + [f"feature_{i}" for i in range(len(names), n_features)]
    return names


def feature_index(names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(names)}


def col(idx: dict[str, int], *candidates: str) -> int | None:
    for name in candidates:
        if name in idx:
            return idx[name]
    return None


def load_full_sequence():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")
    d = np.load(DATA_PATH)
    X_full = np.concatenate([d["X_train"], d["X_val"], d["X_test"]], axis=0)
    y_full = np.concatenate([d["y_train"], d["y_val"], d["y_test"]], axis=0)
    names = load_feature_names(X_full.shape[-1])
    return X_full, y_full, names


def find_longest_continuous_segment(y_full, max_hdg_jump=SESSION_BREAK_HEADING_DEG):
    """Split on abrupt heading jumps (route concatenations). Lat/lon are not in X."""
    hdg = y_full[:, 1]
    if len(hdg) < 2:
        return 0, len(hdg)
    jumps = heading_delta_deg(hdg[:-1], hdg[1:])
    breaks = np.where(jumps > max_hdg_jump)[0]
    bounds = [0] + list(breaks + 1) + [len(hdg)]
    seg_lens = np.diff(bounds)
    best = int(np.argmax(seg_lens))
    return bounds[best], bounds[best + 1]


def reconstruct_gps_path(y_seg, init_lat, init_lon, dt=WINDOW_DT):
    """Integrate GPS-label velocity/heading from a map origin (labels, not X)."""
    n = y_seg.shape[0]
    lat, lon = np.zeros(n), np.zeros(n)
    cur_lat, cur_lon = init_lat, init_lon
    for i in range(n):
        vel = float(np.clip(y_seg[i, 0], 0.0, MAX_VEHICLE_SPEED_MPS))
        hdg = wrap_heading(y_seg[i, 1])
        lat[i], lon[i] = cur_lat, cur_lon
        cur_lat, cur_lon = propagate(cur_lat, cur_lon, vel, hdg, dt)
    return lat, lon


def naive_dead_reckoning_path(X_seg, idx, init_lat, init_lon, init_heading_deg, init_vel):
    """Double-integrate IMU: accel_long (g) → speed, yaw_rate (deg/s) → heading."""
    n = X_seg.shape[0]
    lat, lon = np.zeros(n), np.zeros(n)
    cur_lat, cur_lon = init_lat, init_lon
    heading = wrap_heading(init_heading_deg)
    vel = float(np.clip(init_vel, 0.0, MAX_VEHICLE_SPEED_MPS))

    i_yaw = col(idx, "yaw_rate")
    i_acc = col(idx, "accel_long")
    if i_yaw is None or i_acc is None:
        raise KeyError(
            f"IMU columns missing. Need yaw_rate and accel_long in {list(idx)}"
        )

    for i in range(n):
        lat[i], lon[i] = cur_lat, cur_lon
        accel_long_g = float(X_seg[i, :, i_acc].mean())
        yaw_dps = float(X_seg[i, :, i_yaw].mean())
        vel = float(np.clip(vel + accel_long_g * 9.81 * WINDOW_DT, 0.0, MAX_VEHICLE_SPEED_MPS))
        heading = wrap_heading(heading + yaw_dps * WINDOW_DT)
        cur_lat, cur_lon = propagate(cur_lat, cur_lon, vel, heading, WINDOW_DT)

    return lat, lon


def build_gps_rows(gt_lat, gt_lon, gt_vel, gt_hdg, outage_start, outage_end):
    rows = []
    for i in range(len(gt_lat)):
        lost = outage_start <= i < outage_end
        rows.append({
            "lat": gt_lat[i] if not lost else np.nan,
            "lon": gt_lon[i] if not lost else np.nan,
            "velocity_mps": gt_vel[i],
            "direction_deg": gt_hdg[i],
            "h_acc_m": 1.0 if not lost else 999.0,
        })
    return rows


def run_ai_fusion(X_seg, gps_rows, init_lat, init_lon):
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("sensor_fusion_ekf", ROOT / "scripts" / "08_sensor_fusion_ekf.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)

    pipeline = mod.FusionPipeline(init_lat, init_lon, MODEL_PATH, DATA_PATH)
    results = []
    for window, gps_row in zip(X_seg, gps_rows):
        results.append(pipeline.step(window, gps_row, WINDOW_DT))
    return results


def run_ai_ml_integrate(X_seg, init_lat, init_lon, init_heading_deg):
    """Fallback: integrate LSTM velocity/heading if EKF import fails."""
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("sensor_fusion_ekf", ROOT / "scripts" / "08_sensor_fusion_ekf.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    ml = mod.MLPredictor(MODEL_PATH, DATA_PATH)

    n = X_seg.shape[0]
    lat, lon = np.zeros(n), np.zeros(n)
    cur_lat, cur_lon = init_lat, init_lon
    heading = wrap_heading(init_heading_deg)
    for i in range(n):
        pred = ml.predict(X_seg[i])
        vel = float(np.clip(pred["velocity_mps"], 0.0, MAX_VEHICLE_SPEED_MPS))
        heading = wrap_heading(pred["direction_deg"])
        lat[i], lon[i] = cur_lat, cur_lon
        cur_lat, cur_lon = propagate(cur_lat, cur_lon, vel, heading, WINDOW_DT)
    return lat, lon


def origin_from_existing_csv():
    for path in OUT_PATHS:
        if path.exists():
            try:
                prev = pd.read_csv(path, nrows=1)
                if {"gps_lat", "gps_lng"}.issubset(prev.columns):
                    return float(prev["gps_lat"].iloc[0]), float(prev["gps_lng"].iloc[0])
            except Exception:
                pass
    return DEFAULT_ORIGIN_LAT, DEFAULT_ORIGIN_LON


def blend_gps_reacquire(lat, lon, gps_status, blend_windows=40):
    """Spread the EKF hard-reanchor into a short geodesic so Leaflet polylines stay continuous."""
    lat, lon = np.array(lat, dtype=np.float64), np.array(lon, dtype=np.float64)
    n = len(lat)
    for i in range(1, n):
        if gps_status[i] != "LOCK" or gps_status[i - 1] != "DENIED":
            continue
        end = min(n, i + blend_windows)
        start_lat, start_lon = lat[i - 1], lon[i - 1]
        tgt_lat, tgt_lon = lat[end - 1], lon[end - 1]
        span = end - i
        for k in range(span):
            a = (k + 1) / span
            lat[i + k] = start_lat * (1.0 - a) + tgt_lat * a
            lon[i + k] = start_lon * (1.0 - a) + tgt_lon * a
    return lat, lon


def write_outputs(df: pd.DataFrame):
    written = []
    for path in OUT_PATHS:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment-start", type=int, default=None)
    parser.add_argument("--segment-len", type=int, default=None)
    parser.add_argument("--outage-frac", type=float, nargs=2, default=[0.40, 0.60])
    parser.add_argument("--origin-lat", type=float, default=None)
    parser.add_argument("--origin-lon", type=float, default=None)
    args = parser.parse_args()

    print("Loading leakage-free processed dataset...")
    X_full, y_full, feature_names = load_full_sequence()
    idx = feature_index(feature_names)
    n_feat = X_full.shape[-1]
    print(f"  Windows: {X_full.shape[0]}  |  window={X_full.shape[1]} x {n_feat} features")
    print(f"  Features: {feature_names}")

    if args.segment_start is not None and args.segment_len is not None:
        seg_start = args.segment_start
        seg_end = args.segment_start + args.segment_len
    else:
        seg_start, seg_end = find_longest_continuous_segment(y_full)

    X_seg = X_full[seg_start:seg_end]
    y_seg = y_full[seg_start:seg_end]
    n = X_seg.shape[0]
    print(f"  Continuous segment [{seg_start}:{seg_end}] = {n} windows "
          f"({n * WINDOW_DT:.0f}s = {n * WINDOW_DT / 60:.1f} min)")

    if args.origin_lat is not None and args.origin_lon is not None:
        init_lat, init_lon = args.origin_lat, args.origin_lon
    else:
        init_lat, init_lon = origin_from_existing_csv()

    gt_vel = y_seg[:, 0]
    gt_hdg = y_seg[:, 1]
    print("Reconstructing GPS ground truth from label velocity/heading...")
    gt_lat, gt_lon = reconstruct_gps_path(y_seg, init_lat, init_lon)

    i_wheel = col(idx, "velocity_from_wheels_mps")
    init_vel = float(X_seg[0, :, i_wheel].mean()) if i_wheel is not None else float(gt_vel[0])

    print("Building naive IMU dead reckoning (accel + yaw integration, no ML/GPS)...")
    naive_lat, naive_lon = naive_dead_reckoning_path(
        X_seg, idx, gt_lat[0], gt_lon[0], float(gt_hdg[0]), init_vel
    )

    outage_start = int(args.outage_frac[0] * n)
    outage_end = int(args.outage_frac[1] * n)
    print(f"Simulating GPS outage: windows {outage_start}-{outage_end} "
          f"({(outage_end - outage_start) * WINDOW_DT:.0f}s)")
    gps_rows = build_gps_rows(gt_lat, gt_lon, gt_vel, gt_hdg, outage_start, outage_end)

    print("Running trained LSTM + EKF fusion...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run: python scripts/04_train_lstm.py"
        )
    gps_status = ["DENIED" if outage_start <= i < outage_end else "LOCK" for i in range(n)]
    try:
        ai_results = run_ai_fusion(X_seg, gps_rows, gt_lat[0], gt_lon[0])
        ai_lat = np.array([r["lat"] for r in ai_results], dtype=np.float64)
        ai_lon = np.array([r["lon"] for r in ai_results], dtype=np.float64)
        ai_lat, ai_lon = blend_gps_reacquire(ai_lat, ai_lon, gps_status)
    except Exception as exc:
        print(f"  EKF fusion failed ({exc}); integrating LSTM predictions instead.")
        ai_lat, ai_lon = run_ai_ml_integrate(X_seg, gt_lat[0], gt_lon[0], float(gt_hdg[0]))

    # Leaflet-safe clipping
    for arr in (gt_lat, naive_lat, ai_lat):
        np.clip(arr, -90.0, 90.0, out=arr)
    for arr in (gt_lon, naive_lon, ai_lon):
        np.clip(arr, -180.0, 180.0, out=arr)

    drift_naive = haversine_m(gt_lat, gt_lon, naive_lat, naive_lon)
    drift_ai = haversine_m(gt_lat, gt_lon, ai_lat, ai_lon)

    df = pd.DataFrame({
        "timestamp": np.round(np.arange(n) * WINDOW_DT, 3),
        "gps_lat": gt_lat,
        "gps_lng": gt_lon,
        "naive_lat": naive_lat,
        "naive_lng": naive_lon,
        "ai_lat": ai_lat,
        "ai_lng": ai_lon,
        "gps_status": gps_status,
    })

    written = write_outputs(df)
    print("\nSaved:")
    for path in written:
        print(f"  {path}")
    print(f"  Naive drift  — mean {drift_naive.mean():.1f} m, max {drift_naive.max():.1f} m")
    print(f"  AI drift     — mean {drift_ai.mean():.1f} m, max {drift_ai.max():.1f} m")
    if outage_end > outage_start:
        print(f"  During outage — naive {drift_naive[outage_start:outage_end].mean():.1f} m, "
              f"AI {drift_ai[outage_start:outage_end].mean():.1f} m")


if __name__ == "__main__":
    main()
