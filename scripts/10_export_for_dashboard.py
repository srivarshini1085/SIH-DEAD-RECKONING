

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

DATA_PATH = ROOT / "data" / "processed" / "windowed_dataset_IO-VNBD.npz"
MODEL_PATH = ROOT / "models" / "lstm_velocity_direction_io_vnbd.pt"
OUT_CSV = ROOT / "frontend_data" / "sample_data.csv"

IDX_VEL, IDX_HDG = 0, 1
IDX_YAW = 8
IDX_ACC_LONG, IDX_ACC_LAT = 9, 10
IDX_LAT, IDX_LON = 13, 14

WINDOW_DT = 1.28
EARTH_M_PER_DEG_LAT = 111320.0
SESSION_BREAK_M = 300.0


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


def load_full_sequence():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Real dataset not found: {DATA_PATH}")
    d = np.load(DATA_PATH)
    X_full = np.concatenate([d["X_train"], d["X_val"], d["X_test"]], axis=0)
    y_full = np.concatenate([d["y_train"], d["y_val"], d["y_test"]], axis=0)
    return X_full, y_full


def find_longest_continuous_segment(X_full, max_gap_m=SESSION_BREAK_M):
    lat = X_full[:, :, IDX_LAT].mean(axis=1)
    lon = X_full[:, :, IDX_LON].mean(axis=1)
    step_dist = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    breaks = np.where(step_dist > max_gap_m)[0]
    bounds = [0] + list(breaks + 1) + [len(lat)]
    seg_lens = np.diff(bounds)
    best = int(np.argmax(seg_lens))
    return bounds[best], bounds[best + 1]


def ground_truth_path(X_seg):
    lat = X_seg[:, :, IDX_LAT].mean(axis=1)
    lon = X_seg[:, :, IDX_LON].mean(axis=1)
    return lat, lon


def naive_dead_reckoning_path(X_seg, init_lat, init_lon, init_heading_deg, init_vel):
    n = X_seg.shape[0]
    lat, lon = np.zeros(n), np.zeros(n)
    cur_lat, cur_lon = init_lat, init_lon
    heading = init_heading_deg
    vel = init_vel

    for i in range(n):
        accel_long_mean = float(X_seg[i, :, IDX_ACC_LONG].mean())
        yaw_mean = float(X_seg[i, :, IDX_YAW].mean())

        vel = max(0.0, vel + accel_long_mean * 9.81 * WINDOW_DT)
        heading = (heading + yaw_mean * WINDOW_DT) % 360.0

        cur_lat, cur_lon = propagate(cur_lat, cur_lon, vel, heading, WINDOW_DT)
        lat[i], lon[i] = cur_lat, cur_lon

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment-start", type=int, default=None)
    parser.add_argument("--segment-len", type=int, default=None)
    parser.add_argument("--outage-frac", type=float, nargs=2, default=[0.40, 0.60])
    args = parser.parse_args()

    print("Loading real preprocessed data...")
    X_full, y_full = load_full_sequence()
    print(f"  Full sequence: {X_full.shape[0]} windows")

    if args.segment_start is not None and args.segment_len is not None:
        seg_start, seg_end = args.segment_start, args.segment_start + args.segment_len
    else:
        seg_start, seg_end = find_longest_continuous_segment(X_full)
    X_seg = X_full[seg_start:seg_end]
    n = X_seg.shape[0]
    print(f"  Using continuous segment [{seg_start}:{seg_end}] = {n} windows "
          f"({n * WINDOW_DT:.0f}s = {n * WINDOW_DT / 60:.1f} min of real recorded driving)")

    outage_start = int(args.outage_frac[0] * n)
    outage_end = int(args.outage_frac[1] * n)

    gt_lat, gt_lon = ground_truth_path(X_seg)
    gt_vel = X_seg[:, :, IDX_VEL].mean(axis=1)
    gt_hdg = X_seg[:, :, IDX_HDG].mean(axis=1)

    print("Building naive physics baseline (double integration, no ML/GPS)...")
    naive_lat, naive_lon = naive_dead_reckoning_path(
        X_seg, gt_lat[0], gt_lon[0], gt_hdg[0], gt_vel[0]
    )

    print(f"Simulating GPS outage: windows {outage_start}-{outage_end} "
          f"({(outage_end - outage_start) * WINDOW_DT:.0f}s)")
    gps_rows = build_gps_rows(gt_lat, gt_lon, gt_vel, gt_hdg, outage_start, outage_end)

    print("Running trained LSTM + EKF fusion (models/lstm_velocity_direction_io_vnbd.pt)...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run: python scripts/04_train_lstm.py"
        )
    ai_results = run_ai_fusion(X_seg, gps_rows, gt_lat[0], gt_lon[0])
    ai_lat = np.array([r["lat"] for r in ai_results])
    ai_lon = np.array([r["lon"] for r in ai_results])
    ai_vel = np.array([r["velocity_mps"] for r in ai_results])
    ai_hdg = np.array([r["heading_deg"] for r in ai_results])
    source = [r["source"] for r in ai_results]

    print("Scoring drift against real GPS ground truth...")
    drift_naive = haversine_m(gt_lat, gt_lon, naive_lat, naive_lon)
    drift_ai = haversine_m(gt_lat, gt_lon, ai_lat, ai_lon)

    df = pd.DataFrame({
        "time": np.arange(n),
        "gps_lat": gt_lat,
        "gps_lng": gt_lon,
        "naive_lat": naive_lat,
        "naive_lng": naive_lon,
        "ai_lat": ai_lat,
        "ai_lng": ai_lon,
        "gps_status": ["DENIED" if outage_start <= i < outage_end else "LOCK" for i in range(n)],
        "drift_naive_m": drift_naive,
        "drift_ai_m": drift_ai,
        "velocity_mps": ai_vel,
        "heading_deg": ai_hdg,
        "source": source,
    })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nSaved: {OUT_CSV}")
    print(f"  Naive drift  — mean {drift_naive.mean():.1f} m, max {drift_naive.max():.1f} m")
    print(f"  AI drift     — mean {drift_ai.mean():.1f} m, max {drift_ai.max():.1f} m")
    print(f"  During outage — naive {drift_naive[outage_start:outage_end].mean():.1f} m, "
          f"AI {drift_ai[outage_start:outage_end].mean():.1f} m")


if __name__ == "__main__":
    main()