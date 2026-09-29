"""
Full Dead Reckoning + GNSS Fusion Pipeline
==========================================
ISRO SIH 2026 - Problem Statement 26168

Complete chain:
  Raw sensor data
    -> 10_calibration.py   (phone orientation correction)
    -> LSTM model          (velocity + heading prediction)
    -> 08_sensor_fusion_ekf.py (EKF GNSS+INS fusion)
    -> 11_map_matching.py  (road snapping + NHC)
    -> data/processed/final_trajectory.csv
    -> (optional) 12_live_tracking_server.py (live WebSocket push)

Output CSV columns (frontend-compatible):
  time, gps_lat, gps_lng, naive_lat, naive_lng, ai_lat, ai_lng, gps_status

Performance benchmark printed:
  Drift % = mean positional error during GPS-denied window
            / total distance travelled during that window * 100
  Target: < 10%

Usage:
    py scripts/09_run_fusion_pipeline.py
    py scripts/09_run_fusion_pipeline.py --outage-start 100 --outage-end 200 --plot
    py scripts/09_run_fusion_pipeline.py --plot --live
    py scripts/09_run_fusion_pipeline.py --no-map-match
    py scripts/09_run_fusion_pipeline.py --outage-frac 0.3 0.6
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

DATA_PATH  = ROOT / "data"   / "processed" / "windowed_dataset_IO-VNBD.npz"
MODEL_PATH = ROOT / "models" / "lstm_velocity_direction_io_vnbd.pt"
STATS_PATH = ROOT / "models" / "normalization_stats.npz"
OUT_CSV    = ROOT / "data"   / "processed" / "final_trajectory.csv"
TRAJ_PNG   = ROOT / "models" / "fusion_trajectory.png"

EARTH_M_PER_DEG_LAT  = 111320.0
WINDOW_DT            = 1.28
MAX_VEHICLE_SPEED_MPS = 50.0

ORIGIN_LAT = 52.5184417671875
ORIGIN_LON = -1.5083977265625


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full IDR fusion pipeline with CSV export.")
    p.add_argument("--outage-start", type=int, default=None)
    p.add_argument("--outage-end",   type=int, default=None)
    p.add_argument("--outage-frac",  type=float, nargs=2, default=[0.40, 0.60],
                   metavar=("START_FRAC", "END_FRAC"))
    p.add_argument("--plot",         action="store_true")
    p.add_argument("--live",         action="store_true")
    p.add_argument("--no-map-match", action="store_true")
    p.add_argument("--port",         type=int, default=8000)
    return p.parse_args()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi   = np.radians(lat2 - lat1)
    dlmb   = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2)**2
    return 2 * R * np.arcsin(np.sqrt(np.minimum(1.0, a)))


def propagate(lat, lon, vel, heading_deg, dt):
    hdg     = np.deg2rad(heading_deg)
    lat_rad = np.deg2rad(lat)
    dlat = (vel * np.cos(hdg) * dt) / EARTH_M_PER_DEG_LAT
    dlon = (vel * np.sin(hdg) * dt) / (EARTH_M_PER_DEG_LAT * max(np.cos(lat_rad), 1e-6))
    return lat + dlat, lon + dlon


def wrap_heading(deg):
    return float(np.mod(deg, 360.0))


def load_test_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Processed dataset not found: %s" % DATA_PATH)
    d = np.load(DATA_PATH)
    return d["X_test"], d["y_test"]


def reconstruct_gps_path(y: np.ndarray, init_lat=ORIGIN_LAT, init_lon=ORIGIN_LON):
    """
    SIMULATED GPS: integrates label velocity/heading from a fixed origin.
    IO-VNBD has no absolute GPS coords in X after leakage fix.
    Real deployment uses live GPS coordinates.
    """
    n = len(y)
    lats, lons = np.zeros(n), np.zeros(n)
    cur_lat, cur_lon = init_lat, init_lon
    for i in range(n):
        vel = float(np.clip(y[i, 0], 0.0, MAX_VEHICLE_SPEED_MPS))
        hdg = wrap_heading(y[i, 1])
        lats[i], lons[i] = cur_lat, cur_lon
        cur_lat, cur_lon = propagate(cur_lat, cur_lon, vel, hdg, WINDOW_DT)
    return lats, lons


def naive_dead_reckoning(X: np.ndarray, init_lat, init_lon, init_heading_deg, init_vel):
    """
    Physics-only baseline: integrate raw accel_long + yaw_rate.
    Feature order (leakage-free 12-col dataset):
      ws_fl(0), ws_fr(1), ws_rl(2), ws_rr(3), wheel_speed_avg(4),
      velocity_from_wheels_mps(5), yaw_rate(6), accel_long(7),
      accel_lat(8), steering_angle(9), engine_rpm(10), height_km(11)
    """
    n = len(X)
    lats, lons = np.zeros(n), np.zeros(n)
    cur_lat, cur_lon = init_lat, init_lon
    heading = wrap_heading(init_heading_deg)
    vel     = float(np.clip(init_vel, 0.0, MAX_VEHICLE_SPEED_MPS))

    n_feat = X.shape[-1]
    i_yaw  = 6 if n_feat > 6 else 0
    i_acc  = 7 if n_feat > 7 else min(1, n_feat - 1)

    for i in range(n):
        lats[i], lons[i] = cur_lat, cur_lon
        accel_long_g = float(X[i, :, i_acc].mean())
        yaw_dps      = float(X[i, :, i_yaw].mean())
        vel     = float(np.clip(vel + accel_long_g * 9.81 * WINDOW_DT, 0.0, MAX_VEHICLE_SPEED_MPS))
        heading = wrap_heading(heading + yaw_dps * WINDOW_DT)
        cur_lat, cur_lon = propagate(cur_lat, cur_lon, vel, heading, WINDOW_DT)

    return lats, lons


def build_gps_rows(gt_lats, gt_lons, y, outage_start, outage_end):
    rows = []
    for i in range(len(gt_lats)):
        lost = outage_start <= i < outage_end
        rows.append({
            "lat":           gt_lats[i] if not lost else np.nan,
            "lon":           gt_lons[i] if not lost else np.nan,
            "velocity_mps":  float(y[i, 0]),
            "direction_deg": float(y[i, 1]),
            "h_acc_m":       1.0 if not lost else 999.0,
        })
    return rows


def run_ekf_fusion(X, gps_rows, init_lat, init_lon):
    ekf_mod  = _load_module("sensor_fusion_ekf", ROOT / "scripts" / "08_sensor_fusion_ekf.py")
    pipeline = ekf_mod.FusionPipeline(init_lat, init_lon, MODEL_PATH, DATA_PATH)
    results  = []
    for window, gps_row in zip(X, gps_rows):
        results.append(pipeline.step(window, gps_row, WINDOW_DT))
    return results


def run_map_matching(ai_lats, ai_lons, use_osm=True):
    mm_mod = _load_module("map_matching", ROOT / "scripts" / "11_map_matching.py")
    return mm_mod.map_match(ai_lats, ai_lons, use_osm=use_osm)


def compute_drift_benchmark(gt_lats, gt_lons, ai_lats, ai_lons, outage_start, outage_end):
    """SIH benchmark: drift% = mean error / total distance * 100. Target < 10%."""
    sl = slice(outage_start, outage_end)
    errors = haversine_m(gt_lats[sl], gt_lons[sl], ai_lats[sl], ai_lons[sl])

    gt_lat_seg = gt_lats[sl]
    gt_lon_seg = gt_lons[sl]
    if len(gt_lat_seg) < 2:
        return 0.0, 0.0, 0.0, 0.0

    seg_dists    = haversine_m(gt_lat_seg[:-1], gt_lon_seg[:-1], gt_lat_seg[1:], gt_lon_seg[1:])
    total_dist_m = float(seg_dists.sum())
    mean_err     = float(errors.mean())
    max_err      = float(errors.max())
    drift_pct    = (mean_err / max(total_dist_m, 1.0)) * 100.0
    return mean_err, max_err, total_dist_m, drift_pct


def save_plot(gt_lats, gt_lons, naive_lats, naive_lons, ai_lats, ai_lons,
              outage_start, outage_end, ekf_results):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed - skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(gt_lons,    gt_lats,    "g-",  lw=1.5, label="GPS ground truth")
    ax.plot(naive_lons, naive_lats, "r--", lw=1,   label="Naive DR (physics)")
    ax.plot(ai_lons,    ai_lats,    "b-",  lw=2,   label="AI (LSTM+EKF+MapMatch)")
    ax.plot(ai_lons[outage_start:outage_end], ai_lats[outage_start:outage_end],
            "m-", lw=2.5, label="GPS denied (AI DR)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Trajectory Comparison")
    ax.legend(fontsize=8)

    ax = axes[1]
    vels = [r["velocity_mps"] for r in ekf_results]
    ax.plot(vels, "b-", lw=1, label="EKF velocity")
    ax.axvspan(outage_start, outage_end, alpha=0.2, color="red", label="GPS outage")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Velocity - EKF Output")
    ax.legend(fontsize=8)

    plt.tight_layout()
    TRAJ_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(TRAJ_PNG, dpi=120)
    plt.close()
    print("  Plot saved -> %s" % TRAJ_PNG)


def main():
    args = parse_args()

    # Optional: start live tracking server
    if args.live:
        print("\n-- Starting live tracking server --")
        live_mod = _load_module("live_tracking_server",
                                ROOT / "scripts" / "12_live_tracking_server.py")
        live_mod.start_server_background(port=args.port)
    else:
        live_mod = None

    # Load data
    print("\n-- Loading processed dataset --")
    X_test, y_test = load_test_data()
    n = len(X_test)
    print("  Test windows: %d  |  features: %d  |  window len: %d" % (
        n, X_test.shape[-1], X_test.shape[1]))

    # GPS outage window
    if args.outage_start is not None and args.outage_end is not None:
        outage_start = args.outage_start
        outage_end   = args.outage_end
    else:
        outage_start = int(args.outage_frac[0] * n)
        outage_end   = int(args.outage_frac[1] * n)
    outage_end = min(outage_end, n)
    print("  GPS outage: windows %d-%d (%.0fs)" % (
        outage_start, outage_end, (outage_end - outage_start) * WINDOW_DT))

    # Reconstruct GPS ground truth
    # NOTE: GPS coords are SIMULATED by integrating label velocity/heading
    # from a fixed origin. IO-VNBD has no absolute GPS in X after leakage fix.
    print("\n-- Reconstructing GPS ground truth (simulated from labels) --")
    gt_lats, gt_lons = reconstruct_gps_path(y_test)

    # Naive physics baseline
    print("\n-- Computing naive physics baseline --")
    naive_lats, naive_lons = naive_dead_reckoning(
        X_test, gt_lats[0], gt_lons[0],
        float(y_test[0, 1]), float(y_test[0, 0]),
    )

    # EKF fusion (LSTM + GPS)
    print("\n-- Running LSTM + EKF fusion --")
    gps_rows    = build_gps_rows(gt_lats, gt_lons, y_test, outage_start, outage_end)
    ekf_results = run_ekf_fusion(X_test, gps_rows, gt_lats[0], gt_lons[0])

    ai_lats = np.array([r["lat"]          for r in ekf_results], dtype=np.float64)
    ai_lons = np.array([r["lon"]          for r in ekf_results], dtype=np.float64)
    ai_vels = np.array([r["velocity_mps"] for r in ekf_results], dtype=np.float64)
    ai_hdgs = np.array([r["heading_deg"]  for r in ekf_results], dtype=np.float64)
    sources = [r["source"] for r in ekf_results]

    # Map-matching
    if not args.no_map_match:
        print("\n-- Running map-matching + NHC --")
        ai_lats, ai_lons = run_map_matching(ai_lats, ai_lons, use_osm=True)
    else:
        print("\n-- Map-matching skipped (--no-map-match) --")

    np.clip(ai_lats,     -90.0,  90.0, out=ai_lats)
    np.clip(ai_lons,    -180.0, 180.0, out=ai_lons)
    np.clip(naive_lats,  -90.0,  90.0, out=naive_lats)
    np.clip(naive_lons, -180.0, 180.0, out=naive_lons)

    gps_status = ["DENIED" if outage_start <= i < outage_end else "LOCK" for i in range(n)]

    # Live push
    if live_mod is not None:
        print("\n-- Pushing updates to live tracking server --")
        drift_arr = haversine_m(gt_lats, gt_lons, ai_lats, ai_lons)
        for i in range(n):
            live_mod.push_update({
                "time":         round(i * WINDOW_DT, 3),
                "lat":          float(ai_lats[i]),
                "lng":          float(ai_lons[i]),
                "status":       gps_status[i],
                "drift_m":      float(drift_arr[i]),
                "velocity_mps": float(ai_vels[i]),
                "heading_deg":  float(ai_hdgs[i]),
            })

    # Save final CSV
    print("\n-- Saving final_trajectory.csv --")
    df = pd.DataFrame({
        "time":       np.round(np.arange(n) * WINDOW_DT, 3),
        "gps_lat":    gt_lats,
        "gps_lng":    gt_lons,
        "naive_lat":  naive_lats,
        "naive_lng":  naive_lons,
        "ai_lat":     ai_lats,
        "ai_lng":     ai_lons,
        "gps_status": gps_status,
    })
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print("  Saved -> %s  (%d rows, 8 columns)" % (OUT_CSV, len(df)))

    # Drift benchmark
    mean_err,   max_err,   total_dist, drift_pct  = compute_drift_benchmark(
        gt_lats, gt_lons, ai_lats,    ai_lons,    outage_start, outage_end)
    naive_mean, naive_max, _,          naive_pct  = compute_drift_benchmark(
        gt_lats, gt_lons, naive_lats, naive_lons, outage_start, outage_end)

    outage_s   = (outage_end - outage_start) * WINDOW_DT
    target_met = drift_pct < 10.0

    print("\n" + "="*60)
    print("  SIH PERFORMANCE BENCHMARK - Dead Reckoning Drift")
    print("="*60)
    print("  GPS outage duration   : %.1f s  (%d windows)" % (outage_s, outage_end - outage_start))
    print("  Distance during outage: %.1f m" % total_dist)
    print()
    print("  Naive DR (physics only):")
    print("    Mean error : %.2f m" % naive_mean)
    print("    Max  error : %.2f m" % naive_max)
    print("    Drift %%    : %.2f%%" % naive_pct)
    print()
    print("  AI DR (LSTM + EKF + Map-Match):")
    print("    Mean error : %.2f m" % mean_err)
    print("    Max  error : %.2f m" % max_err)
    print("    Drift %%    : %.2f%%  <- %s" % (drift_pct, "PASS (<10%)" if target_met else "FAIL (>10%)"))
    print()
    print("  SIH target: < 10%% drift  ->  %s" % ("MET" if target_met else "NOT MET"))
    print("="*60)

    dr_steps  = sum(1 for s in sources if s == "ML_DEAD_RECKONING")
    gps_steps = sum(1 for s in sources if s == "GPS+ML")
    print("\n  Total steps    : %d" % n)
    print("  GPS+ML steps   : %d" % gps_steps)
    print("  Dead reckoning : %d" % dr_steps)

    if args.plot:
        print("\n-- Saving trajectory plot --")
        save_plot(gt_lats, gt_lons, naive_lats, naive_lons, ai_lats, ai_lons,
                  outage_start, outage_end, ekf_results)

    if args.live:
        print("\n  Live server running at http://localhost:%d/position" % args.port)
        print("  Press Ctrl+C to stop.")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
