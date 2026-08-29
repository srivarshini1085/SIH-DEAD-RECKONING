"""
Person 3 — Run Fusion Pipeline on Recorded Data
================================================

Simulates a real-time streaming loop over the processed dataset,
demonstrating GPS+ML fusion and GPS-outage dead reckoning.

Usage:
    py scripts/09_run_fusion_pipeline.py
    py scripts/09_run_fusion_pipeline.py --outage-start 100 --outage-end 200
    py scripts/09_run_fusion_pipeline.py --plot
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from importlib.util import spec_from_file_location, module_from_spec as _mfs
_spec = spec_from_file_location("sensor_fusion_ekf", ROOT / "scripts" / "08_sensor_fusion_ekf.py")
_mod  = _mfs(_spec); _spec.loader.exec_module(_mod)
FusionPipeline = _mod.FusionPipeline

DATA_PATH  = ROOT / "data"   / "processed" / "windowed_dataset_IO-VNBD.npz"
MODEL_PATH = ROOT / "models" / "lstm_velocity_direction_io_vnbd.pt"
STATS_PATH = DATA_PATH


def load_data():
    d = np.load(DATA_PATH)
    return d["X_test"], d["y_test"]   # (N, 64, F),  (N, 2) → [vel, dir]


def build_gps_rows(y_test: np.ndarray, outage_start: int, outage_end: int):
    """
    Simulate GPS rows from ground-truth labels.
    Inject a GPS outage between outage_start and outage_end indices.
    """
    rows = []
    # Fake lat/lon origin — real pipeline would use actual GPS coordinates
    lat0, lon0 = 17.3850, 78.4867   # Hyderabad as example origin
    lat, lon = lat0, lon0

    for i, (vel, hdg) in enumerate(y_test):
        dt = 1.28   # 64 samples @ 50 Hz = 1.28 s per window
        hdg_rad = np.deg2rad(hdg)
        lat += (vel * np.cos(hdg_rad) * dt) / 111320.0
        lon += (vel * np.sin(hdg_rad) * dt) / (111320.0 * max(np.cos(np.deg2rad(lat)), 1e-6))

        in_outage = outage_start <= i < outage_end
        rows.append({
            "lat":          lat if not in_outage else np.nan,
            "lon":          lon if not in_outage else np.nan,
            "velocity_mps": vel,
            "direction_deg": hdg,
            "h_acc_m":      1.0 if not in_outage else 999.0,
        })
    return rows, lat0, lon0


def run(outage_start=100, outage_end=150, plot=False):
    print("Loading data...")
    X_test, y_test = load_data()
    gps_rows, init_lat, init_lon = build_gps_rows(y_test, outage_start, outage_end)

    print(f"Initialising FusionPipeline at ({init_lat:.4f}, {init_lon:.4f})")
    print(f"GPS outage simulated: windows {outage_start}-{outage_end}\n")

    pipeline = FusionPipeline(init_lat, init_lon, MODEL_PATH, STATS_PATH)

    results = []
    dt = 1.28

    for i, (window, gps_row) in enumerate(zip(X_test, gps_rows)):
        out = pipeline.step(window, gps_row, dt)
        results.append(out)

        if i % 20 == 0 or outage_start <= i <= outage_end + 2:
            tag = f"[{out['source']:20s}]"
            print(f"  Step {i:04d} {tag}  "
                  f"lat={out['lat']:.6f}  lon={out['lon']:.6f}  "
                  f"vel={out['velocity_mps']:.2f} m/s  "
                  f"hdg={out['heading_deg']:.1f}°")

    # Summary
    dr_steps = sum(1 for r in results if r["source"] == "ML_DEAD_RECKONING")
    fused    = sum(1 for r in results if r["source"] == "GPS+ML")
    print(f"\nTotal steps : {len(results)}")
    print(f"GPS+ML      : {fused}")
    print(f"Dead reckoning (ML only): {dr_steps}")

    if plot:
        _plot(results, gps_rows, outage_start, outage_end)

    return results


def _plot(results, gps_rows, outage_start, outage_end):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    lats_fused = [r["lat"] for r in results]
    lons_fused = [r["lon"] for r in results]
    lats_gps   = [g["lat"] for g in gps_rows]
    lons_gps   = [g["lon"] for g in gps_rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Trajectory ---
    ax = axes[0]
    ax.plot(lons_fused, lats_fused, "b-", lw=1.5, label="EKF Fused")
    valid_gps = [(g["lon"], g["lat"]) for g in gps_rows if not np.isnan(g["lat"])]
    if valid_gps:
        gx, gy = zip(*valid_gps)
        ax.plot(gx, gy, "g.", ms=3, label="GPS (available)")
    # Highlight outage segment
    seg_lons = lons_fused[outage_start:outage_end]
    seg_lats = lats_fused[outage_start:outage_end]
    ax.plot(seg_lons, seg_lats, "r-", lw=2, label="Dead Reckoning (GPS lost)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Fused Trajectory"); ax.legend(fontsize=8)

    # --- Velocity over time ---
    ax = axes[1]
    ax.plot([r["velocity_mps"] for r in results], "b-", lw=1, label="EKF velocity")
    ax.axvspan(outage_start, outage_end, alpha=0.2, color="red", label="GPS outage")
    ax.set_xlabel("Window index"); ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Velocity — EKF Output"); ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = ROOT / "models" / "fusion_trajectory.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nPlot saved -> {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run EKF sensor fusion pipeline on recorded data.")
    parser.add_argument("--outage-start", type=int, default=100,
                        help="Window index where GPS outage begins (default: 100)")
    parser.add_argument("--outage-end",   type=int, default=150,
                        help="Window index where GPS outage ends   (default: 150)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate and save trajectory + velocity plot")
    args = parser.parse_args()
    run(args.outage_start, args.outage_end, args.plot)


if __name__ == "__main__":
    main()
