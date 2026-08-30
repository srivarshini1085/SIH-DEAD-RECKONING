"""
Regenerate sample_data.csv for the frontend dashboard.
Reads the existing CSV (which has correct GPS + AI paths),
recomputes naive DR so it actually drifts instead of freezing.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

FRONTEND_CSV = Path(__file__).resolve().parents[2] / "Dead Recoking Frontend" / "sample_data.csv"
OUT_CSV = FRONTEND_CSV  # overwrite in place

EARTH_M_PER_DEG_LAT = 111320.0


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


def compute_heading(lat1, lon1, lat2, lon2):
    """Bearing from point 1 to point 2 in degrees."""
    dlon = np.radians(lon2 - lon1)
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    y = np.sin(dlon) * np.cos(lat2r)
    x = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def main():
    print(f"Reading: {FRONTEND_CSV}")
    df = pd.read_csv(FRONTEND_CSV)
    n = len(df)

    gps_lat = df["gps_lat"].values
    gps_lng = df["gps_lng"].values
    vel = df["velocity_mps"].values
    hdg = df["heading_deg"].values
    dt = 1.0  # 1 second per row

    # --- Recompute naive DR with realistic drift ---
    # Naive DR: starts correct, uses velocity + heading from sensors
    # but adds cumulative bias: small heading error (0.3 deg/s) + velocity noise (2%)
    rng = np.random.default_rng(42)
    naive_lat = np.zeros(n)
    naive_lng = np.zeros(n)
    cur_lat, cur_lng = gps_lat[0], gps_lng[0]
    cur_hdg = hdg[0]
    cur_vel = vel[0]
    heading_bias = 0.0  # accumulates over time

    for i in range(n):
        naive_lat[i] = cur_lat
        naive_lng[i] = cur_lng

        # Add small sensor errors that accumulate
        heading_bias += rng.normal(0.0, 0.25)          # gyro drift ~0.25 deg/step
        vel_noise = rng.normal(1.0, 0.02)               # 2% velocity scale error

        noisy_hdg = (hdg[i] + heading_bias) % 360.0
        noisy_vel = vel[i] * vel_noise

        cur_lat, cur_lng = propagate(cur_lat, cur_lng, noisy_vel, noisy_hdg, dt)

    # --- Recompute drift ---
    drift_naive = haversine_m(gps_lat, gps_lng, naive_lat, naive_lng)
    drift_ai = haversine_m(gps_lat, gps_lng, df["ai_lat"].values, df["ai_lng"].values)

    df["naive_lat"] = naive_lat
    df["naive_lng"] = naive_lng
    df["drift_naive_m"] = drift_naive
    df["drift_ai_m"] = drift_ai

    df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")
    print(f"  Naive drift — mean {drift_naive.mean():.1f} m, max {drift_naive.max():.1f} m")
    print(f"  AI drift    — mean {drift_ai.mean():.1f} m, max {drift_ai.max():.1f} m")
    outage = df["gps_status"] == "DENIED"
    if outage.any():
        print(f"  During GPS outage ({outage.sum()} rows):")
        print(f"    Naive {drift_naive[outage].mean():.1f} m  vs  AI {drift_ai[outage].mean():.1f} m")


if __name__ == "__main__":
    main()
