"""
Person 3 — Sensor Fusion & Real-Time Pipeline
==============================================

Extended Kalman Filter (EKF) that fuses:
  - GPS position/velocity (when available)  → measurement update
  - ML-predicted velocity/direction (GPS outage) → process model input

State vector: [lat, lon, vel, heading]  (4D)

GPS available  → EKF predict (ML) + EKF update (GPS)
GPS lost       → EKF predict (ML) only  (dead reckoning)
GPS returns    → re-anchor: hard-reset position, keep velocity/heading
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

MODEL_PATH  = ROOT / "models"  / "lstm_velocity_direction_io_vnbd.pt"
STATS_PATH  = ROOT / "data"    / "processed" / "windowed_dataset_IO-VNBD.npz"

# ---------------------------------------------------------------------------
# Re-use the same architecture defined in 04_train_lstm.py
# ---------------------------------------------------------------------------
class SequenceRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :]))


# ---------------------------------------------------------------------------
# ML Model Wrapper  (callable in a streaming loop)
# ---------------------------------------------------------------------------
class MLPredictor:
    """Wraps the trained PyTorch LSTM for single-window inference."""

    def __init__(self, model_path=MODEL_PATH, stats_path=STATS_PATH):
        data = np.load(stats_path)
        X_tr = data["X_train"]
        self.mean = X_tr.mean(axis=(0, 1))          # (F,)
        self.std  = np.where(X_tr.std(axis=(0, 1)) < 1e-8, 1.0,
                             X_tr.std(axis=(0, 1)))  # (F,)
        input_size = X_tr.shape[-1]

        self.model = SequenceRegressor(input_size, hidden_size=128, output_size=3)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def predict(self, window: np.ndarray) -> dict:
        """
        Args:
            window: (64, F) float32 sensor window
        Returns:
            {"velocity_mps": float, "direction_deg": float}
        """
        x = (window - self.mean) / self.std
        with torch.no_grad():
            out = self.model(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
        out = out.numpy()[0]                         # [vel, sin, cos]
        vel = float(out[0])
        deg = float(np.rad2deg(np.arctan2(out[1], out[2])) % 360)
        return {"velocity_mps": vel, "direction_deg": deg}


# ---------------------------------------------------------------------------
# Extended Kalman Filter
# ---------------------------------------------------------------------------
# State:  x = [lat (deg), lon (deg), vel (m/s), heading (rad)]
# Units:  lat/lon in degrees, velocity in m/s, heading in radians
#
# Process model (constant-velocity, heading from ML):
#   lat_new  = lat  + (vel * cos(heading) * dt) / 111320
#   lon_new  = lon  + (vel * sin(heading) * dt) / (111320 * cos(lat_rad))
#   vel_new  = vel_ml          (ML replaces physics integration)
#   hdg_new  = heading_ml
#
# Measurement model (GPS):
#   z = [lat_gps, lon_gps, vel_gps, heading_gps]  → H = I_4
# ---------------------------------------------------------------------------

class EKF:
    def __init__(self, init_lat, init_lon, init_vel=0.0, init_heading=0.0):
        self.x = np.array([init_lat, init_lon, init_vel, init_heading],
                          dtype=np.float64)

        # Covariance — start uncertain about velocity/heading
        self.P = np.diag([1e-8, 1e-8, 1.0, 0.1])

        # Process noise Q  (tuned for vehicle motion)
        self.Q = np.diag([1e-10, 1e-10, 0.25, 0.01])

        # Measurement noise R  (GPS accuracy)
        self.R = np.diag([1e-8, 1e-8, 0.04, 0.01])   # ~1 m pos, 0.2 m/s vel, ~6° hdg

    # ------------------------------------------------------------------
    def predict(self, dt: float, vel_ml: float, heading_ml_deg: float):
        """EKF predict step using ML velocity/heading as process input."""
        lat, lon, vel, hdg = self.x
        hdg_ml = np.deg2rad(heading_ml_deg)

        lat_rad = np.deg2rad(lat)
        dlat = (vel * np.cos(hdg) * dt) / 111320.0
        dlon = (vel * np.sin(hdg) * dt) / (111320.0 * max(np.cos(lat_rad), 1e-6))

        # Predicted state (ML corrects vel & heading)
        self.x = np.array([
            lat + dlat,
            lon + dlon,
            vel_ml,
            hdg_ml,
        ])

        # Jacobian F of process model w.r.t. state
        F = np.eye(4)
        F[0, 2] =  np.cos(hdg) * dt / 111320.0
        F[0, 3] = -vel * np.sin(hdg) * dt / 111320.0
        F[1, 2] =  np.sin(hdg) * dt / (111320.0 * max(np.cos(lat_rad), 1e-6))
        F[1, 3] =  vel * np.cos(hdg) * dt / (111320.0 * max(np.cos(lat_rad), 1e-6))

        self.P = F @ self.P @ F.T + self.Q

    # ------------------------------------------------------------------
    def update(self, gps_lat, gps_lon, gps_vel, gps_heading_deg):
        """EKF update step with GPS measurement."""
        z = np.array([gps_lat, gps_lon, gps_vel,
                      np.deg2rad(gps_heading_deg)])
        H = np.eye(4)
        y = z - H @ self.x

        # Wrap heading residual to [-π, π]
        y[3] = (y[3] + np.pi) % (2 * np.pi) - np.pi

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    # ------------------------------------------------------------------
    def reanchor(self, gps_lat, gps_lon):
        """Hard-reset position when GPS signal returns after outage."""
        self.x[0] = gps_lat
        self.x[1] = gps_lon
        self.P[0, 0] = 1e-8
        self.P[1, 1] = 1e-8

    # ------------------------------------------------------------------
    @property
    def position(self):
        return {"lat": self.x[0], "lon": self.x[1]}

    @property
    def velocity(self):
        return float(self.x[2])

    @property
    def heading_deg(self):
        return float(np.rad2deg(self.x[3]) % 360)


# ---------------------------------------------------------------------------
# Fusion Pipeline  (processes one timestep at a time — streaming-ready)
# ---------------------------------------------------------------------------
GPS_WEAK_THRESHOLD = 5.0   # h_acc_m above this → GPS considered lost

class FusionPipeline:
    """
    Combines GPS + ML dead reckoning into one continuous position stream.

    Usage (streaming loop):
        pipeline = FusionPipeline(init_lat, init_lon)
        for each timestep:
            result = pipeline.step(sensor_window, gps_row, dt)
            print(result["lat"], result["lon"], result["source"])
    """

    def __init__(self, init_lat: float, init_lon: float,
                 model_path=MODEL_PATH, stats_path=STATS_PATH):
        self.ml   = MLPredictor(model_path, stats_path)
        self.ekf  = EKF(init_lat, init_lon)
        self._gps_was_lost = False

    def step(self, sensor_window: np.ndarray, gps_row: dict, dt: float) -> dict:
        """
        Args:
            sensor_window : (64, F) latest sensor window
            gps_row       : dict with keys lat, lon, velocity_mps,
                            direction_deg, h_acc_m (horizontal accuracy)
            dt            : seconds since last call
        Returns:
            dict with lat, lon, velocity_mps, heading_deg, source
        """
        ml_out   = self.ml.predict(sensor_window)
        vel_ml   = ml_out["velocity_mps"]
        hdg_ml   = ml_out["direction_deg"]

        gps_lost = gps_row.get("h_acc_m", 999) > GPS_WEAK_THRESHOLD or \
                   np.isnan(gps_row.get("lat", np.nan))

        # --- EKF predict (always uses ML as process input) ---
        self.ekf.predict(dt, vel_ml, hdg_ml)

        if not gps_lost:
            # GPS available → re-anchor if just recovered, then update
            if self._gps_was_lost:
                self.ekf.reanchor(gps_row["lat"], gps_row["lon"])
            self.ekf.update(
                gps_row["lat"], gps_row["lon"],
                gps_row.get("velocity_mps", vel_ml),
                gps_row.get("direction_deg", hdg_ml),
            )
            source = "GPS+ML"
        else:
            source = "ML_DEAD_RECKONING"

        self._gps_was_lost = gps_lost

        return {
            "lat":          self.ekf.position["lat"],
            "lon":          self.ekf.position["lon"],
            "velocity_mps": self.ekf.velocity,
            "heading_deg":  self.ekf.heading_deg,
            "source":       source,
        }
