"""
Person 3 - Sensor Fusion & Real-Time Pipeline
==============================================

Extended Kalman Filter (EKF) that fuses:
  - GPS position/velocity (when available)  -> measurement update
  - ML-predicted heading + wheel odometry velocity -> process model input

State vector: [lat, lon, vel, heading]  (4D)

GPS available  -> EKF predict + EKF update (GPS)
GPS lost       -> EKF predict only (dead reckoning)
GPS returns    -> re-anchor: hard-reset position, keep velocity/heading

KEY DESIGN DECISIONS:
  1. Velocity source: wheel odometry (velocity_from_wheels_mps, feature idx 5)
     is used directly during GPS outage. Wheel speed is far more accurate than
     LSTM-integrated acceleration for distance tracking.
     LSTM velocity is used only as fallback when wheel speed is unavailable.

  2. Heading source: LSTM sin/cos output averaged over a circular buffer (5 steps).
     Raw per-step heading has noise; circular mean reduces drift significantly.

  3. Tighter Q matrix: heading process noise reduced from 0.01 to 0.002,
     reflecting that heading changes smoothly for a vehicle (not random walk).

  4. VelocityBuffer: rolling median suppresses outlier velocity spikes.
"""

import sys
from pathlib import Path
from collections import deque

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

MODEL_PATH = ROOT / "models"  / "lstm_velocity_direction_io_vnbd.pt"
STATS_PATH = ROOT / "data"    / "processed" / "windowed_dataset_IO-VNBD.npz"

# Feature index of wheel-odometry velocity in the 12-col IO-VNBD dataset
# Order: ws_fl(0), ws_fr(1), ws_rl(2), ws_rr(3), wheel_speed_avg(4),
#        velocity_from_wheels_mps(5), yaw_rate(6), accel_long(7), ...
WHEEL_VEL_IDX = 5
YAW_RATE_IDX  = 6


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


class HeadingBuffer:
    """
    Circular buffer for heading smoothing using circular mean.
    Stores sin/cos to correctly average angles near the 0/360 boundary.
    """
    def __init__(self, maxlen=5):
        self._sins = deque(maxlen=maxlen)
        self._coss = deque(maxlen=maxlen)

    def push(self, deg: float):
        rad = np.deg2rad(deg)
        self._sins.append(np.sin(rad))
        self._coss.append(np.cos(rad))

    def mean_deg(self) -> float:
        if not self._sins:
            return 0.0
        return float(np.rad2deg(np.arctan2(
            np.mean(self._sins), np.mean(self._coss)
        )) % 360)


class VelocityBuffer:
    """Rolling median of last N velocity values to suppress outliers."""
    def __init__(self, maxlen=5):
        self._buf = deque(maxlen=maxlen)

    def push(self, v: float):
        self._buf.append(max(0.0, v))

    def median(self) -> float:
        if not self._buf:
            return 0.0
        return float(np.median(list(self._buf)))


class MLPredictor:
    """Wraps the trained PyTorch LSTM for single-window inference."""

    def __init__(self, model_path=MODEL_PATH, stats_path=STATS_PATH):
        data = np.load(stats_path)
        X_tr = data["X_train"]
        self.mean = X_tr.mean(axis=(0, 1))
        self.std  = np.where(X_tr.std(axis=(0, 1)) < 1e-8, 1.0,
                             X_tr.std(axis=(0, 1)))
        input_size = X_tr.shape[-1]

        self.model = SequenceRegressor(input_size, hidden_size=128, output_size=3)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def predict(self, window: np.ndarray) -> dict:
        """
        Returns velocity_mps, direction_deg, and raw sin/cos heading components.
        Also extracts wheel odometry velocity directly from the sensor window
        (feature index 5) as a more accurate velocity source.
        """
        # --- Direct wheel odometry velocity (no ML needed) ---
        n_feat = window.shape[-1]
        if n_feat > WHEEL_VEL_IDX:
            wheel_vel = float(np.median(window[:, WHEEL_VEL_IDX]))
            wheel_vel = max(0.0, wheel_vel)
        else:
            wheel_vel = None

        # --- Yaw rate integration for heading (direct from sensor) ---
        if n_feat > YAW_RATE_IDX:
            yaw_rate_dps = float(np.mean(window[:, YAW_RATE_IDX]))
        else:
            yaw_rate_dps = None

        # --- LSTM for heading (sin/cos output) ---
        x = (window - self.mean) / self.std
        with torch.no_grad():
            out = self.model(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
        out = out.numpy()[0]
        lstm_vel = float(out[0])
        deg = float(np.rad2deg(np.arctan2(out[1], out[2])) % 360)

        return {
            "velocity_mps":      lstm_vel,
            "wheel_vel_mps":     wheel_vel,
            "yaw_rate_dps":      yaw_rate_dps,
            "direction_deg":     deg,
            "sin_heading":       float(out[1]),
            "cos_heading":       float(out[2]),
        }


class EKF:
    def __init__(self, init_lat, init_lon, init_vel=0.0, init_heading=0.0):
        self.x = np.array([init_lat, init_lon, init_vel, init_heading],
                          dtype=np.float64)
        self.P = np.diag([1e-8, 1e-8, 1.0, 0.1])
        # Tighter heading process noise: vehicle heading changes smoothly
        self.Q = np.diag([1e-10, 1e-10, 0.05, 0.002])
        self.R = np.diag([1e-8, 1e-8, 0.04, 0.01])

    def predict(self, dt: float, vel_ml: float, heading_ml_deg: float):
        lat, lon, vel, hdg = self.x
        hdg_ml  = np.deg2rad(heading_ml_deg)
        lat_rad = np.deg2rad(lat)

        dlat = (vel * np.cos(hdg) * dt) / 111320.0
        dlon = (vel * np.sin(hdg) * dt) / (111320.0 * max(np.cos(lat_rad), 1e-6))

        self.x = np.array([lat + dlat, lon + dlon, vel_ml, hdg_ml])

        F = np.eye(4)
        F[0, 2] =  np.cos(hdg) * dt / 111320.0
        F[0, 3] = -vel * np.sin(hdg) * dt / 111320.0
        F[1, 2] =  np.sin(hdg) * dt / (111320.0 * max(np.cos(lat_rad), 1e-6))
        F[1, 3] =  vel * np.cos(hdg) * dt / (111320.0 * max(np.cos(lat_rad), 1e-6))

        self.P = F @ self.P @ F.T + self.Q

    def update(self, gps_lat, gps_lon, gps_vel, gps_heading_deg):
        z = np.array([gps_lat, gps_lon, gps_vel, np.deg2rad(gps_heading_deg)])
        H = np.eye(4)
        y = z - H @ self.x
        y[3] = (y[3] + np.pi) % (2 * np.pi) - np.pi

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def reanchor(self, gps_lat, gps_lon):
        self.x[0] = gps_lat
        self.x[1] = gps_lon
        self.P[0, 0] = 1e-8
        self.P[1, 1] = 1e-8

    @property
    def position(self):
        return {"lat": self.x[0], "lon": self.x[1]}

    @property
    def velocity(self):
        return float(self.x[2])

    @property
    def heading_deg(self):
        return float(np.rad2deg(self.x[3]) % 360)


GPS_WEAK_THRESHOLD = 5.0


class FusionPipeline:
    """
    Combines GPS + ML dead reckoning into one continuous position stream.

    During GPS outage:
      - Velocity: wheel odometry (direct sensor, very accurate)
      - Heading: LSTM sin/cos output + circular mean smoothing (5-step buffer)
      - Position: EKF integration with tight heading process noise

    During GPS lock:
      - Full EKF update with GPS lat/lon/vel/heading measurement
    """

    def __init__(self, init_lat: float, init_lon: float,
                 model_path=MODEL_PATH, stats_path=STATS_PATH):
        self.ml  = MLPredictor(model_path, stats_path)
        self.ekf = EKF(init_lat, init_lon)
        self._gps_was_lost = False
        self._hdg_buf = HeadingBuffer(maxlen=5)
        self._vel_buf = VelocityBuffer(maxlen=5)
        # Yaw-rate integration heading (independent of LSTM)
        self._yaw_heading = None

    def step(self, sensor_window: np.ndarray, gps_row: dict, dt: float) -> dict:
        ml_out = self.ml.predict(sensor_window)

        # --- Velocity: prefer wheel odometry, fall back to LSTM ---
        raw_vel = ml_out["wheel_vel_mps"] if ml_out["wheel_vel_mps"] is not None \
                  else ml_out["velocity_mps"]
        self._vel_buf.push(raw_vel)
        vel_use = self._vel_buf.median()

        # --- Heading: LSTM circular-mean buffer ---
        self._hdg_buf.push(ml_out["direction_deg"])
        hdg_use = self._hdg_buf.mean_deg()

        # --- GPS status ---
        gps_lost = (gps_row.get("h_acc_m", 999) > GPS_WEAK_THRESHOLD or
                    np.isnan(gps_row.get("lat", np.nan)))

        # EKF predict with best available velocity + smoothed heading
        self.ekf.predict(dt, vel_use, hdg_use)

        if not gps_lost:
            if self._gps_was_lost:
                self.ekf.reanchor(gps_row["lat"], gps_row["lon"])
            self.ekf.update(
                gps_row["lat"], gps_row["lon"],
                gps_row.get("velocity_mps", vel_use),
                gps_row.get("direction_deg", hdg_use),
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
