"""
Fix data leakage in existing windowed_dataset_IO-VNBD.npz
=========================================================
Drops leaky columns (velocity_mps=0, heading_deg=1, lat=13, lon=14)
from X arrays and saves a clean version ready for retraining.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN_NPZ   = ROOT / "data" / "processed" / "windowed_dataset_IO-VNBD.npz"
OUT_NPZ  = ROOT / "data" / "processed" / "windowed_dataset_IO-VNBD.npz"
META     = ROOT / "data" / "processed" / "windowed_dataset_IO-VNBD_meta.json"

# Columns to DROP (they are labels / GPS — leakage)
# Original order: velocity_mps(0), heading_deg(1), ws_fl(2), ws_fr(3),
#   ws_rl(4), ws_rr(5), wheel_speed_avg(6), velocity_from_wheels_mps(7),
#   yaw_rate(8), accel_long(9), accel_lat(10), steering_angle(11),
#   engine_rpm(12), lat(13), lon(14), height_km(15)
DROP_COLS = [0, 1, 13, 14]  # velocity_mps, heading_deg, lat, lon

KEEP_NAMES = [
    "ws_fl", "ws_fr", "ws_rl", "ws_rr",
    "wheel_speed_avg", "velocity_from_wheels_mps",
    "yaw_rate", "accel_long", "accel_lat",
    "steering_angle", "engine_rpm", "height_km",
]

def drop_cols(X):
    all_cols = list(range(X.shape[-1]))
    keep = [c for c in all_cols if c not in DROP_COLS]
    return X[:, :, keep]

print("Loading existing dataset...")
d = np.load(IN_NPZ)
X_train = drop_cols(d["X_train"])
X_val   = drop_cols(d["X_val"])
X_test  = drop_cols(d["X_test"])
y_train = d["y_train"]
y_val   = d["y_val"]
y_test  = d["y_test"]

print(f"Old shape: {d['X_train'].shape}  →  New shape: {X_train.shape}")
print(f"Dropped columns: velocity_mps, heading_deg, lat, lon")

np.savez(OUT_NPZ,
    X_train=X_train, X_val=X_val, X_test=X_test,
    y_train=y_train, y_val=y_val,   y_test=y_test)

meta = {
    "dataset": "IO-VNBD",
    "feature_names": KEEP_NAMES,
    "target_names": ["avg_velocity_mps", "avg_direction_deg"],
    "window_shape": list(X_train.shape[1:]),
    "train_shape": list(X_train.shape),
    "val_shape":   list(X_val.shape),
    "test_shape":  list(X_test.shape),
    "splits": {
        "train_size": int(X_train.shape[0]),
        "val_size":   int(X_val.shape[0]),
        "test_size":  int(X_test.shape[0]),
    }
}
with open(META, "w") as f:
    json.dump(meta, f, indent=2)

print(f"Saved clean dataset → {OUT_NPZ}")
print(f"Updated metadata   → {META}")
print("Now run: py scripts/04_train_lstm.py")
