"""
Real-Time Inference for Dead Reckoning Model
=============================================

This module provides a production-ready inference class for making real-time 
predictions using the trained LSTM model.

Usage:
------
# Single window prediction
predictor = DeadReckoningPredictor()
result = predictor.predict(sensor_window)  # shape: (64, 20)
print(result['velocity_mps'])  # velocity in m/s
print(result['direction_deg'])  # direction in degrees (0-360)

# Batch prediction
results = predictor.predict_batch(windows)  # shape: (N, 64, 20)
# Returns: (N, 2) array with [velocity, direction] for each window
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "lstm_velocity_direction_io_vnbd.pt"
STATS_PATH = ROOT / "data" / "processed" / "windowed_dataset_IO-VNBD.npz"


class SequenceRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :]))


def decode_direction_from_sin_cos(vec):
    """Convert sine/cosine back to degrees (0-360)."""
    rad = np.arctan2(vec[:, 0], vec[:, 1])
    return np.rad2deg(rad) % 360.0


class DeadReckoningPredictor:
    def __init__(self, model_path=MODEL_PATH, stats_path=STATS_PATH):
        self.device = torch.device("cpu")
        
        # Load normalization statistics from training data
        data = np.load(stats_path)
        X_train = data["X_train"]
        self.mean = X_train.mean(axis=(0, 1), keepdims=True)
        self.std = X_train.std(axis=(0, 1), keepdims=True)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        
        # Load model (input_size from data shape)
        input_size = X_train.shape[-1]  # Get from actual data
        self.model = SequenceRegressor(input_size=input_size, hidden_size=32, output_size=3).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, sensor_window):
        """
        Predict velocity and direction from a sensor window.
        
        Args:
            sensor_window: np.ndarray of shape (64, num_features) with 64 timesteps and sensor features
        
        Returns:
            dict with 'velocity_mps' and 'direction_deg'
        """
        expected_features = self.mean.shape[-1]
        if sensor_window.shape[0] != 64 or sensor_window.shape[1] != expected_features:
            raise ValueError(f"Expected shape (64, {expected_features}), got {sensor_window.shape}")
        
        # Normalize using training statistics (mean and std have shape (1, 1, 20))
        X_norm = (sensor_window - self.mean.squeeze()) / self.std.squeeze()
        
        # Convert to torch tensor and predict (add batch dimension)
        with torch.no_grad():
            X_t = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
            pred = self.model(X_t).cpu().numpy()
        
        # Decode predictions (pred has shape (1, 3))
        velocity = float(pred[0, 0])
        sin_cos = pred[0, 1:].reshape(1, 2)  # Shape (1, 2)
        direction = float(decode_direction_from_sin_cos(sin_cos)[0])
        
        return {
            "velocity_mps": velocity,
            "direction_deg": direction,
            "raw_output": pred[0]  # [velocity, sin_heading, cos_heading]
        }

    def predict_batch(self, sensor_windows):
        """
        Predict velocity and direction for multiple windows.
        
        Args:
            sensor_windows: np.ndarray of shape (N, 64, 20)
        
        Returns:
            np.ndarray of shape (N, 2) with [velocity, direction] for each window
        """
        if len(sensor_windows.shape) != 3 or sensor_windows.shape[1:] != (64, 20):
            raise ValueError(f"Expected shape (N, 64, 20), got {sensor_windows.shape}")
        
        # Normalize (mean and std have shape (1, 1, 20))
        X_norm = (sensor_windows - self.mean) / self.std
        
        with torch.no_grad():
            X_t = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
            pred = self.model(X_t).cpu().numpy()
        
        vel = pred[:, 0]
        dir_deg = decode_direction_from_sin_cos(pred[:, 1:])
        
        return np.column_stack([vel, dir_deg])


def main():
    parser = argparse.ArgumentParser(
        description="Real-time inference with trained LSTM dead reckoning model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python 07_inference.py --demo                    # Run with random data
  python 07_inference.py --window-file data.npy    # Predict from saved window
        """
    )
    parser.add_argument(
        "--window-file", 
        type=Path, 
        help="Path to a .npy file containing a (64, 20) sensor window."
    )
    parser.add_argument(
        "--demo", 
        action="store_true", 
        help="Run a demo with random sensor data (default if no window-file)"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("DEAD RECKONING MODEL — REAL-TIME INFERENCE")
    print("=" * 80)
    print()
    
    try:
        predictor = DeadReckoningPredictor(model_path=MODEL_PATH, stats_path=STATS_PATH)
        print("✓ Model loaded successfully")
        print(f"  Model: {MODEL_PATH.name}")
        print(f"  Statistics: {STATS_PATH.name}")
        print()
    except FileNotFoundError as e:
        print(f"ERROR: Could not load model or statistics. {e}")
        print("Please ensure you've run:")
        print("  1. python 04_train_lstm.py  (to train the model)")
        print("  2. python 03_filter_and_features.py  (to generate data statistics)")
        exit(1)

    if args.demo or not args.window_file:
        print("=" * 80)
        print("DEMO MODE: Random Sensor Window")
        print("=" * 80)
        print()
        
        # Get correct input size from model
        input_size = predictor.mean.shape[-1]
        demo_window = np.random.randn(64, input_size).astype(np.float32)
        print(f"Generated random sensor window: shape {demo_window.shape}")
        print(f"  64 timesteps × {input_size} sensor features")
        print(f"  Features: acceleration, gyroscope, magnetometer, etc.")
        print()
        
        print("PREDICTION:")
        print("-" * 80)
        try:
            result = predictor.predict(demo_window)
            print(f"  Velocity: {result['velocity_mps']:.4f} m/s")
            print(f"  Direction: {result['direction_deg']:.2f}° (0-360)")
            print(f"  Raw model output: {result['raw_output']}")
            print()
            print("✓ Inference successful!")
        except Exception as e:
            print(f"ERROR during prediction: {e}")
            exit(1)
        
    elif args.window_file:
        print("=" * 80)
        print("LOADING SENSOR WINDOW FROM FILE")
        print("=" * 80)
        print(f"File: {args.window_file}")
        print()
        
        try:
            if not args.window_file.exists():
                raise FileNotFoundError(f"File not found: {args.window_file}")
            
            window = np.load(args.window_file)
            print(f"✓ Loaded: shape {window.shape}, dtype {window.dtype}")
            
            if window.shape != (64, 20):
                print(f"WARNING: Expected shape (64, 20), got {window.shape}")
                print("Attempting to reshape...")
                if window.size == 64 * 20:
                    window = window.reshape(64, 20)
                    print(f"✓ Reshaped to (64, 20)")
                else:
                    raise ValueError(f"Cannot reshape {window.size} elements to (64, 20)")
            
            print()
            print("PREDICTION:")
            print("-" * 80)
            result = predictor.predict(window)
            print(f"  Velocity: {result['velocity_mps']:.4f} m/s")
            print(f"  Direction: {result['direction_deg']:.2f}° (0-360)")
            print()
            print("✓ Inference successful!")
            
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            exit(1)
        except Exception as e:
            print(f"ERROR during prediction: {e}")
            exit(1)
    
    print("=" * 80)


if __name__ == "__main__":
    main()
