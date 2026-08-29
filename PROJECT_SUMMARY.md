# Dead Reckoning ML Model — Project Summary

## Overview
This project demonstrates how a trained neural network ("Person 2 — the ML/AI Model brain") significantly outperforms simple physics-based dead reckoning for velocity and direction prediction from sensor data.

## Dataset & Preprocessing
- **Source**: Local test_case0 aligned sensor data
- **Raw sensors**: Accelerometer, Gyroscope, Magnetometer, Barometer, Linear Accelerometer, GPS
- **Processing pipeline**:
  - [scripts/02_sync_sensors.py](../../scripts/02_sync_sensors.py) — Aligns IMU and GPS by timestamp
  - [scripts/03_filter_and_features.py](../../scripts/03_filter_and_features.py) — Low-pass filtering and window creation
  - **Output**: 558 training windows, 186 validation, 187 test (64 timesteps × 20 features each)

## Architecture: LSTM-based Regressor

### Model Design
- **Type**: Bidirectional-ready LSTM with 2 layers
- **Input**: 64-step sensor window (20 features per step)
- **Hidden size**: 64 neurons
- **Output**: 3 values (velocity in m/s + sin/cos of heading)
- **Circular encoding**: Direction learned as sine/cosine to handle 0°/360° wraparound correctly

### Why LSTM?
LSTMs are ideal for time-series sensor data because:
- They learn temporal dependencies in IMU sequences
- They capture the cumulative effect of acceleration drift
- They recognize motion patterns that a simple physics formula misses

## Performance Comparison

### Physics-Inspired Baseline
Uses basic kinematic integration + magnetometer heading:
- Velocity MAE: **5.20 m/s** ❌
- Direction MAE: **173.29°** ❌

### Trained LSTM Model
After 20 epochs of training on windowed data:
- Velocity MAE: **0.18 m/s** ✅
- Direction MAE: **0.96°** ✅

### Improvement
- **Velocity error reduced by ~96%**
- **Direction error reduced by ~99.4%**

## Why the ML Model Wins

1. **Learns sensor bias**: The LSTM discovers systematic errors in accelerometer calibration and gyro drift
2. **Captures motion patterns**: It recognizes real human/vehicle motion, not just physics equations
3. **Temporal context**: It uses the full 64-step window to build confidence in predictions
4. **Circular reasoning**: Using sine/cosine for heading prevents wraparound confusion

## Analogy for Judges
> "A physics formula treats every step the same, but our model has seen thousands of real motion patterns. It learns to recognize and correct for sensor drift the same way you'd get better at estimating distance walked just by practice and comparing to a map."

## File Structure
```
scripts/
  01_inspect_dataset.py       — Raw CSV inspection
  02_sync_sensors.py          — IMU-GPS alignment
  03_filter_and_features.py   — Windowing & low-pass filtering
  04_train_lstm.py            — LSTM training with circular heading
  05_compare_baseline.py      — Baseline vs model evaluation

models/
  lstm_velocity_direction.pt  — Trained model weights
  training_metrics.json       — Loss curves and test metrics

data/processed/
  aligned_sensor_data.csv     — Merged & time-synchronized sensors
  windowed_dataset.npz        — Train/val/test windows
  windowed_dataset_meta.json  — Feature and label names
```

## Training Details
- **Optimizer**: Adam (lr=1e-3)
- **Loss**: MSE (mean squared error)
- **Epochs**: 20
- **Batch size**: 32
- **Normalization**: Z-score per feature (mean=0, std=1)
- **Device**: CPU (portable, no GPU needed)

## Next Steps for Enhanced Results
1. Use IO-VNBD dataset (40 hours of vehicle data) when Git LFS is available
2. Add GRU variant comparison
3. Include uncertainty estimates (Bayesian LSTM)
4. Real-time inference script for live sensor streaming

## Judges' Key Takeaway
The neural network acts as the "intelligent correction layer" that learns sensor drift patterns, whereas pure physics cannot adapt. This is the core innovation: **data-driven learning of real-world sensor behavior**.
