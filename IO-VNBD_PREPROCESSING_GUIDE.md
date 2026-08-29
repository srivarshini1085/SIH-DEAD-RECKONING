# IO-VNBD Dataset Preprocessing Guide

## Overview

This guide explains how to preprocess the **IO-VNBD (Inertial and Odometry Vehicle Navigation Benchmark Dataset)** for machine learning model training (Person 2).

The preprocessing pipeline converts raw sensor CSV files into structured ML-ready windows (64 timesteps × 20 features each).

## Prerequisites

**Step 0: Download Real Data**

The IO-VNBD dataset currently in your folder contains **Git LFS pointers** (not real data). You need to:

### Option A: Clone with Git LFS (Recommended)
```bash
# Install Git LFS first: https://git-lfs.com
git lfs install
git clone https://github.com/onyekpeu/IO-VNBD
cd IO-VNBD
git lfs pull
```

### Option B: Download from Releases
Visit: https://github.com/onyekpeu/IO-VNBD/releases and download the data archives

### Option C: Download Pre-Extracted Files
Look for download links in the repo's issues or discussions section

Once downloaded, place the actual CSV files in:
```
C:\Users\heman\OneDrive\Documents\SIH real dataset\IO-VNBD-master\
```

## Data Format Expected

IO-VNBD provides vehicle and smartphone sensor data in separate CSV files:

### Vehicle Data (`V-*.csv`)
File structure:
```
Time (s), X (m/s^2), Y (m/s^2), Z (m/s^2),        # Accelerometer
X (rad/s), Y (rad/s), Z (rad/s),                  # Gyroscope
X (µT), Y (µT), Z (µT),                           # Magnetometer
X (hPa),                                           # Barometer
Latitude (°), Longitude (°), Height (m),          # Position
Velocity (m/s), Direction (°),                    # Velocity & heading (GPS)
Horizontal Accuracy (m), Vertical Accuracy (°)   # Accuracy
```

### Smartphone Data (`S-*.csv`)
Same format, recorded on Android phones at 10 Hz

## Usage

### Run Preprocessing

```bash
cd C:\Users\heman\OneDrive\Documents\DEAD RECKONING\SIH-DEAD-RECKONING-main

# Preprocess vehicle data only
py scripts/08_preprocess_io_vnbd.py \
  --dataset-dir "C:\Users\heman\OneDrive\Documents\SIH real dataset\IO-VNBD-master\Synchronised V abd S datasets\Categorised IOVNB Dataset" \
  --window-size 64 \
  --step-size 32 \
  --output-dir data/processed

# Or with smartphone data included
py scripts/08_preprocess_io_vnbd.py \
  --dataset-dir <path> \
  --use-vehicle \
  --use-smartphone
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset-dir` | `data/raw/IO-VNBD` | Path containing V-*.csv and S-*.csv files |
| `--output-dir` | `data/processed` | Where to save windowed data |
| `--window-size` | 64 | Samples per window (~1.3 sec at 50 Hz) |
| `--step-size` | 32 | Stride between windows (50% overlap) |
| `--test-size` | 0.2 | Fraction for test set |
| `--val-size` | 0.25 | Fraction for validation (from training set) |
| `--use-vehicle` | True | Include vehicle data |
| `--use-smartphone` | False | Also include smartphone data |

## Output Files

The script creates:

1. **`windowed_dataset_IO-VNBD.npz`** — Binary array file containing:
   - `X_train`, `X_val`, `X_test` — Input windows (shape: N × 64 × 20)
   - `y_train`, `y_val`, `y_test` — Labels: [avg_velocity, avg_direction]

2. **`windowed_dataset_IO-VNBD_meta.json`** — Metadata:
   ```json
   {
     "dataset": "IO-VNBD",
     "feature_names": [...],
     "target_names": ["avg_velocity_mps", "avg_direction_deg"],
     "window_shape": [64, 20],
     "train_shape": [...],
     "val_shape": [...],
     "test_shape": [...]
   }
   ```

## Processing Steps

### 1. Load & Align
- Reads V-*.csv and S-*.csv files
- Standardizes column names
- Sorts by timestamp

### 2. Prepare Features
- Converts to numeric types
- Interpolates missing values
- Computes magnitude features: `acc_mag`, `gyro_mag`, `mag_mag`

### 3. Filter
- Applies Butterworth low-pass filter (5 Hz cutoff)
- Removes high-frequency noise while preserving motion

### 4. Create Windows
- Sliding window with configurable stride
- Each window: 64 consecutive timesteps × 20 features
- Labels: average velocity & direction from GPS ground truth

### 5. Split
- 60% training, 20% validation, 20% test (temporal order preserved)
- Shuffling disabled to respect time dependencies

## Dataset Details

### IO-VNBD Coverage
- **40 hours** of vehicle driving data
- **1,300 km** total distance
- **58 hours** of smartphone data (4,400 km)
- **Multiple drivers** in UK, Nigeria, France
- **Varied conditions**: traffic, roundabouts, highway, city driving

### Feature Set (20 per timestep)
```
Accelerometer (4):     a_x, a_y, a_z, acc_mag
Gyroscope (4):         gs_x, gs_y, gs_z, gyro_mag
Magnetometer (4):      m_x, m_y, m_z, mag_mag
Barometer (1):         baro_hpa
GPS Position (3):      lat, lon, height_m
GPS Velocity (4):      velocity_mps, direction_deg, h_acc_m, v_acc_deg
```

## Common Issues

### Issue: "No CSV files found"
**Solution**: Ensure Git LFS data is downloaded (not just pointers). Check file sizes:
```powershell
Get-ChildItem *.csv | Format-Table Name, @{N="SizeMB"; E={[math]::Round($_.Length/1MB,2)}}
# Real files should be 5+ MB, not 0.1 MB
```

### Issue: "Dataset too small for window_size"
**Solution**: Either reduce window size or check that all CSV files were found

### Issue: "Missing Time (s) column"
**Solution**: Check IO-VNBD file format. Should have `Time (s)` in first line

### Issue: ValueError on NaN values
**Solution**: Handled automatically by script (converts to 0)

## Next Step: Model Training

Once preprocessing completes, Person 2 can use the output files for training:

```bash
# The preprocessed data is ready for training
py scripts/04_train_lstm.py \
  --data-path data/processed/windowed_dataset_IO-VNBD.npz \
  --model-path models/lstm_IO-VNBD.pt \
  --epochs 50 \
  --batch-size 32
```

## Expected Results

Using IO-VNBD (40 hours vs our 0.5-hour test_case0):
- Much better generalization
- Lower validation error
- Model can handle diverse driving conditions
- Strong benchmark performance

---

**Summary**: This script is ready to run. Once you have the real IO-VNBD CSV data (40+ MB files), just run the command above and you'll get ML-ready windows for Person 2!
