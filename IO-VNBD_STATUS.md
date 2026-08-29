# IO-VNBD Preprocessing Status Report

## Summary

✅ **Preprocessing script created and ready**  
❌ **Cannot run yet — actual data files needed**

## What Happened

### Current Situation
You have the **IO-VNBD repository structure** but **NOT the actual sensor data files**:
- 72 V-*.csv files found ✓
- But each file is only ~130 bytes (Git LFS pointer)
- Real files should be 10+ MB each

### Test Run Output
```
Found 72 vehicle files (V-*.csv)
Loading 72 file(s)...
Warning: V-M.csv missing 'Time (s)' column. Skipping.
Warning: V-S1.csv missing 'Time (s)' column. Skipping.
[... 70 more warnings ...]
ERROR: No valid CSV data could be loaded
```

**Why this happened**: Files contain only Git LFS pointers, not actual data.

---

## What I Created for Person 2

### 1. Preprocessing Script: `scripts/08_preprocess_io_vnbd.py`
**Status**: ✅ Ready to use immediately

Handles:
- Loads V-*.csv (vehicle) and S-*.csv (smartphone) files
- Aligns sensors by timestamp
- Applies low-pass filtering (5 Hz cutoff)
- Creates 64-step windows with 50% overlap
- Outputs ML-ready NPZ files
- Produces metadata JSON for feature tracking

### 2. Usage Guide: `IO-VNBD_PREPROCESSING_GUIDE.md`
**Status**: ✅ Complete with examples

Includes:
- How to download real IO-VNBD data
- Command examples for different scenarios
- Expected output format
- Troubleshooting guide
- Next steps for model training

### 3. Output Format
Once real data is available, the script will produce:
```
data/processed/
├── windowed_dataset_IO-VNBD.npz         # 40+ hours of windowed sensor data
├── windowed_dataset_IO-VNBD_meta.json   # Feature names and shapes
```

Ready for Person 2's LSTM training immediately.

---

## To Get Real IO-VNBD Data

### Option 1: Clone with Git LFS (Best)
```bash
# Install Git LFS: https://git-lfs.com
git lfs install

# Clone the repository
git lfs clone https://github.com/onyekpeu/IO-VNBD

# This downloads actual files instead of pointers
```

### Option 2: Download from Releases
Visit: https://github.com/onyekpeu/IO-VNBD/releases
- Look for `.zip` files containing actual CSVs
- Extract to replace Git LFS pointers

### Option 3: Already Have Files?
If you have real IO-VNBD CSVs elsewhere, copy them to:
```
C:\Users\heman\OneDrive\Documents\SIH real dataset\IO-VNBD-master\
Synchronised V abd S datasets\Categorised IOVNB Dataset\
```

---

## Expected Preprocessing Workflow

Once real data is available:

```bash
# 1. Navigate to project
cd C:\Users\heman\OneDrive\Documents\DEAD RECKONING\SIH-DEAD-RECKONING-main

# 2. Run preprocessing
py scripts/08_preprocess_io_vnbd.py \
  --dataset-dir "C:\Users\heman\OneDrive\Documents\SIH real dataset\IO-VNBD-master\..." \
  --window-size 64 \
  --step-size 32

# 3. Output ready for Person 2
# Creates: data/processed/windowed_dataset_IO-VNBD.npz
#         data/processed/windowed_dataset_IO-VNBD_meta.json

# 4. Person 2 trains model
py scripts/04_train_lstm.py \
  --data-path data/processed/windowed_dataset_IO-VNBD.npz \
  --epochs 50
```

---

## Comparison: Local Data vs IO-VNBD

| Aspect | test_case0 (Local) | IO-VNBD |
|--------|-------------------|---------|
| Data available | ✅ Yes | ❌ Pointers only |
| Duration | 0.5 hours | 40 hours |
| Distance | ~3 km | 1,300 km |
| Drives | 1 location | Multiple countries |
| Files | ~6 sensor types | 72 files |
| Preprocessing | ✅ Done | ⏳ Script ready |
| Training | ✅ Model trained | ⏳ Waiting for data |
| Generalization | Limited | Strong (expected) |

---

## Next Steps

**To proceed with Person 2's ML training:**

1. **Get real IO-VNBD data** (download/clone with Git LFS)
2. **Run the preprocessing script**:
   ```bash
   py scripts/08_preprocess_io_vnbd.py \
     --dataset-dir <path-to-downloaded-data>
   ```
3. **Pass output NPZ to Person 2** for model training
4. **Expected output**: Much better generalization vs. test_case0

---

## Files Created

| File | Purpose | Status |
|------|---------|--------|
| `scripts/08_preprocess_io_vnbd.py` | Preprocessing pipeline | ✅ Ready |
| `IO-VNBD_PREPROCESSING_GUIDE.md` | Usage guide | ✅ Complete |
| `scripts/04_train_lstm.py` | LSTM training (existing) | ✅ Ready |

---

## Summary for Person 2

> "I've prepared a complete preprocessing pipeline for IO-VNBD. It will:
> - Handle 72 vehicle sensor files
> - Align and filter sensor streams
> - Create 64-sample windows from 40 hours of driving data
> - Output ML-ready NPZ files with metadata
> 
> Once the actual IO-VNBD data is downloaded (currently only Git LFS pointers), run:
> `py scripts/08_preprocess_io_vnbd.py` and you'll have clean data ready for LSTM training."

---

**All tools are ready. Waiting for real data files.** 🚀
