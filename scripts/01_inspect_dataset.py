import os

import pandas as pd

DATA_PATH = os.path.join("data", "raw", "test_case0")

files = [
    "Accelerometer.csv",
    "Gyroscope.csv",
    "Magnetometer.csv",
    "Barometer.csv",
    "Linear Accelerometer.csv",
    "Location_input.csv",
]

for file in files:
    path = os.path.join(DATA_PATH, file)
    print("\n" + "=" * 60)
    print(file)
    print("=" * 60)

    if not os.path.exists(path):
        print(f"MISSING: {path}")
        continue

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
