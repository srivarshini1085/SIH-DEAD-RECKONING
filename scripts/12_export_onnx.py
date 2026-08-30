"""
Export trained LSTM to ONNX for browser inference via onnxruntime-web.
Run: py scripts/12_export_onnx.py
"""
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "lstm_velocity_direction_io_vnbd.pt"
STATS_PATH = ROOT / "models" / "normalization_stats.npz"
FRONTEND = ROOT.parent / "Dead Recoking Frontend"
ONNX_OUT = FRONTEND / "model.onnx"
STATS_OUT = FRONTEND / "norm_stats.json"


class SequenceRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=2, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :]))


def main():
    stats = np.load(STATS_PATH)
    mean = stats["mean"]   # (1,1,F)
    std  = stats["std"]    # (1,1,F)
    input_size = int(stats["input_size"][0])
    print(f"Input size: {input_size} features")

    model = SequenceRegressor(input_size=input_size, hidden_size=128, output_size=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    dummy = torch.zeros(1, 64, input_size)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model, dummy, str(ONNX_OUT),
            input_names=["sensor_window"],
            output_names=["prediction"],
            dynamic_axes={"sensor_window": {0: "batch"}, "prediction": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
    print("Saved ONNX model:", ONNX_OUT)

    # Save norm stats as JSON so JS can read them directly
    import json
    norm = {
        "mean": mean.squeeze().tolist(),   # list of F floats
        "std":  std.squeeze().tolist(),    # list of F floats
        "input_size": input_size,
        "seq_len": 64,
    }
    with open(STATS_OUT, "w") as f:
        json.dump(norm, f)
    print(f"Saved norm stats: {STATS_OUT}")
    print("Done. Now open index.html via Live Server.")


if __name__ == "__main__":
    main()
