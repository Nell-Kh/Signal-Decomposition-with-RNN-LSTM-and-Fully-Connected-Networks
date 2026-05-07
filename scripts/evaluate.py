"""Evaluate trained models on the test set."""

import json
import numpy as np
from pathlib import Path
from signal_decomp.sdk.sdk import SignalDecompSDK
from signal_decomp.shared.config import load_config

def main():
    print("Loading test data...")
    config = load_config()
    
    x_fc = np.load("data/X_FC.npy")
    x_seq = np.load("data/X_seq.npy")
    y = np.load("data/Y.npy")

    split = int(len(y) * config["dataset"]["train_split"])
    datasets = {
        "FC": (x_fc[split:], y[split:]),
        "RNN": (x_seq[split:], y[split:]),
        "LSTM": (x_seq[split:], y[split:]),
    }

    with open("data/meta.json") as f:
        meta = json.load(f)

    sdk = SignalDecompSDK(config, phases=meta["phases"])
    
    print("\n--- Final Evaluation (Unseen Test Data) ---")
    for model_name in ["FC", "RNN", "LSTM"]:
        weight_path = Path("weights") / f"{model_name}.pth"
        if not weight_path.exists():
            print(f"[{model_name}] Weights not found at {weight_path}!")
            continue
            
        print(f"Evaluating {model_name}...")
        model = sdk.load_model(model_name, weight_path)
        x_test, y_test = datasets[model_name]
        
        mse = sdk.test(model, x_test, y_test)
        print(f"> {model_name} Test MSE: {mse:.6f}\n")

if __name__ == "__main__":
    main()
