"""Entry point — train all 3 models using saved dataset."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from signal_decomp.sdk.sdk import SignalDecompSDK
from signal_decomp.shared.config import load_config


def main():
    """Load saved dataset, train all 3 models, run analysis."""
    config = load_config()
    Path("results").mkdir(exist_ok=True)

    # Load pre-built dataset — identical data for all 3 models
    print("Loading dataset...")
    x_fc = np.load("data/X_FC.npy")
    x_seq = np.load("data/X_seq.npy")
    y = np.load("data/Y.npy")
    print(f"X_FC:  {x_fc.shape}")
    print(f"X_seq: {x_seq.shape}")
    print(f"Y:     {y.shape}")

    # 80/20 train/test split
    split = int(len(y) * config["dataset"]["train_split"])
    datasets = {
        "FC": (x_fc[:split], x_fc[split:], y[:split], y[split:]),
        "RNN": (x_seq[:split], x_seq[split:], y[:split], y[split:]),
        "LSTM": (x_seq[:split], x_seq[split:], y[:split], y[split:]),
    }

    # Load metadata to get original phases (ensures evaluation uses same signals as training)
    with open("data/meta.json") as f:
        meta = json.load(f)
    phases = meta["phases"]

    sdk = SignalDecompSDK(config, phases=phases)
    trained_models = {}
    all_histories = {}

    # --- Train all 3 models ---
    for model_name in ["FC", "RNN", "LSTM"]:
        print(f"\n{'='*40}")
        print(f"Training {model_name}...")
        print(f"{'='*40}")
        x_train, x_test, y_train, y_test = datasets[model_name]
        model, history = sdk.train(model_name, x_train, y_train)
        trained_models[model_name] = (model, x_test, y_test)
        all_histories[model_name] = history
        print(f"Final training loss: {history[-1]:.6f}")
        
        # Save model weights
        weight_dir = Path("weights")
        weight_dir.mkdir(exist_ok=True)
        sdk.save_model(model, weight_dir / f"{model_name}.pth")
        print(f"Saved weights to {weight_dir}/{model_name}.pth")

    # --- Test MSE on held-out 20% ---
    print("\n--- Test MSE (held-out 20%) ---")
    test_mse_results = {}
    for model_name, (model, x_test, y_test) in trained_models.items():
        mse = sdk.test(model, x_test, y_test)
        test_mse_results[model_name] = mse
        print(f"{model_name}: {mse:.6f}")

    # --- Noise sensitivity analysis ---
    print("\n--- Noise Sensitivity Analysis ---")
    noise_levels = config["analysis"]["noise_levels"]
    noise_results = {name: [] for name in ["FC", "RNN", "LSTM"]}

    for noise in noise_levels:
        print(f"Testing noise {int(noise*100)}%...")
        for model_name, (model, _, _) in trained_models.items():
            mse = sdk.evaluate_at_noise(model, model_name, noise)
            noise_results[model_name].append(mse)

    # Save noise results
    with open("results/noise_sensitivity.json", "w") as f:
        json.dump({
            "noise_levels": noise_levels,
            "mse_results": noise_results,
            "test_mse_baseline": test_mse_results
        }, f, indent=2)
    print("Saved: results/noise_sensitivity.json")

    # --- Plot 1: Training loss curves ---
    plt.figure(figsize=(10, 5))
    for name, history in all_histories.items():
        plt.plot(history, label=name)
    plt.title("Training Loss per Epoch — All Models")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/training_loss.png", dpi=150)
    print("Saved: results/training_loss.png")

    # --- Plot 2: Noise sensitivity ---
    plt.figure(figsize=(10, 5))
    noise_pct = [n * 100 for n in noise_levels]
    for name in ["FC", "RNN", "LSTM"]:
        plt.plot(noise_pct, noise_results[name], marker="o", label=name)
    plt.title("Reconstruction Error vs Noise Level")
    plt.xlabel("Amplitude Noise Level (%)")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/noise_sensitivity.png", dpi=150)
    print("Saved: results/noise_sensitivity.png")

    # --- Plot 3: Visual Reconstruction Example (The Test) ---
    print("\nGenerating visual reconstruction samples...")
    import torch
    plt.figure(figsize=(15, 10))
    # Pick a random sample from the test set
    idx = np.random.randint(0, len(y_test))
    
    # 1. Noisy Input (Sequence for RNN/LSTM, Flat for FC)
    # We'll plot the noisy sequence component from x_seq[idx] (last column is usually the noisy signal)
    # Actually, let's just get the raw noisy signal from apply_noise if we had it, 
    # but we can reconstruct it from the window.
    noisy_segment = datasets["RNN"][1][idx, :, -1] # (W, 5) -> last col is noisy sig
    clean_target = y_test[idx]
    
    plt.subplot(4, 1, 1)
    plt.plot(noisy_segment, 'k--', alpha=0.6, label="Noisy Input")
    plt.plot(clean_target, 'g', linewidth=2, label="Clean Target")
    plt.title("Input vs Target")
    plt.legend()

    for i, name in enumerate(["FC", "RNN", "LSTM"], 2):
        model, xt, yt = trained_models[name]
        with torch.no_grad():
            sample_in = torch.tensor(xt[idx:idx+1]).float()
            pred = model(sample_in).numpy().flatten()
        
        plt.subplot(4, 1, i)
        plt.plot(clean_target, 'g', alpha=0.4, label="Target")
        plt.plot(pred, 'r', label=f"{name} Prediction")
        plt.title(f"{name} Reconstruction")
        plt.legend()

    plt.tight_layout()
    plt.savefig("results/visual_reconstruction.png", dpi=150)
    print("Saved: results/visual_reconstruction.png")

    if config.get("plots", {}).get("show", False):
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
