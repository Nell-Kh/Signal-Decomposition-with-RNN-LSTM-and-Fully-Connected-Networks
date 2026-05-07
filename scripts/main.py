"""Entry point — train all 3 models using saved dataset."""

import json
from pathlib import Path
import numpy as np
from signal_decomp.sdk.sdk import SignalDecompSDK
from signal_decomp.shared.config import load_config
from signal_decomp.services.visualizer_bench import (
    plot_noise_sensitivity,
    plot_training_loss,
    plot_test_performance_comparison
)
from signal_decomp.services.visualizer_anatomy import plot_window_micro_diagnostic

def main():
    config = load_config()
    Path("results").mkdir(exist_ok=True)

    print("Loading dataset...")
    x_fc, x_seq, y = np.load("data/X_FC.npy"), np.load("data/X_seq.npy"), np.load("data/Y.npy")
    split = int(len(y) * config["dataset"]["train_split"])
    datasets = {
        "FC": (x_fc[:split], x_fc[split:], y[:split], y[split:]),
        "RNN": (x_seq[:split], x_seq[split:], y[:split], y[split:]),
        "LSTM": (x_seq[:split], x_seq[split:], y[:split], y[split:]),
    }

    with open("data/meta.json") as f:
        meta = json.load(f)

    sdk = SignalDecompSDK(config, phases=meta["phases"])
    trained_models, all_histories = {}, {}

    for model_name in ["FC", "RNN", "LSTM"]:
        print(f"\nTraining {model_name}...")
        x_train, x_test, y_train, y_test = datasets[model_name]
        model, history = sdk.train(model_name, x_train, y_train)
        trained_models[model_name] = (model, x_test, y_test)
        all_histories[model_name] = history
        weight_path = Path("weights") / f"{model_name}.pth"
        weight_path.parent.mkdir(exist_ok=True)
        sdk.save_model(model, weight_path)

    print("\n--- Test MSE (held-out 20%) ---")
    test_mse_results = {}
    for name, (model, xt, yt) in trained_models.items():
        mse = sdk.test(model, xt, yt)
        test_mse_results[name] = mse
        print(f"{name}: {mse:.6f}")

    print("\n--- Noise Sensitivity Analysis ---")
    noise_levels = config["analysis"]["noise_levels"]
    noise_results = {name: [] for name in ["FC", "RNN", "LSTM"]}
    for noise in noise_levels:
        print(f"Testing noise {int(noise * 100)}%...")
        for name, (model, _, _) in trained_models.items():
            mse = sdk.evaluate_at_noise(model, name, noise)
            noise_results[name].append(mse)

    # Plotting
    print("\nGenerating final report plots...")
    plot_training_loss(all_histories, "results/training_loss.png")
    plot_noise_sensitivity(noise_levels, noise_results, "results/noise_sensitivity.png")
    plot_test_performance_comparison(test_mse_results, "results/test_performance.png")
    plot_window_micro_diagnostic(trained_models, datasets, "results/micro_diagnostic.png")
    print("Results saved to results/ directory.")

if __name__ == "__main__":
    main()
