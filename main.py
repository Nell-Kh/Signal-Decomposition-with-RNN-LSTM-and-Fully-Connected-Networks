"""Entry point — run the full training pipeline for all 3 models."""
from src.signal_decomp.shared.config import load_config
from src.signal_decomp.sdk.sdk import SignalDecompSDK


def main():
    """Train FC, RNN, and LSTM and print final MSE for each."""
    config = load_config()
    sdk = SignalDecompSDK(config)

    for model_name in ["FC", "RNN", "LSTM"]:
        print(f"\n{'='*40}")
        print(f"Training {model_name} model...")
        print(f"{'='*40}")

        model, history = sdk.run_training_pipeline(model_name)

        print(f"\n{model_name} final training loss: {history[-1]:.6f}")


if __name__ == "__main__":
    main()