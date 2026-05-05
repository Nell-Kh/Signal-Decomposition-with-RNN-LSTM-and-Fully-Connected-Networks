"""SDK — single entry point for all project logic."""
import numpy as np

from ..services.generator import generate_base_signals, apply_noise
from ..services.dataset import build_dataset
from ..services.models import FullyConnectedNetwork, VanillaRNN, LSTMNetwork
from ..services.trainer import train_model, evaluate_model
from ..shared.constants import FREQUENCIES


class SignalDecompSDK:
    """
    Main interface for the Signal Decomposition system.

    All logic — signal generation, dataset construction, training,
    and evaluation — is accessed through this single class.
    No external code should call services directly.

    Usage:
        config = load_config()
        sdk = SignalDecompSDK(config)
        model, history = sdk.run_training_pipeline("FC")
        mse = sdk.evaluate(model, "FC", noise_pct=0.05)
    """

    def __init__(self, config):
        """
        Initialize SDK with config loaded from setup.json.

        Args:
            config: dict loaded from config/setup.json
        """
        self.config = config
        self.t = np.arange(
            0,
            config["signal"]["duration"],
            1 / config["signal"]["fs"]
        )
        # Generate phases ONCE — fixed for all three models
        self._clean, self._phases = generate_base_signals(
            self.t,
            FREQUENCIES,
            config["signal"]["amplitude"]
        )

    def run_training_pipeline(self, model_name):
        """
        Full pipeline: generate data → build dataset → train model.

        Args:
            model_name: "FC", "RNN", or "LSTM"

        Returns:
            trained_model: the trained nn.Module
            history:       list of average loss per epoch
        """
        cfg = self.config
        amplitude = cfg["signal"]["amplitude"]
        noise_amp = cfg["noise"]["amplitude_pct"]
        noise_phi = cfg["noise"]["phase_range"]

        # Apply noise — composite is the sum of all 4 noisy sinusoids
        _, composite = apply_noise(
            self._clean, self._phases, FREQUENCIES,
            self.t, amplitude, noise_amp, noise_phi
        )

        # Build dataset windows
        x_train, y_train = build_dataset(
            self._clean, composite,
            cfg["dataset"]["num_samples"],
            cfg["dataset"]["window_size"],
            model_name
        )

        # Select architecture
        model = self._build_model(model_name)

        # Train
        trained_model, history = train_model(
            model, x_train, y_train,
            lr=cfg["training"]["learning_rate"],
            batch_size=cfg["training"]["batch_size"],
            epochs=cfg["training"]["epochs"]
        )

        return trained_model, history

    def evaluate(self, model, model_name, noise_pct):
        """
        Evaluate a trained model on fresh data at a given noise level.
        Used for the noise sensitivity analysis in Part 4.

        Args:
            model:      trained nn.Module
            model_name: "FC", "RNN", or "LSTM"
            noise_pct:  amplitude noise percentage to test at

        Returns:
            mse: average MSE on 1000 fresh test samples
        """
        cfg = self.config
        amplitude = cfg["signal"]["amplitude"]
        noise_phi = cfg["noise"]["phase_range"]
        test_samples = cfg["analysis"]["test_samples_per_level"]

        _, composite = apply_noise(
            self._clean, self._phases, FREQUENCIES,
            self.t, amplitude, noise_pct, noise_phi
        )

        x_test, y_test = build_dataset(
            self._clean, composite,
            test_samples,
            cfg["dataset"]["window_size"],
            model_name
        )

        return evaluate_model(model, x_test, y_test)

    def _build_model(self, model_name):
        """
        Instantiate the correct model architecture.

        Args:
            model_name: "FC", "RNN", or "LSTM"

        Returns:
            nn.Module instance
        """
        if model_name == "FC":
            return FullyConnectedNetwork()
        elif model_name == "RNN":
            return VanillaRNN()
        elif model_name == "LSTM":
            return LSTMNetwork()
        else:
            raise ValueError(f"Unknown model: {model_name}. Choose FC, RNN, or LSTM.")