import numpy as np
import torch

from ..services.dataset import build_dataset_with_fresh_noise
from ..services.generator import generate_base_signals
from ..services.models import FullyConnectedNetwork, LSTMNetwork, VanillaRNN
from ..services.trainer import evaluate_model, train_model
from ..shared.constants import FREQUENCIES


class SignalDecompSDK:
    def __init__(self, config, phases=None):
        self.config = config
        self.t = np.arange(0, config["signal"]["duration"], 1 / config["signal"]["fs"])

        if phases is not None:
            # Use provided phases (e.g. from meta.json) for evaluation consistency
            self._phases = np.array(phases)
            # Re-generate base signals with these fixed phases
            self._clean, _ = generate_base_signals(self.t, FREQUENCIES, config["signal"]["amplitude"], fixed_phases=self._phases)
        else:
            self._clean, self._phases = generate_base_signals(self.t, FREQUENCIES, config["signal"]["amplitude"])

    def train(self, model_name, x_train, y_train):
        model = self._build_model(model_name)
        return train_model(
            model,
            x_train,
            y_train,
            lr=self.config["training"]["learning_rate"],
            batch_size=self.config["training"]["batch_size"],
            epochs=self.config["training"]["epochs"],
        )

    def test(self, model, x_test, y_test):
        return evaluate_model(model, x_test, y_test)

    def evaluate_at_noise(self, model, model_name, noise_pct):
        cfg = self.config
        x_fc, x_seq, y = build_dataset_with_fresh_noise(
            clean_signals=self._clean,
            t=self.t,
            frequencies=FREQUENCIES,
            amplitude=cfg["signal"]["amplitude"],
            phases=self._phases,
            noise_amplitude_pct=noise_pct,
            noise_phase_range=cfg["noise"]["phase_range"],
            num_samples=cfg["analysis"]["test_samples_per_level"],
            window_size=cfg["dataset"]["window_size"],
        )
        x_input = x_fc if model_name == "FC" else x_seq
        return evaluate_model(model, x_input, y)

    def save_model(self, model, path):
        """Save PyTorch model weights to disk."""
        torch.save(model.state_dict(), path)

    def load_model(self, model_name, path):
        """Load PyTorch model weights from disk."""
        model = self._build_model(model_name)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        return model

    def _build_model(self, model_name):
        h = self.config["training"]["hidden_dim"]
        if model_name == "FC":
            w = self.config["dataset"]["window_size"]
            input_dim = 4 + w   # one-hot(4) + noisy window
            output_dim = w
            return FullyConnectedNetwork(input_dim=input_dim, output_dim=output_dim, hidden_dim=h)
        if model_name == "RNN":
            return VanillaRNN(h)
        if model_name == "LSTM":
            return LSTMNetwork(h)
        raise ValueError(f"Unknown model_name: {model_name}")
