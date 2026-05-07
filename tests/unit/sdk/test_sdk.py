import numpy as np
import pytest

from signal_decomp.sdk.sdk import SignalDecompSDK


@pytest.fixture
def sample_config():
    return {
        "signal": {"duration": 1, "fs": 100, "amplitude": 1.0},
        "noise": {"phase_range": 0.1},
        "dataset": {"num_samples": 10, "window_size": 10},
        "training": {"learning_rate": 0.001, "batch_size": 2, "epochs": 1, "hidden_dim": 32},
        "analysis": {"test_samples_per_level": 5},
    }


def test_sdk_init(sample_config):
    sdk = SignalDecompSDK(sample_config)
    assert sdk.t.shape == (100,)
    assert sdk._clean.shape == (4, 100)


def test_sdk_init_with_phases(sample_config):
    phases = [0.1, 0.2, 0.3, 0.4]
    sdk = SignalDecompSDK(sample_config, phases=phases)
    assert np.array_equal(sdk._phases, phases)


def test_sdk_train_rnn(sample_config):
    sdk = SignalDecompSDK(sample_config)
    x = np.random.randn(10, 10, 5)
    y = np.random.randn(10, 10)
    model, history = sdk.train("RNN", x, y)
    assert len(history) == 1


def test_sdk_train_lstm(sample_config):
    sdk = SignalDecompSDK(sample_config)
    x = np.random.randn(10, 10, 5)
    y = np.random.randn(10, 10)
    model, history = sdk.train("LSTM", x, y)
    assert len(history) == 1


def test_sdk_evaluate_at_noise_rnn(sample_config):
    sdk = SignalDecompSDK(sample_config)
    model = sdk._build_model("RNN")
    mse = sdk.evaluate_at_noise(model, "RNN", 0.05)
    assert isinstance(mse, float)


def test_sdk_save_load_model(sample_config, tmp_path):
    sdk = SignalDecompSDK(sample_config)
    model = sdk._build_model("FC")
    path = tmp_path / "model.pth"
    sdk.save_model(model, path)

    loaded_model = sdk.load_model("FC", path)
    assert isinstance(loaded_model, type(model))


def test_sdk_invalid_model(sample_config):
    sdk = SignalDecompSDK(sample_config)
    with pytest.raises(ValueError, match="Unknown model_name"):
        sdk._build_model("INVALID")
