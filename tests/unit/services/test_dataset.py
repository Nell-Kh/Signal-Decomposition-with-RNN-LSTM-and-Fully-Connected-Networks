import numpy as np

from signal_decomp.services.dataset import build_dataset, build_dataset_with_fresh_noise


def test_build_dataset_fc():
    clean = np.random.randn(4, 100)
    composite = np.random.randn(100)
    num_samples = 10
    window_size = 10

    x, y = build_dataset(clean, composite, num_samples, window_size, "FC")

    assert x.shape == (num_samples, 14)  # 4 one-hot + 10 window
    assert y.shape == (num_samples, 10)


def test_build_dataset_rnn():
    clean = np.random.randn(4, 100)
    composite = np.random.randn(100)
    num_samples = 10
    window_size = 10

    x, y = build_dataset(clean, composite, num_samples, window_size, "RNN")

    assert x.shape == (num_samples, 10, 5)  # 10 steps x (4 one-hot + 1 sample)
    assert y.shape == (num_samples, 10)


def test_build_dataset_with_fresh_noise():
    t = np.linspace(0, 1, 100)
    clean = np.random.randn(4, 100)
    freqs = [1, 2, 3, 4]
    amp = 1.0
    phases = np.zeros(4)
    num_samples = 5
    window_size = 10

    x_fc, x_seq, y = build_dataset_with_fresh_noise(
        clean, t, freqs, amp, phases, 0.02, 0.1, num_samples, window_size
    )

    assert x_fc.shape == (num_samples, 14)
    assert x_seq.shape == (num_samples, 10, 5)
    assert y.shape == (num_samples, 10)
