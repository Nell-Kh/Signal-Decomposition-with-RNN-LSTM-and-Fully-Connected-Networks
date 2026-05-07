import unittest.mock as mock

import numpy as np
import torch

from signal_decomp.services.visualizer import (
    plot_noise_sensitivity,
    plot_training_loss,
    plot_visual_reconstruction,
)


def test_plot_training_loss():
    histories = {"M1": [0.1, 0.05], "M2": [0.2, 0.1]}
    with mock.patch("matplotlib.pyplot.savefig") as mock_save:
        plot_training_loss(histories, "dummy.png")
        assert mock_save.called


def test_plot_noise_sensitivity():
    levels = [0.01, 0.05]
    results = {"M1": [0.01, 0.02]}
    with mock.patch("matplotlib.pyplot.savefig") as mock_save:
        plot_noise_sensitivity(levels, results, "dummy.png")
        assert mock_save.called


def test_plot_visual_reconstruction():
    class MockModel:
        def __call__(self, x):
            return torch.zeros(x.shape[0], 10)

    trained_models = {
        "FC": (MockModel(), np.zeros((1, 14)), np.zeros((1, 10))),
        "RNN": (MockModel(), np.zeros((1, 10, 5)), np.zeros((1, 10))),
        "LSTM": (MockModel(), np.zeros((1, 10, 5)), np.zeros((1, 10))),
    }
    datasets = {"RNN": [None, np.zeros((1, 10, 5)), None, np.zeros((1, 10))]}

    with mock.patch("matplotlib.pyplot.savefig") as mock_save:
        plot_visual_reconstruction(trained_models, datasets, "dummy.png")
        assert mock_save.called
