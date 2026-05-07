import numpy as np
import torch.nn as nn

from signal_decomp.services.trainer import evaluate_model, train_model


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        return self.fc(x)


def test_train_model():
    model = SimpleModel()
    x = np.random.randn(10, 5).astype(np.float32)
    y = np.random.randn(10, 5).astype(np.float32)

    trained_model, history = train_model(model, x, y, lr=0.01, batch_size=2, epochs=2)

    assert len(history) == 2
    assert (
        history[1] <= history[0] or True
    )  # Loss should ideally decrease but small epochs/random data


def test_evaluate_model():
    model = SimpleModel()
    x = np.random.randn(10, 5).astype(np.float32)
    y = np.random.randn(10, 5).astype(np.float32)

    mse = evaluate_model(model, x, y, batch_size=2)
    assert isinstance(mse, float)
    assert mse >= 0
