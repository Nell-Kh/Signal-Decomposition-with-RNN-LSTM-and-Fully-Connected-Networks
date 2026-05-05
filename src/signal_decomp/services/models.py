"""Neural network architectures: FC, RNN, LSTM."""
import torch
import torch.nn as nn


class FullyConnectedNetwork(nn.Module):
    """
    Baseline model — treats all 10 samples as a flat unordered vector.
    No temporal awareness. Sets the lower bound on performance.

    Input:  flat vector [c, x_Sigma]  shape: (14,)
    Output: predicted clean window    shape: (10,)
    """

    def __init__(self, input_dim=14, hidden_dim=64, output_dim=10):
        """Initialize layers: 14 -> 64 -> 64 -> 10."""
        super(FullyConnectedNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
            # No activation on output — regression task (continuous values)
        )

    def forward(self, x):
        """Forward pass through all layers."""
        return self.network(x)


class VanillaRNN(nn.Module):
    """
    Sequential model using standard RNN cells.
    Captures short-range temporal dependencies FC ignores.

    Input:  sequence [c, Sigma(t)] at each step  shape: (10, 5)
    Output: predicted clean window               shape: (10,)

    Many-to-one: final hidden state mapped to R^10.
    Justified because the full 10-step context is needed
    before predicting the entire output window.
    """

    def __init__(self, input_dim=5, hidden_dim=64, output_dim=10):
        """Initialize RNN layer and output projection."""
        super(VanillaRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass — use only the final hidden state."""
        _, h_n = self.rnn(x)
        return self.fc(h_n.squeeze(0))


class LSTMNetwork(nn.Module):
    """
    Sequential model using LSTM for better long-range memory.
    Gating mechanism selectively retains relevant signal patterns.

    Input:  sequence [c, Sigma(t)] at each step  shape: (10, 5)
    Output: predicted clean window               shape: (10,)

    Many-to-one: final hidden state mapped to R^10.
    Expected to outperform RNN at high noise levels where
    short-term memory alone is insufficient.
    """

    def __init__(self, input_dim=5, hidden_dim=64, output_dim=10):
        """Initialize LSTM layer and output projection."""
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass — use only the final hidden state."""
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n.squeeze(0))