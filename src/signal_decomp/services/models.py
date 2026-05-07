import torch.nn as nn


class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim=14, output_dim=10, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class VanillaRNN(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.rnn = nn.RNN(5, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        """Standard orthogonal initialization to prevent dead gradients."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out).squeeze(-1)

class LSTMNetwork(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.lstm = nn.LSTM(5, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out).squeeze(-1)
