import torch
import torch.nn as nn

class FullyConnectedNetwork(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )
    def forward(self, x):
        return self.network(x)

class VanillaRNN(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU()
        )
        self.rnn = nn.RNN(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        x = self.input_proj(x)
        out, _ = self.rnn(x)
        return self.fc(out).squeeze(-1)

class LSTMNetwork(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        x = self.input_proj(x)
        out, _ = self.lstm(x)
        return self.fc(out).squeeze(-1)