import torch.nn as nn


class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim=14, output_dim=10, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class VanillaRNN(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        # 5 inputs: [noisy_composite, c1, c2, c3, c4]
        self.rnn = nn.RNN(5, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # Many-to-Many: map each hidden state to 1 clean sample
        self.fc = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        # x: (batch, 10, 5)
        out, _ = self.rnn(x)
        # out: (batch, 10, hidden_dim)
        out = self.layer_norm(out)
        # Apply FC to every time step: (batch, 10, hidden_dim) -> (batch, 10, 1)
        preds = self.fc(out)
        return preds.squeeze(-1) # -> (batch, 10)


class LSTMNetwork(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.lstm = nn.LSTM(5, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        preds = self.fc(out)
        return preds.squeeze(-1) # -> (batch, 10)
