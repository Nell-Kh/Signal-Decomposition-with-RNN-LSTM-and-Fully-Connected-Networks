import torch

from signal_decomp.services.models import FullyConnectedNetwork, LSTMNetwork, VanillaRNN


def test_fc_network_forward():
    input_dim = 14
    output_dim = 10
    model = FullyConnectedNetwork(input_dim=input_dim, output_dim=output_dim, hidden_dim=32)
    model.eval()
    x = torch.randn(5, input_dim)
    y = model(x)
    assert y.shape == (5, output_dim)


def test_rnn_network_forward():
    hidden_dim = 32
    model = VanillaRNN(hidden_dim=hidden_dim)
    model.eval()
    x = torch.randn(5, 10, 5)  # batch, seq_len, input_size
    y = model(x)
    assert y.shape == (5, 10)


def test_lstm_network_forward():
    hidden_dim = 32
    model = LSTMNetwork(hidden_dim=hidden_dim)
    model.eval()
    x = torch.randn(5, 10, 5)
    y = model(x)
    assert y.shape == (5, 10)
