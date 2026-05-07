"""Training loop — works for FC, RNN, and LSTM."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_model(model, x_np, y_np, lr, batch_size, epochs):
    """
    Train any of the three architectures using MSE loss and Adam optimizer.

    Steps:
        1. Convert numpy arrays to PyTorch tensors
        2. Wrap in DataLoader for batching and shuffling
        3. Run training loop — forward, loss, backward, update
        4. Log loss every 10 epochs

    Args:
        model:      nn.Module — FC, RNN, or LSTM
        x_np:       numpy input array
        y_np:       numpy target array
        lr:         learning rate (from config)
        batch_size: batch size (from config)
        epochs:     number of training epochs (from config)

    Returns:
        model:        trained model
        loss_history: list of average loss per epoch
    """
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # MSE required by assignment; Adam chosen for adaptive learning rate
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            # Clip gradients to prevent explosion (crucial for RNN stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    return model, loss_history


def evaluate_model(model, x_np, y_np, batch_size=64):
    """
    Evaluate a trained model on test data and return average MSE.

    Args:
        model:      trained nn.Module
        x_np:       numpy input array (test set)
        y_np:       numpy target array (test set)
        batch_size: batch size for evaluation

    Returns:
        avg_mse: average MSE across all test batches
    """
    model.eval()
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            total_loss += loss.item()

    return total_loss / len(loader)
