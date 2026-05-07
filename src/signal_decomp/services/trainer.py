import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_model(model, x_np, y_np, lr=0.001, batch_size=64, epochs=150):
    """
    Train any of the three architectures using MSE loss and Adam optimizer.
    Outputs a scrolling, academic-style log (one line per epoch).
    """
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Scheduler to help with stability and 'straight line' recovery
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)

    print(f"\n🚀 Training started: {model.__class__.__name__}")
    print("-" * 50)

    loss_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()

            # Clip gradients for RNN/LSTM stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        
        # Simple academic log format: one line per epoch
        print(f"Epoch [{epoch+1:3d}/{epochs}] | Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")

    print(f"✅ Training Complete. Final Loss: {avg_loss:.6f}\n")
    return model, loss_history


def evaluate_model(model, x_np, y_np, batch_size=64):
    """Standard evaluation for MSE metrics."""
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size)

    criterion = nn.MSELoss()
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            total_loss += loss.item()

    return total_loss / len(loader)
