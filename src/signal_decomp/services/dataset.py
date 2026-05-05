"""Dataset construction: sliding windows + one-hot selector."""
import numpy as np


def build_dataset(clean_signals, composite, num_samples, window_size, model_type):
    """
    Build the training dataset by extracting random windows.

    Each sample contains:
      - c:       one-hot vector selecting which sinusoid to extract
      - x_input: window from the noisy composite (the network input)
      - y:       window from the corresponding clean sinusoid (the target)

    Input shape depends on model type:
      FC:       flat vector [c, x_window]  shape: (14,)  = 4 + 10
      RNN/LSTM: sequence per timestep      shape: (10, 5) = 10 steps x (4 one-hot + 1 sample)

    The one-hot vector c is included at every timestep for RNN/LSTM
    so the network always knows which frequency to extract.

    Args:
        clean_signals: shape (4, N) — the 4 clean sinusoids
        composite:     shape (N,)   — the noisy composite signal
        num_samples:   number of dataset samples to generate
        window_size:   number of consecutive samples per window (W=10)
        model_type:    "FC", "RNN", or "LSTM"

    Returns:
        X: input array
        Y: target array, shape (num_samples, window_size)
    """
    x_data = []
    y_data = []
    one_hots = np.eye(4)
    max_start = len(composite) - window_size

    for _ in range(num_samples):
        # Pick which frequency the network should extract
        target_idx = np.random.randint(0, 4)
        c = one_hots[target_idx]

        # Pick a random starting point in the signal
        start_i = np.random.randint(0, max_start + 1)

        # Extract windows
        x_window = composite[start_i: start_i + window_size]
        y_target = clean_signals[target_idx, start_i: start_i + window_size]

        # Format input based on model type
        if model_type == "FC":
            # Flat vector: [4 one-hot values + 10 signal samples] = 14
            x_input = np.concatenate([c, x_window])
        else:
            # Sequence: 10 timesteps, each with [4 one-hot + 1 signal sample] = 5
            c_repeated = np.tile(c, (window_size, 1))       # (10, 4)
            x_reshaped = x_window.reshape(window_size, 1)   # (10, 1)
            x_input = np.hstack([c_repeated, x_reshaped])   # (10, 5)

        x_data.append(x_input)
        y_data.append(y_target)

    return np.array(x_data), np.array(y_data)