# PRD_rnn_network.md — Vanilla RNN Network
# Signal Decomposition Project

**Authors:** Nell Khoury, Yanal Serhan
**Version:** 1.00
**Last Updated:** 07.05.2026

---

## 1. Overview

### 1.1 Purpose
The Vanilla RNN (Recurrent Neural Network) serves as the first step into temporal signal processing for this project. Unlike the Fully Connected baseline, the RNN processes the input window sequentially, allowing it to maintain a "hidden state" that captures temporal dependencies.

### 1.2 Role in the Project
The RNN is the **intermediate model**. It is expected to outperform the FC network on clean signals by utilizing sequential patterns, but it is expected to be more fragile than the LSTM under high noise due to the "vanishing gradient" problem inherent in simple recurrent loops.

### 1.3 Why Vanilla RNN?
A Vanilla RNN represents the simplest form of recurrent architecture. By comparing it to the FC network, we prove the value of sequential processing. By comparing it to the LSTM, we prove the necessity of "gating" mechanisms for handling noise and long-term dependencies.

---

## 2. Input / Output Specification

### 2.1 Input
```
X = sequence of 10 timesteps

At each timestep j:
  input_j = [c, Sigma_noisy(t_i+j)]    shape: (5,)

where:
  c                   = one-hot vector  shape: (4,)  repeated at every step
  Sigma_noisy(t_i+j)  = one noisy composite sample  shape: (1,)

Full input shape: (10, 5)
  = 10 timesteps x 5 features per step
```

### 2.2 Output
```
Y = predicted clean sinusoid window    shape: (10,)
```

**Architecture Style:** Many-to-one. The network processes the entire 10-step sequence, and the final hidden state is used to project the full 10-sample output window.

---

## 3. Architecture

### 3.1 Layer Design
```
Input sequence  (10, 5)
  |
RNN(input_size=5, hidden_size=64, num_layers=1, batch_first=True)
  |
Final hidden state h_n              shape: (64,)
  |
Linear(64 -> 10)
  |
Output  (10,)
```

### 3.2 Design Decisions

**Hidden dimension = 64:**
Standardized across all three models (FC, RNN, LSTM) to ensure the comparison is based on architecture type rather than parameter count.

**Orthogonal Initialization:**
Simple RNNs are notoriously difficult to train. We use orthogonal weight initialization for the recurrent weights to keep the gradient scale close to 1.0 during the early stages of training.

**Many-to-one Mapping:**
We collect the final hidden state after the 10th sample is processed. This ensures the model has seen the entire noisy window before attempting to reconstruct the clean version.

---

## 4. Training Configuration

| Parameter     | Value  | Justification                              |
|---------------|--------|--------------------------------------------|
| Loss function | MSE    | Standard for signal reconstruction         |
| Optimizer     | Adam   | Robustness against noisy gradients         |
| Learning rate | 0.001  | Consistent with FC and LSTM                |
| Batch size    | 64     | Efficient GPU/CPU utilization              |
| Epochs        | 50     | Convergence checked during validation      |

---

## 5. Limitations

- **Vanishing Gradients:** Simple RNNs struggle to pass information across many steps without it being "washed out" or "exploding".
- **Noise Sensitivity:** Without gates (like LSTM), the RNN hidden state is directly perturbed by every noisy input sample, making it less robust than gated architectures.
- **Limited Context:** While it has "memory", it is effectively a "short-term" memory compared to LSTM.

---

## 6. Acceptance Criteria

| Criterion                        | Target                    |
|----------------------------------|---------------------------|
| Training converges               | Loss decreasing over epochs|
| Test MSE at 2% noise             | Better than FC (ideally)   |
| Performance vs Noise             | Degrades faster than LSTM  |
| Inference runs without error     | Validated on all frequencies|
