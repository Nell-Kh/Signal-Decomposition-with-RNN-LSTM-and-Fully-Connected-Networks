# PRD_lstm_network.md — LSTM Network
# Signal Decomposition Project

**Authors:** Nell Khoury, Yanal Serhan
**Version:** 1.00
**Last Updated:** 05.05.2026

---

## 1. Overview

### 1.1 Purpose
The LSTM (Long Short-Term Memory) network extends the Vanilla RNN
with a **gating mechanism** that allows it to selectively remember
and forget information across timesteps. This makes it significantly
more robust at high noise levels where the vanilla RNN's simple
hidden state becomes overwhelmed.

### 1.2 Role in the Project
The LSTM is the **strongest model** in this comparison.
It is expected to outperform both FC and RNN at medium-to-high
noise levels. The key research question this model answers is:
"At what noise level does the gating advantage of LSTM
become clearly visible over vanilla RNN?"

### 1.3 Why LSTM?
LSTM was introduced specifically to solve the vanishing gradient
problem in vanilla RNNs. Its three gates (input, forget, output)
allow it to:
  - Selectively store new information (input gate)
  - Selectively discard old information (forget gate)
  - Selectively expose its memory (output gate)

For noisy sinusoidal decomposition, this means the LSTM can
learn to focus on the periodic pattern of the target frequency
while suppressing noise — something vanilla RNN cannot do reliably.

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

Identical input format to the Vanilla RNN — the only difference
is the internal processing of each timestep.

### 2.2 Output
```
Y = predicted clean sinusoid window    shape: (10,)
```

Output style: **many-to-one** (same as RNN, same justification)
The final hidden state after all 10 steps is mapped to R^10.

---

## 3. Architecture

### 3.1 Layer Design
```
Input sequence  (10, 5)
  |
LSTM(input_size=5, hidden_size=64, num_layers=1, batch_first=True)
  |
Final hidden state h_n              shape: (64,)
  |
Linear(64 -> 10)
  |
Output  (10,)
```

### 3.2 How the LSTM Gates Work
At each timestep j, the LSTM computes:

```
Forget gate:  f_j = sigmoid(W_f * [h_{j-1}, x_j] + b_f)
Input gate:   i_j = sigmoid(W_i * [h_{j-1}, x_j] + b_i)
Candidate:    g_j = tanh(W_g * [h_{j-1}, x_j] + b_g)
Cell state:   c_j = f_j * c_{j-1} + i_j * g_j
Output gate:  o_j = sigmoid(W_o * [h_{j-1}, x_j] + b_o)
Hidden state: h_j = o_j * tanh(c_j)

where:
  c_j = cell state  (long-term memory)
  h_j = hidden state (short-term memory, exposed to output)
  x_j = current input [c, Sigma_noisy(t_i+j)]
```

The **cell state c_j** is the key difference from vanilla RNN.
It acts as a conveyor belt carrying information across timesteps
with minimal transformation, preventing the vanishing gradient problem.

### 3.3 Design Decisions

**Hidden dimension = 64:**
Same as FC and RNN for fair comparison.

**1 LSTM layer:**
Sufficient for sequences of length 10. Additional layers
would add training time without measurable benefit here.

**Many-to-one output:**
Same justification as RNN — full context before prediction.

---

## 4. Training Configuration

| Parameter     | Value  | Justification                              |
|---------------|--------|--------------------------------------------|
| Loss function | MSE    | Required by assignment                     |
| Optimizer     | Adam   | Adaptive learning rate, robust to noise    |
| Learning rate | 0.001  | Same as FC and RNN for fair comparison     |
| Batch size    | 64     | Same as FC and RNN for fair comparison     |
| Epochs        | 50     | Same as FC and RNN for fair comparison     |

---

## 5. Expected Behavior vs RNN

| Noise Level | RNN Expected  | LSTM Expected | Why                              |
|-------------|---------------|---------------|----------------------------------|
| 1% - 5%     | Good          | Good          | Both handle this range well      |
| 8% - 10%    | Starts to fail| Still good    | Forget gate filters noise        |
| 15%+        | Poor          | Better        | Cell state preserves signal      |
| 20%+        | Very poor     | Degrades      | Even LSTM struggles at this level|

---

## 6. Advantages Over Vanilla RNN

| Property           | Vanilla RNN    | LSTM                        |
|--------------------|----------------|-----------------------------|
| Memory type        | h only         | h (short) + c (long)        |
| Forgetting         | Implicit only  | Explicit forget gate        |
| Gradient flow      | Vanishes       | Protected by cell state     |
| Noise robustness   | Medium         | High                        |
| Parameter count    | Lower          | ~4x more (worth the trade)  |

---

## 7. Acceptance Criteria

| Criterion                          | Target                      |
|------------------------------------|-----------------------------|
| Training converges                 | Loss decreasing over epochs  |
| LSTM MSE < RNN MSE at 10%+ noise  | Demonstrated in analysis     |
| LSTM MSE < FC MSE at all levels   | Demonstrated in analysis     |
| Inference runs without error       | All 4 frequency selectors    |