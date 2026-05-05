# PRD_rnn_network.md — Vanilla RNN Network
# Signal Decomposition Project

**Authors:** Nell Khoury, Yanal Serhan
**Version:** 1.00
**Last Updated:** 05.05.2026

---

## 1. Overview

### 1.1 Purpose
The Vanilla RNN processes the noisy composite signal as a
**sequence of timesteps**, allowing it to capture temporal
dependencies that the FC baseline completely ignores.
Each timestep carries both the one-hot frequency selector
and one noisy composite sample.

### 1.2 Role in the Project
The RNN is the **middle ground** between FC and LSTM.
It is expected to outperform FC at medium noise levels
because it understands that sample t+1 follows sample t.
However, it is expected to degrade before LSTM at high noise
because its memory is limited — it cannot selectively forget
irrelevant information the way LSTM can.

### 1.3 Why Vanilla RNN?
Vanilla RNN is the simplest sequential model. Including it
between FC and LSTM lets us measure the exact benefit of:
  1. Adding temporal awareness at all (FC -> RNN improvement)
  2. Adding gated memory on top (RNN -> LSTM improvement)

Without the vanilla RNN step, we cannot separate these two effects.

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

The one-hot vector c is repeated at every timestep so the network
always knows which frequency to extract, even mid-sequence.

### 2.2 Output
```
Y = predicted clean sinusoid window    shape: (10,)
```

Output style: **many-to-one**
The final hidden state (after processing all 10 steps) is mapped
to R^10 via a linear layer.

Justification for many-to-one:
The network needs to see the full 10-step context before making
a prediction. Outputting after every step (many-to-many) would
force predictions before enough context is available.

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

### 3.2 How the RNN Hidden State Works
```
At each timestep j:
  h_j = tanh(W_h * h_{j-1} + W_x * x_j + b)

where:
  h_{j-1} = previous hidden state (memory)
  x_j     = current input [c, Sigma_noisy(t_i+j)]
  W_h, W_x, b = learned parameters
```

The final h_10 encodes information from all 10 timesteps
and is passed to the linear output layer.

### 3.3 Design Decisions

**Hidden dimension = 64:**
Same as FC for fair comparison — the only difference
between models should be the architecture, not the capacity.

**1 RNN layer:**
Sufficient for a window of 10 steps. Stacking layers adds
complexity without clear benefit at this sequence length.

**Many-to-one output:**
Justified above — full context needed before prediction.

**tanh activation (built into nn.RNN):**
Standard for RNN hidden states. Keeps values bounded
which stabilizes training.

---

## 4. Training Configuration

| Parameter     | Value  | Justification                              |
|---------------|--------|--------------------------------------------|
| Loss function | MSE    | Required by assignment                     |
| Optimizer     | Adam   | Adaptive learning rate, robust to noise    |
| Learning rate | 0.001  | Same as FC for fair comparison             |
| Batch size    | 64     | Same as FC for fair comparison             |
| Epochs        | 50     | Same as FC for fair comparison             |

All training parameters are identical across all three models
so that architecture is the only variable being compared.

---

## 5. Expected Behavior vs FC

| Noise Level | FC Expected | RNN Expected | Why                        |
|-------------|-------------|--------------|----------------------------|
| 1% - 2%     | Good        | Good         | Both handle low noise well |
| 5% - 8%     | Degrades    | Better       | RNN uses temporal order    |
| 10%+        | Poor        | Degrades     | RNN memory is too short    |
| 15%+        | Very poor   | Poor         | Only LSTM survives here    |

---

## 6. Limitations

- **Short memory:** hidden state from step 1 is diluted by step 10
- **No gating:** cannot selectively forget irrelevant noise
- **Vanishing gradient:** difficult to learn long-range dependencies
- **No cell state:** unlike LSTM, has only one memory vector h

---

## 7. Acceptance Criteria

| Criterion                         | Target                     |
|-----------------------------------|----------------------------|
| Training converges                | Loss decreasing over epochs |
| RNN MSE < FC MSE at 5-8% noise   | Demonstrated in analysis    |
| RNN MSE > LSTM MSE at 10%+ noise | Demonstrated in analysis    |
| Inference runs without error      | All 4 frequency selectors   |