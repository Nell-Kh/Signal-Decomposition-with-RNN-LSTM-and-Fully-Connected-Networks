# PRD_fc_network.md — Fully Connected Network
# Signal Decomposition Project

**Authors:** Nell Khoury, Yanal Serhan
**Version:** 1.00
**Last Updated:** 05.05.2026

---

## 1. Overview

### 1.1 Purpose
The Fully Connected (FC) Network serves as the **baseline model**
for the signal decomposition task. It receives a flat vector
containing the one-hot frequency selector and a window of noisy
composite samples, and predicts the corresponding clean sinusoid window.

### 1.2 Role in the Project
The FC network sets the **lower bound on performance**.
Because it has no concept of time or sequence order, it is expected
to perform well at low noise but degrade faster than RNN and LSTM
as noise increases. This degradation pattern is what we measure
in Part 4 of the assignment.

### 1.3 Why Fully Connected?
A fully connected network treats all 10 input samples as an
unordered flat vector — it has no built-in notion of "which sample
came before which". This is intentional for the baseline role:
it shows what can be achieved with zero temporal awareness,
and makes the advantage of RNN and LSTM measurable and explainable.

---

## 2. Input / Output Specification

### 2.1 Input
```
X = [c, x_Sigma]

where:
  c       = one-hot vector              shape: (4,)
  x_Sigma = noisy composite window      shape: (10,)
  X       = flat concatenation of both  shape: (14,)
```

The one-hot vector c selects which of the 4 sinusoids to extract:
```
[1,0,0,0] -> extract S1 (1 Hz)
[0,1,0,0] -> extract S2 (3 Hz)
[0,0,1,0] -> extract S3 (5 Hz)
[0,0,0,1] -> extract S4 (7 Hz)
```

### 2.2 Output
```
Y = predicted clean sinusoid window    shape: (10,)
```
No activation on the output layer — this is a regression task
predicting continuous wave values, not a classification task.

---

## 3. Architecture

### 3.1 Layer Design
```
Input  (14,)
  |
Linear(14 -> 64) + ReLU
  |
Linear(64 -> 64) + ReLU
  |
Linear(64 -> 10)
  |
Output (10,)
```

### 3.2 Design Decisions

**Hidden dimension = 64:**
Large enough to learn the mapping from noisy composite to clean
sinusoid, small enough to train quickly on CPU in under 2 minutes.

**Two hidden layers:**
One layer is insufficient to learn the non-linear decomposition.
Three or more layers would overfit given the dataset size of 10,000.
Two layers was validated empirically during development.

**ReLU activation:**
Standard choice for regression tasks. Avoids the vanishing gradient
problem that sigmoid/tanh suffer from in deeper networks.

**No output activation:**
The clean sinusoid values range from -A to +A (i.e. -1.0 to +1.0).
Applying sigmoid or tanh would clip values incorrectly.
A linear output layer allows the network to predict any real value.

---

## 4. Training Configuration

| Parameter     | Value  | Justification                              |
|---------------|--------|--------------------------------------------|
| Loss function | MSE    | Required by assignment                     |
| Optimizer     | Adam   | Adaptive learning rate, robust to noise    |
| Learning rate | 0.001  | Standard starting point for Adam           |
| Batch size    | 64     | Good balance of speed and gradient quality |
| Epochs        | 50     | Sufficient for convergence at this scale   |

---

## 5. Limitations

- **No temporal awareness:** treats all 10 samples as unordered
- **No memory:** cannot learn that sample t+1 follows sample t
- **Expected degradation:** MSE rises faster than RNN/LSTM as noise increases
- **No gating:** cannot selectively focus on relevant signal features

---

## 6. Acceptance Criteria

| Criterion                        | Target                    |
|----------------------------------|---------------------------|
| Training converges               | Loss decreasing over epochs|
| Test MSE at 2% noise             | Recorded as baseline       |
| Test MSE increases with noise    | Monotonically increasing   |
| FC MSE > RNN MSE at medium noise | Demonstrated in analysis   |
| Inference runs without error     | All 4 frequency selectors  |