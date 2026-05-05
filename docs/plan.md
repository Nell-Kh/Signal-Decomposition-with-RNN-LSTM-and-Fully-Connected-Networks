# PLAN.md — Architecture & Planning Document
# Signal Decomposition with RNN, LSTM, and Fully Connected Networks

**Authors:** Nell Khoury, Yanal Serhan
**Version:** 1.00
**Last Updated:** 05.05.2026

---

## 1. System Architecture Overview (C4 Model)

### 1.1 Context Level — What the system does
```
+---------------------------------------------------------------+
|                        USER / RESEARCHER                      |
|           runs main.py to train and compare models            |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
|                    Signal Decomposition System                |
|                                                               |
|  Given: noisy composite of 4 sinusoids + one-hot selector    |
|  Output: clean reconstruction of the selected sinusoid        |
+---------------------------------------------------------------+
                                |
              +-----------------+-----------------+
              |                 |                 |
              v                 v                 v
        FC Network         RNN Network       LSTM Network
        (baseline)         (sequential)      (gated memory)
```

### 1.2 Container Level — How the system is organized
```
+------------------------------------------------------------------+
|                        main.py (entry point)                     |
+------------------------------------------------------------------+
                               |
                               v
+------------------------------------------------------------------+
|                         SDK Layer                                |
|                     sdk/sdk.py                                   |
|   Single entry point for ALL logic — no external code            |
|   calls services directly                                        |
+--------+-----------+-------------------+-------------------------+
         |           |                   |
         v           v                   v
+--------+--+ +------+------+ +----------+--------+
| generator | |   dataset   | |  trainer + models |
| .py       | |   .py       | |  .py       .py    |
+-----------+ +-------------+ +-------------------+
         |           |                   |
         v           v                   v
+------------------------------------------------------------------+
|                       shared/                                    |
|            config.py    constants.py                             |
+------------------------------------------------------------------+
                               |
                               v
+------------------------------------------------------------------+
|                       config/setup.json                          |
|   All hyperparameters live here — nothing hardcoded in code      |
+------------------------------------------------------------------+
```

### 1.3 Component Level — Data flow through the pipeline
```
setup.json
    |
    | load_config()
    v
SignalDecompSDK.__init__()
    |
    | generate_base_signals(t, frequencies, amplitude)
    v
clean_signals (4, 10000)   phases (4,)
    |
    | apply_noise(clean, phases, frequencies, t, A, noise_amp, noise_phi)
    v
noisy_signals (4, 10000)   composite (10000,)
    |
    | build_dataset(clean, composite, num_samples, window_size, model_type)
    v
X_train  Y_train
    |
    | train_model(model, X, Y, lr, batch_size, epochs)
    v
trained_model   loss_history
    |
    | evaluate(model, model_name, noise_pct)  [for each noise level]
    v
MSE results -> results/ folder -> analysis notebook
```

---

## 2. Signal Generation Design

### 2.1 The Four Sinusoids
```
Time vector:  t = [0, 0.001, 0.002, ..., 9.999]   length: 10,000

For k = 1, 2, 3, 4:
  Clean:  Sk(t)       = A * sin(2*pi*fk*t + phi_k)
  Noisy:  S_noisy_k(t)= (A + epsilon_A) * sin(2*pi*fk*t + phi_k + epsilon_phi)

  where:
    A       = 1.0   (fixed, same for all)
    phi_k   = random in [0, 2*pi], drawn ONCE, fixed for entire project
    epsilon_A   ~ Uniform(-0.02*A, +0.02*A)  per sample
    epsilon_phi ~ Uniform(-0.1,    +0.1)     per sample

Composite:  Sigma(t) = S_noisy_1(t) + S_noisy_2(t) + S_noisy_3(t) + S_noisy_4(t)
```

### 2.2 What we keep (5 vectors total)
```
S_clean_1   shape: (10000,)   clean 1 Hz sinusoid
S_clean_2   shape: (10000,)   clean 3 Hz sinusoid
S_clean_3   shape: (10000,)   clean 5 Hz sinusoid
S_clean_4   shape: (10000,)   clean 7 Hz sinusoid
Sigma       shape: (10000,)   noisy composite (sum of 4 noisy versions)
```

---

## 3. Dataset Design

### 3.1 One sample construction
```
1. Pick target index k in {0, 1, 2, 3}
2. c = one_hot(k)                           shape: (4,)
3. Pick start index i in {0, ..., 9990}
4. x_window = Sigma[i : i+10]              shape: (10,)  <- NOISY input
5. y_target = S_clean_k[i : i+10]          shape: (10,)  <- CLEAN target
```

### 3.2 Input formatting per model
```
FC:
  X = concat(c, x_window)                  shape: (14,)

RNN / LSTM:
  c_repeated = tile(c, 10 times)           shape: (10, 4)
  x_reshaped = x_window.reshape(10, 1)     shape: (10, 1)
  X = hstack(c_repeated, x_reshaped)       shape: (10, 5)
```

---

## 4. Model Architecture Summary

```
+-------------+------------------+------------------+------------------+
|             | FC               | RNN              | LSTM             |
+-------------+------------------+------------------+------------------+
| Input shape | (14,)            | (10, 5)          | (10, 5)          |
| Architecture| 14->64->64->10   | RNN(5,64)+FC(64->10)| LSTM(5,64)+FC(64->10)|
| Output shape| (10,)            | (10,)            | (10,)            |
| Output style| N/A (flat)       | many-to-one      | many-to-one      |
| Temporal    | None             | Short memory     | Long + short     |
| Parameters  | ~5,000           | ~8,000           | ~30,000          |
+-------------+------------------+------------------+------------------+
```

---

## 5. Experiment Design (Part 4)

### 5.1 Noise sensitivity study
```
For each model in [FC, RNN, LSTM]:
  Train once at default noise (2%)
  Then test at noise levels: [1%, 2%, 5%, 8%, 10%, 15%, 20%]
    -> generate 1000 fresh test samples at each noise level
    -> record MSE
  Plot: noise level (x) vs MSE (y) for all 3 models on same graph
```

### 5.2 Expected result shape
```
MSE
 |        FC
 |       /
 |      / RNN
 |     /  /
 |    /  / LSTM
 |___/__/____
 1% 5% 10% 20%   Noise
```

---

## 6. File Structure
```
Signal-Decomposition-with-RNN-LSTM-and-Fully-Connected-Networks/
|
+-- src/
|   +-- main.py                         Entry point
|   +-- signal_decomp/
|       +-- __init__.py
|       +-- sdk/
|       |   +-- __init__.py
|       |   +-- sdk.py                  Single entry point for all logic
|       +-- services/
|       |   +-- __init__.py
|       |   +-- generator.py            Signal generation
|       |   +-- dataset.py              Dataset construction
|       |   +-- models.py               FC, RNN, LSTM architectures
|       |   +-- trainer.py              Training and evaluation loops
|       +-- shared/
|           +-- __init__.py
|           +-- constants.py            Immutable project constants
|           +-- config.py               Config file loader
|
+-- config/
|   +-- setup.json                      All hyperparameters
|
+-- docs/
|   +-- PRD.md                          Main requirements document
|   +-- PRD_fc_network.md               FC architecture PRD
|   +-- PRD_rnn_network.md              RNN architecture PRD
|   +-- PRD_lstm_network.md             LSTM architecture PRD
|   +-- PLAN.md                         This file
|   +-- TODO.md                         Task tracking
|
+-- tests/
|   +-- __init__.py
|   +-- unit/
|       +-- __init__.py
|       +-- test_generator.py           Tests for signal generation
|       +-- test_dataset.py             Tests for dataset construction
|       +-- test_models.py              Tests for model forward passes
|
+-- results/                            Saved plots and metrics
+-- notebooks/                          Analysis notebook
+-- assets/                             Images for README
+-- README.md                           Full lab report
+-- pyproject.toml                      Dependencies and tool config
+-- .gitignore
+-- .env-example
```

---

## 7. Architectural Decisions (ADRs)

### ADR-1: SDK as single entry point
**Decision:** All logic accessed through SignalDecompSDK only.
**Reason:** Enforces clean separation. Any future GUI or CLI
can use the SDK without touching service code.
**Trade-off:** Slightly more code, but much more maintainable.

### ADR-2: Fixed phases shared across all models
**Decision:** Phases drawn once in SDK.__init__, reused for all models.
**Reason:** Fair comparison — models differ only in architecture,
not in the data they see.
**Trade-off:** Less randomness in experiments, but more scientific rigor.

### ADR-3: Many-to-one output for RNN and LSTM
**Decision:** Use final hidden state, not output at every step.
**Reason:** The full 10-step window is needed before prediction.
Many-to-many would predict before seeing the full context.
**Trade-off:** Simpler output layer, slight information loss
from intermediate steps.

### ADR-4: All hyperparameters in setup.json
**Decision:** No hardcoded values in any .py file.
**Reason:** Required by Dr. Segal's guidelines. Makes experiments
reproducible and tunable without touching code.
**Trade-off:** Requires loading config at startup.