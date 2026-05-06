# TODO.md — Task Tracking
# Signal Decomposition with RNN, LSTM, and Fully Connected Networks

**Authors:** Nell Khoury, Yanal Serhan
**Version:** 1.00
**Last Updated:** 06.05.2026

Status legend:
  [x] = Done
  [/] = In Progress
  [ ] = Not Started

---

## PHASE 1 — Documentation (COMPLETE)

### 1.1 PRD Documents
- [x] Write docs/PRD.md — main project requirements
- [x] Write docs/PRD_fc_network.md — FC architecture requirements
- [x] Write docs/PRD_rnn_network.md — RNN architecture requirements
- [x] Write docs/PRD_lstm_network.md — LSTM architecture requirements
- [x] Write docs/PLAN.md — architecture and planning document
- [x] Write docs/TODO.md — this file
- [x] Write docs/PROMPT_LOG.md — AI interaction log

### 1.2 Project Setup
- [x] Initialize project with uv
- [x] Create full folder structure
- [x] Create config/setup.json with all hyperparameters
- [x] Create src/signal_decomp/shared/constants.py
- [x] Create .gitignore (include .venv, __pycache__, .env)
- [x] Create .env-example
- [x] Create pyproject.toml with ruff and pytest config
- [x] Install all dependencies with uv add

---

## PHASE 2 — Signal Generation (COMPLETE)

### 2.1 generator.py
- [x] Implement generate_base_signals(t, frequencies, amplitude)
- [x] Generate clean sinusoids for all 4 frequencies
- [x] Draw phases randomly ONCE and return them
- [x] Implement apply_noise(clean, phases, frequencies, t, A, noise_amp, noise_phi)
- [x] Apply per-sample Uniform amplitude noise epsilon_A
- [x] Apply per-sample Uniform phase noise epsilon_phi
- [x] Apply correct formula: (A + epsilon_A) * sin(2pi*f*t + phi + epsilon_phi)
- [x] Sum all 4 noisy sinusoids into composite
- [x] Return both noisy_signals (4, N) and composite (N,)
- [x] Verify output shapes are correct: clean (4,10000), composite (10000,)
- [x] Fix: Support for fixed_phases in generator for evaluation consistency

### 2.2 dataset.py
- [x] Implement build_dataset_with_fresh_noise()
- [x] Generate one-hot vectors using np.eye(4)
- [x] Randomly pick target frequency index per sample
- [x] Randomly pick start index i
- [x] Extract noisy composite window x_window (10 samples)
- [x] Extract clean target window y_target (10 samples)
- [x] Format FC input: concat(c, x_window) -> shape (14,)
- [x] Format RNN/LSTM input: (10, 5) sequence with c repeated per step
- [x] **CRITICAL FIX:** Synchronize noise across all model formats in one pass

---

## PHASE 3 — Neural Network Models (COMPLETE)

### 3.1 models.py — Fully Connected Network
- [x] Define FullyConnectedNetwork class with docstring
- [x] Architecture: Linear(14, 512) -> ReLU -> BatchNorm -> Linear(512, 512) -> ReLU -> Linear(512, 10)
- [x] Implement forward() method

### 3.2 models.py — Vanilla RNN
- [x] Define VanillaRNN class with docstring
- [x] RNN layer: input_size=5, hidden_size=512, batch_first=True
- [x] Output projection: Linear(512, 256) -> ReLU -> Linear(256, 1)
- [x] **STABILITY FIX:** Reduced to 1 layer for stability
- [x] **STABILITY FIX:** Orthogonal Weight Initialization

### 3.3 models.py — LSTM
- [x] Define LSTMNetwork class with docstring
- [x] LSTM layer: input_size=5, hidden_size=512, batch_first=True, num_layers=2
- [x] Output projection: Linear(512, 256) -> ReLU -> Linear(256, 1)
- [x] **STABILITY FIX:** Orthogonal Weight Initialization

---

## PHASE 4 — Training Infrastructure (COMPLETE)

### 4.1 trainer.py
- [x] Convert numpy arrays to PyTorch tensors
- [x] Create TensorDataset and DataLoader
- [x] Initialize MSELoss criterion
- [x] Initialize Adam optimizer
- [x] **STABILITY FIX:** Implement Gradient Clipping (max_norm=1.0)
- [x] Track loss per epoch

### 4.2 sdk.py — SignalDecompSDK
- [x] Implement __init__ with phase loading from metadata
- [x] Implement train(), test(), evaluate_at_noise()
- [x] Ensure SDK is the only entry point for external callers

---

## PHASE 5 — Testing & Quality (IN PROGRESS)

### 5.1 Linting & Formatting
- [/] Run: uv run ruff check src/
- [/] Fix all linting errors (target: 0 errors)

### 5.2 Unit Tests (Target: 85% coverage)
- [ ] Test generator.py
- [ ] Test dataset.py
- [ ] Test models.py
- [ ] Run coverage report

---

## PHASE 6 — Analysis & Results (COMPLETE)

### 6.1 Results Generation
- [x] Train all 3 models at default noise (2%)
- [x] Evaluate all 3 at noise levels: [1%, 2%, 5%, 8%, 10%, 15%, 20%]
- [x] Save MSE results to results/noise_sensitivity.json
- [x] Plot 1: Training loss curves
- [x] Plot 2: Noise level vs MSE (Crossover analysis)

---

## PHASE 7 — Submission

### 7.1 README (Lab Report)
- [/] Finalize README.md with screenshots and crossover analysis
- [x] Link to GitHub repository
- [ ] Ensure PDF submission contains the link

### 7.2 Final Check
- [ ] Verify everything runs with `uv run python main.py`
- [ ] Ensure .gitignore is correct