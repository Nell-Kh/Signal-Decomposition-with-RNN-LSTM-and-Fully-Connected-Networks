# TODO.md — Task Tracking
# Signal Decomposition with RNN, LSTM, and Fully Connected Networks

**Authors:** Nell Khoury, Yanal Serhan
**Version:** 1.00
**Last Updated:** 05.05.2026

Status legend:
  [x] = Done
  [-] = In Progress
  [ ] = Not Started

---

## PHASE 1 — Documentation (Must be done BEFORE any code)

### 1.1 PRD Documents
- [x] Write docs/PRD.md — main project requirements
- [x] Write docs/PRD_fc_network.md — FC architecture requirements
- [x] Write docs/PRD_rnn_network.md — RNN architecture requirements
- [x] Write docs/PRD_lstm_network.md — LSTM architecture requirements
- [x] Write docs/PLAN.md — architecture and planning document
- [x] Write docs/TODO.md — this file

### 1.2 Project Setup
- [x] Initialize project with uv
- [x] Create full folder structure
- [x] Create config/setup.json with all hyperparameters
- [x] Create src/signal_decomp/shared/constants.py
- [x] Create .gitignore (include .venv, __pycache__, .env)
- [x] Create .env-example
- [ ] Create pyproject.toml with ruff and pytest config
- [ ] Install all dependencies with uv add

---

## PHASE 2 — Signal Generation

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
- [ ] Verify output shapes are correct: clean (4,10000), composite (10000,)
- [ ] Verify noise is truly random (not deterministic) per sample

### 2.2 dataset.py
- [x] Implement build_dataset(clean, composite, num_samples, window_size, model_type)
- [x] Generate one-hot vectors using np.eye(4)
- [x] Randomly pick target frequency index per sample
- [x] Randomly pick start index i in {0, ..., 9990}
- [x] Extract noisy composite window x_window (10 samples)
- [x] Extract clean target window y_target (10 samples)
- [x] Format FC input: concat(c, x_window) -> shape (14,)
- [x] Format RNN/LSTM input: (10, 5) sequence with c repeated per step
- [ ] Verify FC shapes: X (10000, 14), Y (10000, 10)
- [ ] Verify RNN shapes: X (10000, 10, 5), Y (10000, 10)
- [ ] Verify no data leakage between train and test sets

---

## PHASE 3 — Neural Network Models

### 3.1 models.py — Fully Connected Network
- [x] Define FullyConnectedNetwork class with docstring
- [x] Input layer: Linear(14, 64) + ReLU
- [x] Hidden layer: Linear(64, 64) + ReLU
- [x] Output layer: Linear(64, 10) — no activation
- [x] Implement forward() method
- [ ] Verify output shape: (batch_size, 10)
- [ ] Verify no activation on output layer

### 3.2 models.py — Vanilla RNN
- [x] Define VanillaRNN class with docstring
- [x] RNN layer: input_size=5, hidden_size=64, batch_first=True
- [x] Output projection: Linear(64, 10)
- [x] Implement forward() — use final hidden state h_n only
- [x] Squeeze h_n correctly before linear layer
- [ ] Verify output shape: (batch_size, 10)
- [ ] Verify many-to-one: only final hidden state used

### 3.3 models.py — LSTM
- [x] Define LSTMNetwork class with docstring
- [x] LSTM layer: input_size=5, hidden_size=64, batch_first=True
- [x] Output projection: Linear(64, 10)
- [x] Implement forward() — use final hidden state h_n only
- [x] Unpack (h_n, c_n) correctly, use only h_n
- [ ] Verify output shape: (batch_size, 10)
- [ ] Verify cell state c_n is correctly ignored in output

---

## PHASE 4 — Training Infrastructure

### 4.1 trainer.py — train_model
- [x] Convert numpy arrays to PyTorch tensors
- [x] Create TensorDataset and DataLoader
- [x] Initialize MSELoss criterion
- [x] Initialize Adam optimizer
- [x] Implement training loop with zero_grad, forward, backward, step
- [x] Track loss per epoch in loss_history list
- [x] Print loss every 10 epochs
- [ ] Verify loss decreases over epochs for all 3 models
- [ ] Verify model.train() is called before loop

### 4.2 trainer.py — evaluate_model
- [x] Implement evaluate_model(model, x_np, y_np, batch_size)
- [x] Call model.eval() before evaluation
- [x] Use torch.no_grad() context
- [x] Return average MSE across all batches
- [ ] Verify model.eval() disables dropout/batchnorm correctly

### 4.3 sdk.py — SignalDecompSDK
- [x] Implement __init__ with config loading and time vector
- [x] Generate phases ONCE in __init__ and store as self._phases
- [x] Implement run_training_pipeline(model_name)
- [x] Implement evaluate(model, model_name, noise_pct)
- [x] Implement _build_model(model_name) helper
- [x] Raise ValueError for unknown model names
- [ ] Verify phases are truly fixed across all 3 model runs
- [ ] Verify SDK is the only entry point (no direct service calls from main)

---

## PHASE 5 — Testing (minimum 85% coverage)

### 5.1 test_generator.py
- [ ] Test generate_base_signals returns correct shapes (4, 10000) and (4,)
- [ ] Test phases are in range [0, 2*pi]
- [ ] Test clean signal values are in range [-A, +A]
- [ ] Test apply_noise returns noisy_signals (4, N) and composite (N,)
- [ ] Test composite equals sum of noisy_signals
- [ ] Test noise actually changes values (noisy != clean)
- [ ] Test higher noise_amplitude_pct produces larger deviations

### 5.2 test_dataset.py
- [ ] Test FC output shapes: X (num_samples, 14), Y (num_samples, 10)
- [ ] Test RNN output shapes: X (num_samples, 10, 5), Y (num_samples, 10)
- [ ] Test one-hot vectors are valid (sum to 1, binary values)
- [ ] Test window extraction stays within signal bounds
- [ ] Test all 4 target frequencies appear in dataset
- [ ] Test FC first 4 features match a valid one-hot vector

### 5.3 test_models.py
- [ ] Test FC forward pass: input (batch, 14) -> output (batch, 10)
- [ ] Test RNN forward pass: input (batch, 10, 5) -> output (batch, 10)
- [ ] Test LSTM forward pass: input (batch, 10, 5) -> output (batch, 10)
- [ ] Test all models produce finite values (no NaN or Inf)
- [ ] Test models are in train mode after instantiation

### 5.4 Run coverage check
- [ ] Run: uv run pytest tests/ --cov=src --cov-report=term-missing
- [ ] Verify coverage >= 85%
- [ ] Fix any uncovered critical paths

---

## PHASE 6 — Configuration & Code Quality

### 6.1 Ruff linting
- [ ] Run: uv run ruff check src/
- [ ] Fix all linting errors (target: 0 errors)
- [ ] Run: uv run ruff check tests/
- [ ] Fix all linting errors in tests

### 6.2 Code quality checklist
- [ ] Every function has a docstring explaining WHY not just WHAT
- [ ] All variable names are descriptive and in English
- [ ] No file exceeds 150 lines
- [ ] No hardcoded values in any .py file
- [ ] All imports are used (no unused imports)
- [ ] No print statements except in trainer.py epoch logging

### 6.3 pyproject.toml
- [ ] Add ruff configuration (line-length=100, select E,F,W,I,N)
- [ ] Add pytest configuration
- [ ] Add coverage configuration (fail_under=85)
- [ ] Verify uv.lock is up to date

---

## PHASE 7 — Analysis & Results (Part 4)

### 7.1 Noise sensitivity experiment
- [ ] Train FC model at default noise (2%)
- [ ] Train RNN model at default noise (2%)
- [ ] Train LSTM model at default noise (2%)
- [ ] Evaluate all 3 at noise levels: [1%, 2%, 5%, 8%, 10%, 15%, 20%]
- [ ] Save MSE results to results/noise_sensitivity.json

### 7.2 Plots — save all to results/ and assets/
- [ ] Plot 1: Training loss curves for all 3 models (same graph)
- [ ] Plot 2: Noise level vs MSE for all 3 models (main analysis plot)
- [ ] Plot 3: Example clean vs noisy vs predicted signals (per model)
- [ ] Plot 4: Bar chart of final test MSE per model at each noise level
- [ ] All plots: clear labels, legend, title, grid, high resolution

### 7.3 Analysis conclusions
- [ ] Identify exact noise level where RNN beats FC
- [ ] Identify exact noise level where LSTM beats RNN
- [ ] Write explanation of WHY each crossover happens
- [ ] Find settings where RNN is clearly better than FC
- [ ] Find settings where LSTM is clearly better than RNN

---

## PHASE 8 — Notebook

### 8.1 notebooks/analysis.ipynb
- [ ] Cell 1: imports and config loading
- [ ] Cell 2: signal generation and visualization
- [ ] Cell 3: dataset construction and shape verification
- [ ] Cell 4: train all 3 models
- [ ] Cell 5: noise sensitivity analysis with plots
- [ ] Cell 6: example predictions vs ground truth
- [ ] Cell 7: conclusions and summary table
- [ ] Run all cells top to bottom without errors

---

## PHASE 9 — README (Lab Report)

### 9.1 Content
- [ ] Project overview and goal
- [ ] Mathematical formulas for signal generation and noise
- [ ] Dataset construction explanation with shapes
- [ ] Architecture description for each model with justification
- [ ] Training configuration table
- [ ] Screenshot: project folder structure in VS Code
- [ ] Screenshot: training loss curves
- [ ] Screenshot: noise sensitivity plot (main result)
- [ ] Screenshot: example predictions
- [ ] Results table: MSE per model per noise level
- [ ] Analysis: when does each model win and why
- [ ] How to install and run (step by step)
- [ ] Link to GitHub repository

### 9.2 Quality
- [ ] README reads as a complete lab report
- [ ] All screenshots are high resolution and labeled
- [ ] All graphs have titles, axis labels, legend
- [ ] Formulas are clearly written
- [ ] Conclusions are clear and backed by results

---

## PHASE 10 — Final Checklist (before submission)

- [ ] uv run python src/main.py runs without errors
- [ ] uv run pytest tests/ --cov=src passes with >= 85% coverage
- [ ] uv run ruff check src/ returns 0 errors
- [ ] All docs/ files are complete and up to date
- [ ] README.md is complete with screenshots and analysis
- [ ] results/ folder contains all saved plots
- [ ] notebooks/analysis.ipynb runs top to bottom cleanly
- [ ] .gitignore includes .venv, __pycache__, .env
- [ ] No API keys or secrets in any file
- [ ] GitHub repository is public with clean commit history
- [ ] PDF submitted with GitHub link inside
- [ ] Submission before 07.05.2026