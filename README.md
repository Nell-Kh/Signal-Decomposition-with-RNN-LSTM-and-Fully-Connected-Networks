# Signal Decomposition: An Analysis of Temporal Robustness vs. Architectural Simplicity

**Authors:** Nell Khoury, Yanal Serhan
**Affiliation:** Signal Processing & Machine Learning Lab
**Version:** 1.0.0

## 1. Project Overview
This project implements a neural signal decomposition system designed to extract individual sinusoidal components from a noisy composite signal. We compare three distinct architectures—**Fully Connected (FC)**, **Vanilla RNN**, and **LSTM**—to evaluate their performance across varying noise regimes and sequence lengths.

## 2. Mathematical Formulation

### 2.1 Signal Generation
The composite signal $\tilde{\Sigma}(t)$ is generated as the sum of four sinusoids with frequencies $f \in \{1, 3, 5, 7\}$ Hz:
$$\tilde{\Sigma}(t) = \sum_{k=1}^{4} (A + \epsilon_{A,k}) \sin(2\pi f_k t + \phi_k + \epsilon_{\phi,k})$$
Where:
*   $A$: Base amplitude (1.0)
*   $\epsilon_{A}$: Uniform amplitude noise $\mathcal{U}(-0.02A, 0.02A)$
*   $\epsilon_{\phi}$: Uniform phase noise $\mathcal{U}(-0.1, 0.1)$

### 2.2 Dataset Construction
Each training sample consists of a window $W=10$ (10ms at 1000Hz).
*   **FC Input:** $\mathbf{x} = [\mathbf{c}, \tilde{\Sigma}_{i}, \dots, \tilde{\Sigma}_{i+9}] \in \mathbb{R}^{14}$
*   **Sequential Input:** $\mathbf{X} = [\mathbf{x}_1, \dots, \mathbf{x}_{10}] \in \mathbb{R}^{10 \times 5}$, where $\mathbf{x}_t = [\mathbf{c}, \tilde{\Sigma}_{t}]$
*   **Target:** $\mathbf{y} = [y_i, \dots, y_{i+9}] \in \mathbb{R}^{10}$ (Clean target sinusoid)

## 3. Neural Architectures

### 3.1 Fully Connected (Baseline)
A deep feed-forward network with 2 hidden layers (512 units each) and BatchNorm for rapid convergence.
*   **Justification:** High efficiency for short-range spatial patterns.

### 3.2 Vanilla RNN
A single-layer recurrent network stabilized with **Orthogonal Weight Initialization** and **Gradient Norm Clipping**.
*   **Justification:** Minimalist temporal model designed to test basic recurrent stability.

### 3.3 LSTM (Long Short-Term Memory)
A 2-layer gated recurrent network designed to capture longer-range dependencies and resist gradient vanishing.
*   **Justification:** Advanced temporal modeling capable of distinguishing frequencies through phase-aware memory gates.

## 4. Results & Analysis

### 4.1 Training Convergence
All models successfully converged, with the FC model reaching the lowest training MSE.
![Training Loss](results/training_loss.png)

### 4.2 Robustness Analysis (Noise Sensitivity)
We evaluated the models across noise levels from 1% to 20%. 
![Noise Sensitivity](results/noise_sensitivity.png)

| Noise Level | FC MSE | RNN MSE | LSTM MSE |
| :--- | :--- | :--- | :--- |
| 1% | 0.063 | 0.209 | 0.138 |
| 10% | 0.070 | 0.235 | 0.165 |
| 20% | 0.098 | 0.289 | 0.252 |

### 4.3 Interpretation (The "Crossover")
At a sequence length of $W=10$, the **FC model** remains superior due to the low dimensionality of the input space. However, as hypothesized in our **Sequence Length Scalability Study (See docs/PLAN.md)**, we expect the LSTM to outperform the FC model as the window size $W$ increases or as the signal-to-noise ratio significantly degrades.

## 5. Installation & Usage
1.  **Dependencies:** Managed via `uv`.
    ```bash
    uv sync
    ```
2.  **Generate Data:**
    ```bash
    uv run python build_data.py
    ```
3.  **Run Pipeline:**
    ```bash
    uv run python main.py
    ```

---
**Repository:** [https://github.com/Nell-Kh/Signal-Decomposition-with-RNN-LSTM-and-Fully-Connected-Networks](https://github.com/Nell-Kh/Signal-Decomposition-with-RNN-LSTM-and-Fully-Connected-Networks)
**Submission Date:** May 7, 2026
