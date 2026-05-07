# Signal Decomposition with RNN, LSTM, and Fully Connected Networks

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/nellkh/Signal-Decomposition-with-RNN-LSTM-and-Fully-Connected-Networks)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📖 Executive Summary
This project evaluates three neural architectures—**Fully Connected (FC)**, **Vanilla RNN**, and **LSTM**—on their ability to perform real-time signal decomposition. The task is to extract a specific target sinusoid from a noisy composite signal.

### 🏆 Key Results
*   **Winner:** The **Fully Connected** model achieved the lowest MSE (**0.041**), benefiting from full spatial context of the 10-sample window.
*   **Recurrent Standout:** The **LSTM** (**0.166 MSE**) significantly outperformed the **Vanilla RNN** (**0.354 MSE**), demonstrating the power of gated memory units in noisy environments.

---

## 📊 Final Performance Metrics

| Architecture | Test MSE (2% Noise) | Robustness (20% Noise) | Final Rank |
| :--- | :--- | :--- | :--- |
| **Fully Connected** | **0.0413** | **0.1232** | 🥇 |
| **LSTM** | 0.1670 | 0.2520 | 🥈 |
| **Vanilla RNN** | 0.3542 | 0.3906 | 🥉 |

---

## 🖼️ Test Graphs & Visualizations

### 1. Training Convergence
![Training Loss](results/training_loss.png)
*The LSTM shows superior, stable convergence compared to the jittery performance of the Vanilla RNN.*

### 2. Noise Sensitivity (Stress Test)
![Noise Sensitivity](results/noise_sensitivity.png)
*Evaluation of model robustness across noise levels from 1% to 20%.*

### 3. Visual Wave Reconstruction
![Visual Reconstruction](results/visual_reconstruction.png)
*Real-time prediction vs. clean ground truth on unseen test data.*

---

## 🔍 Architectural Findings
1.  **Spatial Context:** The FC model's success confirms that for small window sizes ($W=10$), processing the window as a static vector is more efficient than sequential processing.
2.  **Gated Memory:** The performance gap between LSTM and RNN proves that "forget gates" are crucial for filtering noise in temporal sequences.
3.  **Baseline Success:** All models successfully learned the basic frequencies, but only the FC and LSTM maintained phase-alignment under stress.

---

## 🚀 Reproduction
1. **Install:** `uv sync`
2. **Build Data:** `uv run python scripts/build_data.py`
3. **Train:** `uv run python scripts/main.py`

**Submission Date:** May 7, 2026
**Created by:** Nell Khoury
