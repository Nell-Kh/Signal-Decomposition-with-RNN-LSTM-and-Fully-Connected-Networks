# PRD - Signal Decomposition Project

## 1. Goal
Extract a clean target sinusoid from a noisy composite signal using three different neural network architectures[cite: 5].

## 2. Target Signals
We use four base frequencies: **1Hz, 3Hz, 5Hz, and 7Hz**[cite: 5].
* **Input**: A noisy composite signal (the sum of all 4 noisy signals)[cite: 5].
* **Selector**: A 4-dimensional one-hot vector $c$ that tells the model which frequency to extract[cite: 5].

## 3. Input Structures
* **For FC**: A flat vector of 14 elements (4 for $c$ + 10 for the signal window)[cite: 5].
* **For RNN/LSTM**: A sequence of 10 time steps. At each step, the input is 5 elements (4 for $c$ + 1 for the signal sample at that moment)[cite: 5].

## 4. Success Criteria
* **MSE Loss**: We must use Mean Squared Error to measure how close the "cleaned" signal is to the original[cite: 5].
* **Comparison**: Demonstrate that LSTM handles high noise better than a standard RNN or FC network[cite: 5].