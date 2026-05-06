# Prompt Engineering Log

This log documents the significant prompts used during the development and debugging of the Signal Decomposition project.

## 1. Initial Architecture Design
**Goal:** Create a professional folder structure following Software Excellence Guidelines.
**Prompt:**
> "I need to build a signal decomposition system with FC, RNN, and LSTM. Generate a project structure that follows the SDK pattern, uses uv for dependencies, and separates data generation, model definitions, and training into services under src/."
**Result:** Created the basic skeleton with `src/signal_decomp/sdk` and `src/signal_decomp/services`.

## 2. Model Implementation
**Goal:** Implement the three neural architectures.
**Prompt:**
> "Implement three PyTorch models: 1. A Fully Connected Network for flattened windows. 2. A Vanilla RNN. 3. An LSTM. Use a hidden dimension of 512. For RNN/LSTM, make them deep (3 layers) and include LayerNorm."
**Result:** Generated the initial `models.py`. (Note: Later discovered that 3 layers caused stability issues).

## 3. Debugging RNN Learning Issues
**Goal:** Identify why the RNN MSE was stuck at 0.5.
**Prompt:**
> "The RNN isn't learning; the MSE is stuck at 0.5. Analyze the code for data synchronization issues and gradient flow. Check if the noise generation in build_data.py is matched between FC and RNN."
**Result:** Identified the data synchronization bug in `build_data.py` (noise generated twice) and the gradient explosion in the deep RNN.

## 4. Stability Fixes
**Goal:** Stabilize RNN training.
**Prompt:**
> "Simplify the Vanilla RNN to 1 layer. Add orthogonal initialization to all RNN weights. In the trainer, add gradient clipping with max_norm=1.0. Re-synchronize the dataset builder to return synchronized FC and Sequential data in one call."
**Result:** Successfully lowered RNN loss from 0.5 to < 0.25.

## Best Practices Identified
1.  **Synchronization:** Always generate inputs and targets in a single logic block when dealing with stochastic synthetic data.
2.  **Stability:** Vanilla RNNs require gradient clipping and careful initialization to avoid diverging.
3.  **Simplicity:** For short sequences (W=10), shallower RNNs are more stable and easier to train than deep stacks.
