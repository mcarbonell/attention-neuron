# Findings V259: Residual Spectrum-Gated Transformer (RSGT)

## Overview
We introduced **Residual Connections** and doubled the hidden dimension to **1024** in the Spectrum-Gated Transformer. The goal was to solve the convergence plateaus seen in the vanilla SGT.

## Results (10 Epochs, MNIST)

| Metric | Value |
| :--- | :--- |
| **Final Accuracy** | **72.12%** |
| **Learnable Params** | **3,082** |
| **Frozen Params** | 2,127,872 |
| **Mixing Method** | Spectral (Hadamard 64x64) |

## Key Technical Insights

### 1. Residual Stability
Adding $x = x + Block(x)$ allowed the model to start with a much higher loss (~16.0) but converge steadily. The residuals preserve the "Raw Patch Information" throughout the depth, which is critical for frozen backbones where the signal can be easily lost in random projections.

### 2. Capacity vs. Geometry
Despite doubling the parameters (1024 dim), the improvement was marginal (+1% over v258). This confirmed the **Geometry Law**: No amount of parameter scaling can compensate for the lack of positional awareness in a global mixing architecture.

### 3. Training Overhead
The 1024-dimension matmuls with 64 patches significantly increased training time. While the "Vectorization First" rule kept it manageable, the efficiency gain (PEI) dropped compared to v258.

## Conclusion
Residuals are necessary for deep gated networks, but for Vision tasks, they must be paired with Positional Encodings to unlock high accuracy.

**Reference Script**: `scratch/prototype_v259_spectrum_residual.py`
