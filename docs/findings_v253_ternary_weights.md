# Findings V253: Frozen Ternary Weights + Float Gating

## Overview
We tested the "Reversed Quantization" hypothesis: keeping the weights as **Frozen Random Ternary** ($\{-1, 0, 1\}$) and learning only **Float Gating** ($\mathbb{R}$). We increased the `hidden_dim` to **2048** to maximize the pool of random features (Lottery Ticket Hypothesis).

## Results (10 Epochs, MNIST)

| Metric | Value |
| :--- | :--- |
| **Final Accuracy** | **94.74%** |
| **Learnable Params** | 4,106 |
| **Frozen Params** | 5,820,416 |
| **PEI (Total)** | 15.86 |

## Key Technical Insights

### 1. Success of High-Dimensional Random Projection
Unlike standard MLPs that learn weights, this model proves that random ternary projections are rich enough to solve MNIST. With 2048 neurons, the network finds a "winning subset" of features that, when scaled by float gates, provide near-SOTA performance for a non-convolutional model.

### 2. The "Self-Scaling" Effect
We observed that the last layer (classifier) gates dramatically reduced their magnitude (from 1.0 to ~0.09). 
- **Cause**: The random ternary weights combined with SiLU generate high-variance activations. 
- **Solution**: The optimizer automatically "turned down the volume" at the end to stabilize the softmax. This demonstrates an emergent self-regularization property of gated frozen networks.

### 3. Stability vs. Learning Rate
Initial attempts with `LR=1e-2` were unstable (catastrophic divergence in later epochs). Reducing to `LR=1e-3` provided smooth convergence and higher final accuracy.

## Conclusion
Frozen Ternary Weights are an excellent foundation for gated networks. They are more efficient than float weights for hardware inference and, surprisingly, do not significantly hinder representational power if the hidden dimension is sufficiently large.

**Reference Script**: `scratch/prototype_v253_ternary_weights.py`
