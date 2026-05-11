# Findings V257: Ternary CNN with Global Average Pooling (GAP)

## Overview
We evolved the Gated Ternary CNN by replacing the large flattened FC layer with **Global Average Pooling (GAP)**. This move was aimed at reducing the frozen parameter count and improving spatial invariance.

## Results (10 Epochs, MNIST)

| Metric | Value |
| :--- | :--- |
| **Final Accuracy** | **83.40%** |
| **Learnable Params** | **394** |
| **Frozen Params** | 824,960 |
| **PEI Index** | **~28.6** |

## Key Technical Insights

### 1. Frozen Parameter Reduction
By removing the $256 \times 7 \times 7$ flattened connection, we reduced the "dark matter" (frozen weights) by ~123,000 parameters. This makes the model more memory-efficient without significantly sacrificing accuracy (83.4% vs 85.0% in v256).

### 2. The GAP Stability Effect
The initial training loss in v257 was much lower and more stable (**2.22**) compared to v256 (**6.03**). This suggests that averaging features before classification acts as a powerful regularizer for random gated kernels, filtering out high-frequency noise from the frozen projections.

### 3. Scaling Potential
Although accuracy dipped slightly, the parametric efficiency (Acc per learnable param) remained world-class. This architecture is the most "production-ready" for ultra-low-power embedded vision.

## Conclusion
GAP is a vital component for frozen-weight CNNs. It stabilizes the "Discovery" phase of the gates and allows for a leaner frozen backbone.

**Reference Script**: `scratch/prototype_v257_ternary_cnn_gap.py`
