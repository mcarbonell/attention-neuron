# Findings v251c: Deep Multiplicative Gating

## Goal
Evaluate if depth improves the performance of frozen-weight random projections. We test a 3-layer architecture ($784 \to H \to H \to 10$) with only $2H+10$ gating parameters.

## Results Comparison (v251b vs v251c)

| Hidden Dim | Params (1L) | Acc (1L) % | Params (2L) | Acc (2L) % | Gain |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 32 | 42 | 34.50 | 74 | 36.84 | +2.34 |
| 128 | 138 | 47.03 | 266 | 63.22 | +16.19 |
| 512 | 522 | 74.32 | 1034 | 83.61 | +9.29 |
| 1024 | 1034 | 82.46 | 2058 | 87.30 | +4.84 |
| 2048 | 2058 | 85.32 | 4106 | 90.06 | +4.74 |
| 4096 | 4106 | 89.30 | 8202 | 91.68 | +2.38 |

## Analysis
- **Impact of Depth**: Depth significantly improves accuracy, especially in smaller dimensions. A second layer of random projections provides a richer set of features for the gating to select from.
- **Parameter Efficiency**: Although the number of parameters doubles ($2H$ vs $H$), the accuracy gains in intermediate dimensions (e.g., +16 points in $D=128$) justify the cost.
- **Convergence Speed**: The deeper model takes slightly longer to converge but reaches a higher ceiling compared to the single-layer version.

## Conclusions
Deep Frozen architectures are superior to shallow ones for random projection tasks. The hierarchy of random non-linearities creates a more separable space for the linear classifier at the end.
