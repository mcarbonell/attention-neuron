# Findings v251d: LR Sweep on Multiplicative Gating

## Goal
Determine the sensitivity of the frozen-weight gated architecture to the Learning Rate (LR). We suspect that higher LRs might be beneficial given the relatively restricted parameter space.

## Results Grid (Final Accuracy %)

| LR | 512 | 1024 | 2048 |
| :--- | :---: | :---: | :---: |
| 1e-3 | 75.30 | 81.40 | 86.41 |
| 5e-3 | 79.99 | 86.90 | 90.71 |
| 1e-2 | 80.67 | 87.17 | 91.09 |
| 5e-2 | 79.25 | 87.48 | 91.37 |

## Analysis
- **Stability**: The architecture proved surprisingly stable even at high LRs like `5e-2`, although some oscillations were observed in the final epochs (e.g., 91.73 -> 91.37).
- **Optimal LR**: For 10 epochs, an LR between `5e-3` and `1e-2` provides the best balance of speed and stability.
- **Speed of Convergence**: With LR `5e-2`, the model reached >90% accuracy in just 1 epoch for $D=2048$.

## Conclusions
[TBD]
