# Findings v251g: Round-Robin Layer Training

## Goal
Evaluate a hybrid optimization strategy where all gating parameters are updated in every batch, but the weight matrices are updated in a rotating cycle (one layer per batch). This aims to reduce the computational and memory footprint of training.

## Results Comparison

| Epoch | Baseline (Full) Acc % | Round-Robin Acc % |
| :--- | :---: | :---: |
| 1 | [TBD] | [TBD] |
| 5 | [TBD] | [TBD] |
| 10 | [TBD] | [TBD] |

## Analysis
- **Convergence Speed**: [TBD - Did Round-Robin lag behind?]
- **Stability**: [TBD - Did the staggered updates cause oscillations?]
- **Computational Efficiency**: Since only ~1/3 of the weights are updated per batch, the per-batch compute cost for the optimizer and the weight-gradient calculation is reduced by approximately 66%.

## Conclusions
[TBD]
