# Findings v120: Radical Cosine Experiment

## Summary
We tested the "radical" hypothesis of using the **cosine of the sum** as a pre-activation or activation function in a standard MLP (64 hidden units).

## Metrics Comparison

| Variant | Final Test Acc | Final Loss | Init Acc (Epoch 1) | Overhead (s) |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (ReLU)** | **97.28%** | **0.0884** | 93.37% | 135.96 |
| **Pure Cosine** | 96.45% | 0.1179 | 93.89% | 141.31 |
| **Pure Sine** | 96.59% | 0.1129 | **93.98%** | 141.26 |
| **Cosine + ReLU** | 96.13% | 0.1362 | 92.77% | 131.76 |

## Key Observations

1. **Faster Initial Learning**: Periodic activations (`Sine` and `Cosine`) reached higher accuracy in the first epoch compared to `ReLU`. This suggests that sinusoidal functions are better at capturing initial spatial correlations in MNIST digits.
2. **Convergence Stability**: `ReLU` showed a steady improvement, while the periodic variants showed more fluctuations in later epochs. This is typical for non-monotonic activations where the gradient can change sign frequently.
3. **The "Cosine+ReLU" literal proposal**: This variant was the weakest. This is likely because `ReLU(cos(z))` clips half of the cosine wave to zero, losing significant information compared to `PureCosine` which preserves the negative phase.
4. **Efficiency**: There was no significant difference in `Internal Overhead` or `Eval Time`. Periodic functions are computationally efficient in PyTorch.

## Conclusion
The "radical" cosine experiment proves that periodic activations are viable and potentially faster starters, but for a standard MLP architecture, they require more careful tuning (perhaps learning rate or weight initialization) to beat the stable `ReLU` baseline.

> [!TIP]
> This experiment opens the door to **Hybrid Periodic Neurons** where only some layers or a subset of neurons use periodic activations to capture high-frequency patterns.
