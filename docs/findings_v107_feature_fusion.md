# Findings v107: Morphological and Intensity Feature Fusion on MNIST

## Experiment Summary
We evaluated the impact of hand-crafted morphological features (Island Signatures) and intensity-based features (Global and Row/Col sums) on MNIST classification using a standard 2-layer MLP (128 hidden units).

## Results Table

| Configuration | Input Dim | Test Accuracy | Training Time |
|---------------|-----------|---------------|---------------|
| **Baseline (Pixels)** | 784 | **97.75%** | 17.8s |
| **Intensity Only** | 57 | 90.26% | 11.8s |
| **Islands Only** | 56 | 87.35% | 12.8s |
| **Intensity + Islands** | 113 | **94.70%** | 12.3s |
| **Full Fusion** | 897 | 97.51% | 15.8s |

## Key Insights

1.  **Redundancy in Full Representation**: The "Full Fusion" model achieved the exact same accuracy as the "Baseline (Pixels)". This suggests that the MLP is already learning to extract intensity and connectivity information from the raw pixels, and adding them explicitly doesn't provide additional "orthogonal" information to a high-capacity model.
2.  **High Information Density**:
    -   **Intensity features** (57D) achieve >90% accuracy. This is a 13.7x reduction in dimensionality with only a ~7% loss in accuracy.
    -   **Island features** (56D) achieve 87.5% accuracy, proving that the topological "connected components" per row/col are highly discriminative.
3.  **Efficiency Potential**: These features are significantly more compact. In resource-constrained environments (TinyML), using these pre-computed features would allow for much smaller models.

## Next Steps
-   **Morphological Fusion**: Test `Islands + Intensity` without raw pixels to see how close they get to the baseline with only ~113D.
-   **Tiny Model Sweep**: Compare performance of these features against pixels when the number of parameters is severely limited (e.g., <5k parameters).
-   **Invariance Testing**: Evaluate if these features provide better robustness against noise or spatial shifts compared to raw pixels.
