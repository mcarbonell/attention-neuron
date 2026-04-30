# Findings v110: Tri-Walsh Hybrid Model (The "Cerebro-Cerebelo" Architecture)

## Experiment Summary
We implemented a hybrid architecture that processes MNIST using two parallel paths:
1.  **Structural Path**: A **Triangular Layer** (32 units) processing **Island Signatures** (56D).
2.  **Spectral Path**: A **Walsh Layer** (32 units, k=16) processing **Raw Pixels** (784D).

This ensemble combines morphological skeletal information with high-frequency spectral details.

## Results

| Model | Parameters | Test Accuracy | Compression vs Baseline |
|-------|------------|---------------|-------------------------|
| **Baseline MLP (Pixels)** | 25,450 | 96.13% | 1x |
| **Hybrid Tri-Walsh (v110)** | **1,290** | **93.03%** | **~20x** |

## Key Insights

1.  **Complementary Feature Spaces**: Neither Islands nor Pixels are sufficient on their own to reach 93% with such low parameter counts. The synergy between the local structural bias of Triangular neurons and the global spectral bias of Walsh neurons is the key.
2.  **High Parameter Efficiency**: Achieving >93% accuracy with only 1.3k parameters is a record for this repository. It demonstrates that matching neuron types to their "natural" data representations is far more effective than simply using wider dense layers.
3.  **Hardware Friendly**: Both the Triangular and Walsh layers use highly efficient operations (distance calculations and simple +/- additions) compared to the floating-point multiplications of standard dense layers.

## Next Steps: Experiment v111
Scaling the Hybrid Model to ~4k parameters by increasing the width of both paths (h=96) to target >95% accuracy.
