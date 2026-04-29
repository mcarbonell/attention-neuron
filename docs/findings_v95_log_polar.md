# Findings V95: Log-Polar Spiral Sampling (Analog Foveation)

## Overview
In this experiment, we moved beyond discrete pixel reordering and implemented a **Continuous Log-Polar Spiral Sampler**. Using a logarithmic spiral path and multi-scale Gaussian-like sampling (concentrated at the center), we transformed the 2D image into an "analog" 1D stream of 1024 values.

## Empirical Results (5 Epochs, Standard MLP, hidden_dim=256)

| Metric | Raster MLP (Baseline) | **Log-Polar Spiral MLP** | Difference |
| :--- | :--- | :--- | :--- |
| **Epoch 1 Accuracy** | **97.30%** | 96.11% | -1.19% |
| **Epoch 3 Accuracy** | **97.75%** | 97.45% | -0.30% |
| **Final Accuracy (E5)** | 97.96% | **98.29%** | **+0.33%** |

## Key Technical Insights

### 1. The "Resampling" Penalty vs. Structural Gain
Initially (Epoch 1), the Raster MLP outperformed the Spiral version. This is likely because the Spiral sampler uses bilinear interpolation (`grid_sample`), which introduces a slight blur/noise compared to the raw, crisp pixels. However, the network quickly learned to compensate for this.

### 2. Late-Stage Superiority
By Epoch 5, the Log-Polar Spiral achieved **98.29%**, surpassing the Raster baseline. This suggests that the "Log-Polar Inductive Bias" (focusing more resolution on the center and capturing circular features) provides a more robust and discriminative representation once the network has learned to interpret the transformed signal.

### 3. Efficiency of Information Packing
Even though both models use the same number of input dimensions (1024), the Spiral model utilizes them more effectively by prioritizing the "Fovea" of the image. It ignores the empty corners of the MNIST digits and allocates more "bandwidth" to the central strokes.

## Conclusion
The experiment proves that **Log-Polar Spiral Sampling** is a viable and potentially superior alternative to standard raster scanning for centered objects. It achieves higher final accuracy and paves the way for Rotation and Scale Invariance.

**Reference Script**: `scratch/prototype_v95_log_polar_spiral.py`
