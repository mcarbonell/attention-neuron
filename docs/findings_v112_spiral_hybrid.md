# Findings v112: Spiral-Hybrid Model Analysis

## Experiment Summary
We attempted to enhance the Hybrid architecture (v111) by replacing the raster pixels in the Walsh path with **Log-Polar Spiral Sampling** (1024 points), based on the positive results from v95.

## Results

| Model | Representation | Parameters | Test Accuracy |
|-------|----------------|------------|---------------|
| Hybrid v111 | Raster + Islands | 3,850 | **94.20%** |
| **Hybrid v112** | **Spiral + Islands** | 3,850 | 91.73% |

## Why did it perform worse?

1.  **Interpolation vs. Resolution**: The Log-Polar Spiral uses `grid_sample` (bilinear interpolation). In a model with so few parameters (3.8k), the network might not have enough capacity to "denoise" the interpolation artifacts that a larger MLP (v95 used 256 hidden units) could easily handle.
2.  **Walsh-Spiral Mismatch**: The Walsh-Hadamard transform is designed for discrete, rectangularly structured data. By remapping the image into a spiral, we disrupt the natural "sequency" of the strokes as seen by the spectral basis. The raster scan's horizontal/vertical regularity might actually be more "Walsh-friendly".
3.  **Sampling Density**: While the spiral focuses on the center, the standard raster at 28x28 is already very efficient. The spiral's 1024 samples might be redundant or missing crucial edge information that the raster captures more uniformly.

## Conclusion
For ultra-compact models (<5k params), the **Raster + Islands** combination remains the king of efficiency. The "Analog Foveation" of the spiral requires more neural capacity to be effective.
