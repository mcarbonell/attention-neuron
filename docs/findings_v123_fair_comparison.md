# Findings V123: Fair Smooth Comparison (Walsh vs DCT)

## Executive Summary
This experiment matched parameter counts across all variants to isolate the effect of the reconstruction method (Smoothing vs Blocky) and the basis type (Walsh vs DCT).

## Results (MNIST, 10 Epochs, 128 Hidden Neurons)

| Method | Resolution (K) | Accuracy | Parameters | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Walsh Blocky** | 8 | 0.9618 | 9,866 | - |
| **Walsh Smooth** | 8 | **0.9707** | 9,866 | **Smoothing Wins (+0.9%)** |
| **Walsh Blocky** | 16 | 0.9818 | 34,442 | - |
| **Walsh Smooth** | 16 | **0.9823** | 34,442 | **Smoothing Wins (+0.05%)** |
| **DCT Pure** | 8 | **0.9756** | 9,866 | **Spectral Wins (+0.5%)** |
| **DCT Smooth** | 8 | 0.9701 | 9,866 | Interpolation degrades DCT |
| **DCT Pure** | 16 | 0.9801 | 34,442 | - |
| **DCT Smooth** | 16 | **0.9812** | 34,442 | **Smooth Wins slightly** |

## Key Insights
1. **Walsh loves Smoothing**: Because Walsh bases are discontinuous step functions, bilinear interpolation is essential to create organic filters. The gain is massive at low resolutions (K=8).
2. **DCT loves Purity**: For DCT, the "Pure" spectral padding (sinc interpolation) is superior at low resolutions. Bilinear interpolation actually hurts DCT at K=8 by introducing artifacts that break its trigonometric properties.
3. **Efficiency Milestone**: DCT Pure (K=8) achieves **97.56%** with only **9k parameters**, proving to be the most efficient "compact" representation so far.

## Conclusion
Use **Smooth Walsh** if using Walsh bases, but use **Pure Spectral Padding** if using DCT bases.
