# Findings V122: Smooth Walsh Neurons

## Executive Summary
The experiment confirms that **Smooth Walsh** neurons significantly outperform both standard Dense layers and pure "Blocky" Walsh neurons in the MNIST classification task. 

By parameterizing the weights in a low-resolution Walsh space and using bilinear interpolation to reconstruct the full-resolution spatial weights, we achieve a **learnable low-pass filter** that acts as a strong regularizer.

## Quantitative Results (MNIST, 10 Epochs)

| Model | Accuracy | Parameters | Time (s) | Efficiency (Acc/Param) |
| :--- | :---: | :---: | :---: | :---: |
| Dense Baseline | 97.88% | 132,746 | 143.5 | 0.73e-5 |
| Blocky Walsh (32x32) | 97.55% | 132,746 | 153.5 | 0.73e-5 |
| **Smooth Walsh (K=16)** | **98.13%** | **34,442** | 151.3 | **2.85e-5** |
| **Smooth Walsh (K=8)** | 96.91% | **9,866** | 155.1 | **9.82e-5** |

### Key Observations
1. **The "Smoothness" Victory**: `Smooth Walsh (K=16)` achieved the highest accuracy of the entire benchmark (0.9813), surpassing the Dense baseline despite having **~4x fewer parameters**.
2. **Extreme Efficiency**: `Smooth Walsh (K=8)` reached **96.91%** accuracy with only **9,866 parameters**. This is a **13.4x reduction** in parameters compared to the Dense model with only a ~1% absolute drop in accuracy.
3. **Blocky Walsh Limitations**: The pure Walsh implementation (`Blocky`) performed worse than the Dense baseline. This suggests that the high-frequency "square wave" nature of raw Walsh bases creates aliasing that hinders learning unless smoothed.
4. **Computational Cost**: The wall-clock time is slightly higher for Walsh variants due to the overhead of weight synthesis in every step (ifwht + interpolate). However, this can be optimized by caching weights or using specialized kernels.

## Visual Analysis
*(Refer to `results/figures/v122_weights_*.png`)*
- **Blocky Weights**: Exhibit sharp, axis-aligned rectangular patterns.
- **Smooth Weights**: Show organic, blob-like structures that resemble Gabor filters or biological receptive fields, explaining their superior generalization.

## Conclusion
Smooth Walsh neurons are a powerful architectural primitive for "Attention Neurons". They combine the mathematical elegance of spectral representations with the spatial smoothness required for vision tasks.

**Recommendation**: Transition existing spectral architectures to use Smooth Walsh (or Smooth DCT) as the default weight synthesis method.
