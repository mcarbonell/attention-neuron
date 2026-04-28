# Findings V84: Spectral Basis Comparison (DCT vs. Walsh-Hadamard)

## Overview
This experiment compared two different mathematical bases for image reconstruction within the Attention Neuron framework: **Discrete Cosine Transform (DCT)** and **Walsh-Hadamard Transform (WHT)**.

## Comparison Table

| Feature | DCT (Discrete Cosine) | WHT (Walsh-Hadamard) |
| :--- | :--- | :--- |
| **Basis Function** | Cosine Waves (Smooth) | Walsh Functions (Square Waves) |
| **Inductive Bias** | Smoothness, Natural gradients | Blocky, Discrete subdivisions |
| **Resolution Independence** | Excellent (Smooth interpolation) | "Neural Zoom" (Fractal-like blocks) |
| **Computational Cost** | Medium (Floating point mults) | Very Low (Additions/Subtractions only) |
| **Best For** | Images, Audio, Organic data | Digital logs, Hardware optimization |

## Key Insights
1. **Visual Quality**: DCT is significantly more efficient at capturing the essence of MNIST digits with few parameters ($K=8$). Walsh requires higher frequencies ($K=16$ or $32$) to suppress its blocky artifacts.
2. **Sequency Ordering**: For Walsh to be useful in image processing, it **must** be reordered by zero-crossings (sequency). Without this, the energy is scattered, and low-parameter reconstruction fails.
3. **Hardware Potential**: The FWHT is mathematically simpler than the DCT. In custom hardware or FPGAs, a Walsh-based Attention Neuron could run significantly faster and with less energy because it avoids the complexity of transcendental functions (cosines).
4. **Resolution Independence**: Both transforms allow for zero-padding in the frequency domain to increase output resolution. DCT produces a "vectorized" smooth look, while Walsh produces a "high-definition pixel art" look.

## Conclusion
While DCT is the clear winner for image fidelity, Walsh-Hadamard offers a fascinating alternative for scenarios where computational power is extremely limited or where the input data has a discrete, blocky structure.

**Visual Grid saved in**: `results/spectral_comparison_grid.png`
