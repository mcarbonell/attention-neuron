# Findings V59: DCT Attention Neurons

## Overview
Inspired by the **JPEG compression algorithm**, this experiment replaces dense linear weights with a **Differentiable DCT (Discrete Cosine Transform) Modulation** mechanism. Instead of learning 784 independent weights for a 28x28 input, each neuron learns a small $K \times K$ kernel of DCT coefficients.

## Methodology
- **Input**: MNIST digits (28x28).
- **Transform**: The input image is globally transformed to the DCT-II domain: $X_{dct} = D \cdot X \cdot D^T$.
- **Mechanism**: Each hidden neuron has a $K \times K$ learnable matrix $C$. The activation is computed as the dot product between $C$ and the top-left (low-frequency) $K \times K$ quadrant of $X_{dct}$.
- **Equivalence**: This is mathematically equivalent to a dense layer where weights are constrained to be a reconstruction from low-frequency DCT bases.
- **Parameters**: 
    - $K=8$ (64 parameters per neuron).
    - Compression Ratio: $784 / 64 = 12.25\times$ for the first layer weights.

## Results (MNIST)
- **Final Accuracy**: **98.12%** (Epoch 15).
- **Hidden Units**: 512.
- **Total Parameters**: ~39k (vs ~400k for a standard MLP of similar width).
- **Convergence**: Very fast (96% in the first epoch).

| Metric | Value |
| --- | --- |
| `final_objective` (Accuracy) | 0.9812 |
| `weight_compression` | 12.2x |
| `wall_clock_time` | 121.6s |

## Observations
1. **Low-Frequency Bias**: The model naturally focuses on the structural shapes of digits. By discarding high-frequency coefficients (the remaining 720 out of 784), we effectively regularize the model against pixel-level noise.
2. **JPEG Analogy**: Just as JPEG can represent an image with few coefficients, the "Attention Neuron" can represent a meaningful visual filter with very few learnable parameters in the frequency domain.
3. **Efficiency**: The computational overhead of the DCT transform is negligible ($O(N^3)$ or $O(N^2 \log N)$) compared to the savings in parameter updates and memory.

## Learned Patterns
The visualization in `v59_dct_gallery.png` shows that the neurons synthesize:
- Smooth vertical and horizontal gradients.
- Simple Gabor-like edges.
- Center-surround structures.
All these are achieved with only 64 coefficients per neuron.

## Conclusion
The DCT-Attention mechanism is one of the most parameter-efficient ways to implement a global receptive field. It provides a strong inductive bias for "smoothness" which is ideal for natural image processing.
