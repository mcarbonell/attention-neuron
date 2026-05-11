# Findings V258: Spectrum-Gated Transformer (SGT)

## Overview
We built a "Vision-like" Transformer where the Self-Attention mechanism is replaced by a **Fast Walsh-Hadamard Transform (FWHT)**. All weights are **Frozen Ternary** and learning happens via **Float Gating**.

## Results (10 Epochs, MNIST)

| Metric | Value |
| :--- | :--- |
| **Peak Accuracy** | **71.20%** |
| **Learnable Params** | **1,546** |
| **Frozen Params** | 541,696 |
| **Mixing Method** | Spectral (Hadamard) |

## Key Technical Insights

### 1. The Power of Spectral Mixing
Even without learned attention, the model reached **71%** by simply mixing the 64 patches globally using a Hadamard matrix. This proves that "Attention" is often just a sophisticated way of doing **Global Information Routing**, which can be approximated by fixed spectral transforms.

### 2. Efficiency Paradox
Although the model has very few learnable parameters, its accuracy is lower than the Gated CNN (85%).
- **CNN**: Has a local spatial prior (convolution).
- **SGT**: Has a global spectral prior.
For a structured task like MNIST, the local prior of the CNN seems to be more "Lottery Ticket friendly" than the global mixing of the Transformer at this scale.

### 3. The "Vectorization or Death" Rule
Initial training took **1 hour per epoch** due to a recursive Python implementation of FWHT. Switching to a **precomputed Hadamard Matrix** multiplication reduced the time to **~150 seconds**. This is a critical lesson for spectral architectures: always favor matrix-ops over algorithmic recursion in high-level languages like Python.

## Conclusion
The Spectrum-Gated Transformer is a viable, ultra-lightweight architecture for global feature fusion. While it currently trails the CNN in accuracy for small-scale vision, its multiplication-free mixing makes it a prime candidate for extremely large sequence modeling where $O(N^2)$ attention is the bottleneck.

**Reference Script**: `scratch/prototype_v258_spectrum_transformer.py`
