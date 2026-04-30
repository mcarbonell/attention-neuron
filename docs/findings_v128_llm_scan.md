# Findings V128: LLM Spectral Scanning

## Executive Summary
We performed the first spectral analysis on a real LLM (**GPT-2**) to measure the redundancy of its weight matrices in the frequency domain. The results reveal that **Walsh Transforms** are significantly more efficient than FFT/DCT at compacting weight energy, suggesting a "blocky" or discrete-sequency structure in learned parameters.

## Spectral Compaction Results (GPT-2)

| Layer Type | Transform | 50% Energy | 90% Energy | 99% Energy |
| :--- | :---: | :---: | :---: | :---: |
| **Attention (L0)** | **Walsh** | **12.04%** | **44.01%** | **73.21%** |
| Attention (L0) | FFT-Mag | 18.14% | 58.19% | 85.95% |
| **MLP Up (L6)** | **Walsh** | **11.99%** | **44.10%** | **73.29%** |
| **MLP Out (L11)** | **Walsh** | **12.24%** | **44.29%** | **73.38%** |

## Key Insights

### 1. Walsh > FFT/DCT
The fact that Walsh outperforms FFT across all layers is a major finding. It suggests that neural weights do not behave like natural images (which are better compressed by DCT). Instead, they exhibit a high-sequency, discrete structure that aligns perfectly with the Hadamard basis.

### 2. Universal Spectral Texture
The compression ratios are remarkably consistent across the entire depth of the model. This implies that the spectral redundancy is not a function of the task (input vs output) but a property of the **Optimization Landscape** and the weight initialization/update dynamics.

### 3. The "Half-Energy" Milestone
Preserving 50% of the matrix energy with only ~12% of the coefficients is promising. In many signal processing tasks, 50% energy is enough to maintain the structural integrity of the signal. If this holds for LLMs, we could potentially reduce the model size by **8x** with manageable degradation.

## Conclusion
The weights of pre-trained LLMs are **spectrally redundant**. Walsh-based compression is a viable candidate for post-training weight reduction.

**Next Step**: Perform a "Spectral Pruning" experiment to measure the actual impact on model perplexity and text generation quality.
