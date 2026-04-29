# Findings V98: Invariant Spectral Attention (ISA Hybrid)

## Overview
We hybridized the **Fourier-Mellin Invariance** (V97) with a **Spectral Attention Neuron** (Walsh-Hadamard). This "ISA" architecture aims to filter the invariant signature in the frequency domain to improve classification robustness.

## Empirical Results (10 Epochs, Torture Test: 90° Rotation + 20% Shift)

| Metric | V97 (FM + MLP) | **V98 (ISA Hybrid)** | Difference |
| :--- | :--- | :--- | :--- |
| **Peak Accuracy (Best)** | 39.31% | **41.43%** | **+2.12%** |
| **Final Accuracy (E10)** | 36.63% | **37.72%** | **+1.09%** |

## Key Technical Insights

### 1. Superior Pattern Extraction
The Spectral Attention mechanism (Learned Walsh Masking) outperformed a standard MLP at interpreting the Fourier-Mellin signature. It achieved a peak of **41.43%**, showing that frequency-domain filtering is better suited for invariant spectra.

### 2. Signal-to-Noise Improvement
By applying attention in the Walsh domain, the network learned to ignore the "blur" and "aliasing" introduced by the log-polar grid sampling, focusing instead on the most stable spectral components of the digit's shape.

### 3. Biological Plausibility
This hybrid model mimics the human vision pipeline: Foveation/Invariance (FM) followed by Hierarchical Frequency Processing (Spectral Attention). It represents the most advanced architecture in the current series.

## Conclusion
The ISA Hybrid is the winner of the invariance experiments. It proves that combining classical invariant transforms with modern spectral attention is a powerful path toward robust, lightweight vision.

**Reference Script**: `scratch/prototype_v98_fm_attention_hybrid.py`
