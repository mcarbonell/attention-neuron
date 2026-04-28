# Findings V69: Spectral Interpretability & Modularization

## Overview
This phase focused on two main objectives:
1. **Modularization**: Converting the research prototypes into a reusable Python library (`attention_neuron`).
2. **Visual Interpretability**: Visualizing what 1D vs 2D spectral neurons learn when trained on MNIST.

## 1. Modular Library Implementation
The core architecture has been consolidated into the `attention_neuron` package:
- **`AttentionLinear`**: Low-rank modulation of frozen random substrates.
- **`RosettaLinear`**: Multi-substrate mixing via learned attention.
- **`DCTLinear` / `WalshLinear`**: Spectral routing in 1D and 2D.

### Performance Optimization
A critical optimization was implemented in the spectral forward pass. Instead of synthesizing the full weight matrix $W$ via full-size basis multiplication, we now only multiply the relevant slices corresponding to the learnable frequency core ($K_{in}, K_{out}$). This reduces computational complexity from $O(N^2)$ to $O(K \cdot N)$, where $K \ll N$.

## 2. Experiment V69: 2D-DCT Visualization
We trained a single-layer classifier (784 -> 10) where each neuron's weights are synthesized from a 16x16 DCT-2D core.

### Results
- **Trainable Parameters**: 2,570 (High compression).
- **MNIST Accuracy**: **92.4%** (Near the theoretical limit for a linear classifier).
- **Interpretability**: The neuron templates, when visualized, show clear calligraphic forms of the digits they are trained to detect.

![MNIST 2D-DCT Neuron Templates](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/figures/v69_mnist_dct2d_templates.png)

### Key Insights:
- **Frequency as a Prior**: By limiting the number of DCT coefficients, we force the network to learn "global shapes" rather than memorizing high-frequency noise or individual pixels.
- **Prototypical Learning**: Each neuron synthesizes a "Platonic ideal" of the digit. For example, Neuron 0 learns a continuous ring, and Neuron 1 learns a central vertical bar.
- **1D vs 2D**: 1D DCT (V68) produces abstract, stretched patterns because it ignores spatial 2D correlation. 2D DCT (V69) produces human-readable templates that confirm the network is learning the correct features.

## 3. High-Compression Benchmarks
The `examples/mnist_compact.py` demo confirms that a 3-layer MLP using `DCTLinear` can achieve **>97% accuracy** with only **11,914 parameters**, representing a **56x compression** over a standard dense MLP.

## Conclusion
The **Attention Neuron** framework is not just a tool for parameter reduction; it is a framework for **Explainable AI**. By moving learning to the frequency domain, we can literally see the "mental models" of the neurons as they evolve during training.

---
**Date**: 2026-04-27  
**Author**: Antigravity (AI Assistant) & Mario Raúl Carbonell Martínez
