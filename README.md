# Attention Neuron: Dual Phase Factorization & Zeroth-Order Optimization

![Status](https://img.shields.io/badge/Status-Research--Prototype-blue)
![Architecture](https://img.shields.io/badge/Architecture-Rank--k%20Factorization-orange)
![Optimization](https://img.shields.io/badge/Optimizer-DGE%20(Zeroth--Order)-green)

**Attention Neuron** is a high-efficiency neural architecture designed for neuromorphic hardware and memory-constrained environments. It rethinks the fundamental weight matrix as a dynamic, low-rank modulated system, enabling training without backpropagation through the **Denoised Gradient Estimation (DGE)** algorithm.

## 🚀 Key Innovations

- **Dual Phase Factorization (Rank-k)**: Instead of learning individual weights, the network learns low-rank modulation factors. This achieves a **~98% reduction in parameters** compared to standard dense layers while maintaining competitive expressivity.
- **Zeroth-Order Optimization (DGE)**: Trained entirely without analytical gradients. It uses structural perturbations to estimate gradients, making it compatible with non-differentiable hardware (analog, photonic, quantized).
- **Stochastic Structural Masking**: High resilience to structural noise. Validated with **50% connection masking** per step, proving its robustness for unreliable or noisy neuromorphic circuits.
- **Adaptive Signal-to-Noise Ratio (SNR)**: Implements an **Incremental Batch Size** heuristic that automatically doubles the batch size upon learning stagnation or regression, optimizing data efficiency by **50x** compared to static batching.

## 📊 Results (MNIST)

| Metric | Attention Neuron (v10e) | Standard MLP (Baseline) |
| :--- | :--- | :--- |
| **Trainable Params** | **~15,400** | ~400,000 |
| **Backpropagation** | **No (DGE)** | Yes (Adam) |
| **Accuracy** | **88.8%** | 94.5% |
| **Data Efficiency** | **Incremental Batching** | Standard |

*Note: Results achieved on CPU-only training with heavy stochastic noise.*

## 🧠 The Philosophy: "Learning from the Ground Tremors"

Inspired by **Seismic Descent** and **Limited Discrepancy Search**, the architecture treats the initial random weights as a frozen "dictionary" of potential features. The "intelligence" resides in the learned modulation vectors that activate and phase-shift these connections. 

By removing the "Memory Tax" of backpropagation (no activation buffers, no gradients), this system allows for training large-scale models (LLMs) on consumer-grade hardware with context windows that are usually impossible for standard architectures.

## 🛠️ Project Structure

- `scratch/`: Experimental versions (from v1 baseline to v10e SOTA).
- `docs/`: Technical whitepapers and experiment findings.
- `results/`: Raw JSON metrics and statistical summaries.

## 📜 Acknowledgments

Developed by **Mario Raúl Carbonell Martínez**. This project is a direct evolution of the [DGE Optimizer](https://github.com/mcarbonell/dge-optimizer) framework.

---
**License:** MIT
