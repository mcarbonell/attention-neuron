# Brainstorming: Cosine-Based Neurons and the Grokking Phenomenon

## Executive Summary
This document explores the theoretical foundation, mathematical mechanics, and practical implications of introducing **Cosine-Based Neurons** with learnable parameters ($A$ and $B$) into neural network architectures. Inspired by the internal representations discovered by networks during **grokking** on modular arithmetic tasks, this approach provides a direct periodic inductive bias that accelerates Fourier representation learning and mitigates delayed generalization.

---

## 1. Context & Motivation: The $\cos\left(\frac{8\pi}{113} x\right)$ Emergence

In mechanized interpretability studies of grokking (e.g., modular addition $a + b \pmod{113}$), standard MLPs and Transformers exhibit a distinct transition:
1. **Memorization Phase**: High overfitting on training data with zero validation accuracy.
2. **Grokking Transition**: Sudden jump to 100% validation accuracy after thousands of optimization steps.

During this transition, internal weights align into trigonometric representations such as:
$$\cos\left( \frac{8\pi}{113} x \right)$$

### Mathematical Significance
* **113**: The prime modulus of the cyclic group $\mathbb{Z}_{113}$.
* **$\frac{8\pi}{113} x$**: Equivalent to $\frac{2\pi \cdot k}{113} x$ for $k=4$. This represents the 4th harmonic frequency in the **Discrete Fourier Transform (DFT)** over $\mathbb{Z}_{113}$.
* Standard networks spend thousands of epochs learning to construct trigonometric functions out of linear combinations of ReLU/GELU activations.

---

## 2. Cosine Neuron Formulations

Instead of forcing networks to construct sinusoids through polynomial or piecewise-linear approximations, we define explicit cosine-activated neuron formulations:

### Formulation A: Learnable Frequency & Phase (1D)
$$f(x) = \cos(A \cdot x + B)$$
* **$A$ (Learnable Frequency/Scale)**: Controls the rate of periodic oscillation.
* **$B$ (Learnable Phase/Shift)**: Controls horizontal phase translation.

### Formulation B: Amplitude & Multivariable Frequency Vector
$$f(\mathbf{x}) = A \cdot \cos(\mathbf{W}^T \mathbf{x} + B)$$
* **$\mathbf{W}$**: Weight vector defining spatial frequency and orientation across input dimensions.
* **$B$**: Phase offset.
* **$A$**: Output amplitude scalar.

---

## 3. Trigonometric Angle-Addition Mechanics

Cosine-activated neurons naturally leverage trigonometric angle-addition identities to perform additions and subtractions in the frequency domain:

$$\cos(A x_1 + B_1) \cdot \cos(A x_2 + B_2) = \frac{1}{2} \left[ \cos\Big(A(x_1 + x_2) + (B_1 + B_2)\Big) + \cos\Big(A(x_1 - x_2) + (B_1 - B_2)\Big) \right]$$

When quadratic nonlinearities, elementwise multiplications, or attention dot-products interact with cosine features, the network computes modular addition and subtraction directly via frequency multiplication and phase shifts without needing intermediate layers.

---

## 4. Impact on Grokking Dynamics

| Aspect | Standard ReLU / GELU MLP | Cosine-Based Neuron Layer |
| :--- | :--- | :--- |
| **Inductive Bias** | Piecewise linear / continuous smooth | Periodic / Harmonic Fourier basis |
| **Fourier Feature Learning** | Slow; constructed over thousands of epochs | Immediate; tuned via direct gradient on $A, B$ |
| **Grokking Delay** | Significant delay (overfitting $\rightarrow$ sudden grokking) | Minimal to no delay (instant generalization) |
| **Parameter Efficiency** | High parameter count required for Fourier synthesis | Compact; 1-2 parameters per harmonic component |

---

## 5. Connections to Existing Literature

1. **SIREN (Sinusoidal Representation Networks - Sitzmann et al., 2020)**
   * Uses $\sin(\mathbf{W}\mathbf{x} + \mathbf{b})$ activation functions across all layers.
   * State-of-the-art for continuous implicit neural representations (audio, images, NeRFs, PINNs).
2. **Fourier Features (Tancik et al., 2020 / Rahimi & Recht, 2007)**
   * Maps inputs through fixed/learned $\cos(\mathbf{B}\mathbf{x})$ and $\sin(\mathbf{B}\mathbf{x})$ encodings prior to dense processing.
3. **Fourier KANs (Kolmogorov-Arnold Networks)**
   * Replaces spline activation edges with univariate Fourier series expansions: $f(x) = \sum \left( a_k \cos(kx) + b_k \sin(kx) \right)$.

---

## 6. Engineering Challenges & Trade-offs

1. **Jagged Loss Landscapes & Local Minima**
   * Derivative: $\frac{d}{dx}\cos(Ax+B) = -A \sin(Ax+B)$.
   * High frequencies ($A \gg 1$) cause rapidly oscillating gradients, leading to training instability or entrapment in local minima without specialized weight initialization (e.g., SIREN initialization schemes).
2. **Non-Periodic Generalization**
   * For monotonic, linear, or step-function targets outside periodic domains, cosine neurons require infinite series expansions, rendering them less efficient than standard ReLUs.

---

## 7. Next Steps & Proposed Experiments

1. **Synthetic Modular Arithmetic Benchmark**:
   * Train a minimal network with $f(x) = \cos(A x + B)$ on $a + b \pmod{p}$ vs a standard GELU MLP.
   * Plot validation accuracy vs epoch curves to measure grokking elimination.
2. **Phase & Frequency Dynamics Tracking**:
   * Track trajectory of $A$ and $B$ parameters during SGD optimization to verify direct convergence to discrete Fourier harmonics.
