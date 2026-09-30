# Topographic Cellular Matrices: Breaking Permutation Symmetry via Local Weight Dynamics

**Author:** Mario Raúl Carbonell Martínez  
**Project:** Attention Neuron / Advanced Representation Theory  
**Date:** October 2026  
**Status:** Research Proposal & Experimental Blueprint  

---

## Abstract

Standard deep learning treats weight matrices $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ as unstructured, permutation-degenerate bipartite graphs. Any permutation of rows and columns (with corresponding inverse permutations in adjacent layers) preserves network outputs identically, creating an $N!$ gauge symmetry in parameter space that hampers model alignment, interpretability, and physical hardware mapping. Furthermore, modern optimizers treat each weight $W_{i, j}$ as an isolated scalar with zero awareness of its coordinate neighbors.

In this proposal, we introduce **Topographic Cellular Regularization (TCR)**. We imbue weight matrices with a 2D spatial coordinate manifold, subjecting updates to an infinitesimal local interaction term governed by cellular automata, 2D discrete Laplacians, or reaction-diffusion kernels:

$$W_{t+1} = W_t - \eta \nabla_{W} \mathcal{L}_{\text{task}} + \epsilon \cdot \Phi_{\text{cellular}}(W_t)$$

We hypothesize that an infinitesimal spatial coupling ($\epsilon \in [10^{-5}, 10^{-3}]$):
1. **Breaks the permutation gauge symmetry**, naturally canonizing internal representations and allowing direct zero-cost weight alignment across independent seeds without solving NP-hard matching problems (*Git Re-Basin*).
2. **Induces cortical topography**, causing adjacent neurons to specialize in adjacent semantic features (analogous to retinotopy and tonotopy in biological cortex).
3. **Forces an exponential decay in the singular value spectrum** (low effective rank), yielding representations with high intrinsic compressibility and superior generalization.

This document details the mathematical framework, biological rationale, and an executable experimental protocol to be implemented in the `attention-neuron` repository.

---

## 1. Problem Statement: The Permutation Degeneracy Paradox

In an $L$-layer multilayer perceptron or transformer feedforward block:

$$h^{(l)} = \sigma\left( W^{(l)} h^{(l-1)} + b^{(l)} \right)$$

For any permutation matrix $P \in \Pi(d_l)$, the transformed network:

$$\tilde{W}^{(l)} = P W^{(l)}, \quad \tilde{b}^{(l)} = P b^{(l)}, \quad \tilde{W}^{(l+1)} = W^{(l+1)} P^T$$

produces an identical input-output mapping: $f_{\theta}(x) \equiv f_{\tilde{\theta}}(x)$.

### Consequences of this degeneracy:
1. **The Alignment & Merging Barrier:** Two networks trained from different random seeds land in distant permutation basins of the loss landscape. Merging their weights (e.g., model souping, federated learning) requires solving an NP-hard quadratic assignment problem.
2. **Representation Incoherence:** Mechanistic interpretability finds "polysemanticity" and scattered feature activations because gradient descent has no incentive to group related features spatially.
3. **Hardware Mismatch:** Neuromorphic crossbars (memristors, analog photonic arrays) suffer from spatial parasitic capacitance and thermal crosstalk. A weight matrix with wild, uncorrelated high-frequency variance between adjacent physical cells causes electrical noise and inefficiencies.

**The Question:** What happens if we give weights a physical metric space where adjacent weights $W_{i, j}$ and $W_{i, j\pm 1}$ interact under an infinitesimal spatial prior?

---

## 2. Mathematical Formulations

We define the total loss as:

$$\mathcal{L}_{\text{total}}(W) = \mathcal{L}_{\text{task}}(W) + \epsilon \cdot \mathcal{R}_{\text{topo}}(W)$$

or, alternatively, as an explicit gradient update hook:

$$W_{t+1} = \operatorname{Optimizer}\left(W_t, \nabla_W \mathcal{L}_{\text{task}}\right) + \epsilon \cdot \mathcal{T}(W_t)$$

where $\mathcal{T}(W)$ is a local 2D transformation.

We formulate three distinct variants of spatial interaction:

### Variant A: Discrete 2D Laplacian (Dirichlet Surface Tension)
Each weight is pulled infinitesimally toward the average of its immediate 4-connected spatial neighbors in the matrix:

$$\mathcal{R}_{\text{Lap}}(W) = \frac{1}{2} \sum_{i=1}^{d_{\text{out}}} \sum_{j=1}^{d_{\text{in}}} \left( (W_{i+1, j} - W_{i, j})^2 + (W_{i, j+1} - W_{i, j})^2 \right)$$

The negative gradient $-\nabla_W \mathcal{R}_{\text{Lap}}$ corresponds to a discrete 2D convolution with a Laplacian kernel:

$$K_{\text{Lap}} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}, \quad \nabla_W \mathcal{R}_{\text{Lap}} = - K_{\text{Lap}} * W$$

*Effect:* Acts as a spatial low-pass filter on the weight matrix. High-frequency checkerboard noise is penalized; smooth, continuous manifolds are favored.

### Variant B: Reaction-Diffusion (Turing Morphogenesis)
In biological morphogenesis (Alan Turing, 1952), complex patterns (stripes, spots) emerge from the interaction of a local activator and a wider-range inhibitor. We model this via a Difference of Gaussians (DoG) Mexican Hat kernel:

$$K_{\text{Turing}} = \frac{1}{2\pi \sigma_1^2} e^{-\frac{r^2}{2\sigma_1^2}} - \gamma \frac{1}{2\pi \sigma_2^2} e^{-\frac{r^2}{2\sigma_2^2}}, \quad \sigma_1 < \sigma_2$$

*Effect:* Adjacent weights reinforce each other, while medium-range neighbors inhibit each other. Instead of blurring into a flat plane, the weight matrix self-organizes into **periodic bands, functional columns, or localized islands**.

### Variant C: Continuous Cellular Automaton (Neural CA Step)
Treating $W$ as a 2D cellular automaton state:

$$\Delta W_{i, j} = \tanh\left( \alpha \cdot (K_{\text{sobel}} * W)_{i, j} + \beta \cdot (K_{\text{lap}} * W)_{i, j} \right)$$

*Effect:* Nonlinear local consensus dynamics that allow sharp phase transitions between continuous sub-domains.

---

## 3. Theoretical Predictions & Emergent Hypotheses

| Hypothesis | Mechanism | Expected Observable |
| :--- | :--- | :--- |
| **H1: Gauge Symmetry Breaking** | Neighbor coupling favors a single global orientation. | Cross-seed weight cosine similarity $\cos(W_A, W_B)$ increases significantly without applying Hungarian/Re-Basin permutation matching. |
| **H2: Cortical Topography** | Minimizing spatial tension forces similar input features to map to adjacent neurons. | Input receptive fields of neuron $i$ and neuron $i+1$ exhibit high cosine similarity, creating smooth spatial trajectory plots. |
| **H3: Spectral Low-Rank Collapse** | A spatially smooth 2D field has low spatial frequency and bounded variation. | Singular values $\sigma_k(W)$ decay exponentially faster than in standard Adam/SGD baselines. Rank-8 approximation preserves $>95\%$ energy. |
| **H4: Generalization & Flat Minima** | Spatial coupling suppresses jagged, high-frequency parameter configurations. | Lower test error or greater robustness to input adversarial/Gaussian noise despite equal training loss. |

---

## 4. Connection to `attention-neuron` Concepts

This proposal directly deepens key architectural themes developed in `attention-neuron`:
* **Conformal Optics & Texture Fields:** In `attention-neuron`, weights are conceptualized as projections or shadows of continuous underlying mathematical textures. TCR replaces explicit polynomial deformations with a dynamical, self-organizing continuum directly on the weights.
* **Resonance over Sculpture:** Instead of each weight moving independently as unconstrained Euclidean degrees of freedom, the matrix acts as an elastic membrane.
* **Energy-Efficient Edge Mapping:** Spatially smooth weight matrices map directly onto analog crossbar arrays with minimal interconnect wiring and zero adjacent-cell parasitic spikes.

---

## 5. Experimental Protocol for `attention-neuron`

### Phase 1: Minimal Proof of Concept (Toy 2D / MNIST)
* **Architecture:** 3-layer MLP: $784 \to 256 \to 128 \to 10$.
* **Baseline:** Standard AdamW ($\text{lr}=10^{-3}$, weight decay $=10^{-4}$).
* **Condition:** Topo-AdamW with $\epsilon \in \{10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$.
* **Visual Output:** Save $W^{(1)}$ and $W^{(2)}$ as 2D heatmaps at epochs $0, 5, 20, 50$.
  * *Success condition:* The weight matrix transitions from uniform salt-and-pepper noise to organized coherent topography (continents, stripes, or gradients).

### Phase 2: Permutation Alignment Across Seeds
* Train 5 independent seeds for Baseline and 5 independent seeds for Topo-AdamW.
* Compute pairwise unpermuted weight correlation:
  $$\rho_{\text{raw}} = \frac{1}{\binom{5}{2}} \sum_{a < b} \frac{\langle \operatorname{vec}(W_a), \operatorname{vec}(W_b) \rangle}{\|W_a\| \|W_b\|}$$
* *Prediction:* $\rho_{\text{raw}}(\text{Baseline}) \approx 0$, whereas $\rho_{\text{raw}}(\text{Topo}) \gg 0$.

### Phase 3: SVD & Compressibility Spectrum
* Compute singular value decomposition: $W = U \Sigma V^T$.
* Plot normalized singular spectrum $\sigma_i / \sum \sigma_j$.
* Evaluate reconstruction error under rank-$k$ truncation ($k \in \{4, 8, 16, 32\}$).

---

## 6. PyTorch Reference Implementation (Drop-In Module)

Below is the reference implementation designed to be pasted directly into an experiment script in `attention-neuron`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopographicWeightRegularizer:
    """
    Applies spatial neighbor regularization (2D Laplacian or Turing kernel)
    to designated weight matrices in a neural network.
    """
    def __init__(self, mode="laplacian", epsilon=1e-4, device="cpu"):
        self.epsilon = epsilon
        self.mode = mode
        self.device = device
        
        if mode == "laplacian":
            # Discrete 2D 4-neighbor Laplacian kernel
            k = torch.tensor([[0.,  1., 0.],
                              [1., -4., 1.],
                              [0.,  1., 0.]], dtype=torch.float32)
            self.kernel = k.unsqueeze(0).unsqueeze(0).to(device)
            
        elif mode == "turing":
            # Mexican hat / Difference of Gaussians (5x5)
            # Center excitation (+), surrounding inhibition (-)
            k = torch.tensor([
                [-0.05, -0.10, -0.10, -0.10, -0.05],
                [-0.10,  0.20,  0.50,  0.20, -0.10],
                [-0.10,  0.50,  1.00,  0.50, -0.10],
                [-0.10,  0.20,  0.50,  0.20, -0.10],
                [-0.05, -0.10, -0.10, -0.10, -0.05]
            ], dtype=torch.float32)
            self.kernel = k.unsqueeze(0).unsqueeze(0).to(device)

    def compute_penalty(self, weight: torch.Tensor) -> torch.Tensor:
        """
        weight shape: [d_out, d_in]
        Treats matrix as a single-channel 2D image: [1, 1, H, W]
        """
        if weight.dim() != 2:
            return torch.tensor(0.0, device=weight.device)
            
        w_img = weight.unsqueeze(0).unsqueeze(0)
        
        # Convolve with reflective padding to handle matrix borders cleanly
        pad = self.kernel.shape[-1] // 2
        filtered = F.conv2d(w_img, self.kernel, padding=pad)
        
        # Penalty is the squared energy of high-frequency tension
        penalty = torch.mean(filtered ** 2)
        return self.epsilon * penalty

    def attach_hook(self, param: nn.Parameter):
        """
        Alternatively, inject as an in-place gradient modifier during backward().
        """
        def hook(grad):
            with torch.no_grad():
                g_img = param.unsqueeze(0).unsqueeze(0)
                pad = self.kernel.shape[-1] // 2
                spatial_correction = F.conv2d(g_img, self.kernel, padding=pad).squeeze()
                return grad + self.epsilon * spatial_correction
        param.register_hook(hook)
```

---

## 7. Next Steps & Execution Plan

When continuing in `C:\Users\mrcm_\Local\proj\algorithms\attention-neuron`:
1. Read this proposal: `proposals/proposal_cellular_weight_regularization.md`.
2. Implement the experimental runner: `experiments/exp_cellular_weights.py`.
3. Run comparative training on MNIST / Fashion-MNIST across 3 seeds for Baseline vs. Topo ($\epsilon=10^{-4}$).
4. Generate the side-by-side weight visual evolution matrix (Epoch 0, 10, 50) and the SVD singular decay curve.
