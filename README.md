# Attention Neuron: Resonance Intelligence & Spectral Architectures

![Status](https://img.shields.io/badge/Status-Research--Prototype-blue)
![Architecture](https://img.shields.io/badge/Architecture-Spectral%20%2F%20Conformal%20%2F%20Hyperbolic-orange)
![Efficiency](https://img.shields.io/badge/Efficiency-up%20to%2098%25%20Parameter%20Reduction-green)

**Attention Neuron** is a neural network design framework that challenges the traditional paradigm of weight optimization. Instead of treating learning as an act of sculpting millions of individual weights, this repository explores the thesis that:

> **The intelligence of a neural network does not reside in the specific values of its weights, but in the sintonization of a spectrum of frequencies over fixed orthogonal or geometric substrates. Learning is equalization, not sculpting.**

By shifting the unit of learning from individual weights $w_{ij}$ to low-rank gating, spectral projections, complex-valued phase modulations, and conformal/hyperbolic maps, we can construct networks that achieve competitive accuracy with a fraction of the trainable parameters.

---

## 🧠 Core Philosophy: "Equalize, Don't Sculpt"

Most neural networks treat learning like marble sculpting: starting with a block of random parameters and slowly carving them into shape, one weight at a time. This requires massive memory, floating-point operations, and gradients for millions of parameters.

Attention Neuron proposes the opposite. It treats the weight matrix as a fixed, structured substrate (a deterministic dictionary or spectral space) and learns only **how to read from and modulate it**. 

Depending on the layer, the weights are synthesized on-the-fly using low-dimensional control spaces:
* **In the Spatial Domain**: Through asymmetric input-output multiplicative gating ($W_{init} \odot (\delta_{in} \otimes \delta_{out})$).
* **In the Frequency Domain**: Through a small set of trainable coefficients over deterministic bases like the Discrete Cosine Transform (DCT) or Walsh-Hadamard Transform (WHT).
* **In the Complex Domain**: Through analytic phase modulations ($e^{i\varphi}$) that represent spatial patterns and positions directly.
* **In Geodesic/Conformal Domains**: Through conformal deformations of complex textures or mappings in non-Euclidean Poincaré disks.

---

## 🚀 The 6 Research Eras of Attention Neuron

This repository documents an evolutionary arc of **287 experimental iterations** divided into six key eras:

### 1. Multiplicative Gating & Phase Bias (v1–v18)
* **Insight**: Multiplicative gating over frozen random backbones is vastly superior to additive adjustments (LoRA-style from scratch). While additive tuning failed (~42.6% accuracy), gating achieved **94.53%** with only **7.8k trainable parameters** (98% compression) on MNIST.
* **Phase Bias**: Replaced unbounded linear bias with $\sin(\theta_{bias})$ to constrain signals within $[-1, 1]$, making it highly suitable for analog neuromorphic hardware.
* *Findings*: [Consolidation Phase 1](docs/findings_phase1_consolidation.md) | [Comparison Phase 2](docs/findings_phase2_comparison.md)

### 2. Substrate Interference & Spatial Priors (v19–v33)
* **Rosetta (v22)**: Fused multiple random substrates via attention-like mixing, showing that the network performs synthesis by constructive phase interference (canceling useless frequencies).
* **Structured Priors**: Replacing white noise with Perlin noise (correlated spatial frequencies) dramatically accelerated learning, proving that structural priors in frozen substrates act as natural edge-detectors.
* *Findings*: [Rosetta Insights](docs/findings_v22_rosetta_cifar.md) | [Perlin Noise Impacts](docs/findings_v26_perlin_cifar.md)

### 3. Spectral Orthogonal Bases (v35–v67)
* **The Transition**: Instead of random substrates, we introduced deterministic, complete, and orthogonal bases.
* **DCT vs. Walsh**:
  * **Discrete Cosine Transform (DCT)**: Best for smooth, semantic attention mechanisms (continuous cosine waves).
  * **Walsh-Hadamard (WHT)**: Best for feed-forward layers (discrete square waves of $\pm 1$). Because WHT has no multiplications, FFNs can be synthesized entirely with additions/subtractions—perfect for multiplier-free FPGA execution.
* *Findings*: [All-DCT MLP](docs/findings_v63_all_dct_mlp.md) | [Fully-JPEG LLM](docs/findings_v66_fully_jpeg_llm.md) | [DCT-LLM Blueprint](docs/BLUEPRINT_DCT_LLM.md)

### 4. Geometric Stroke & Matchstick Neurons (v50–v57)
* **Insight**: Neurons do not need to look at pixel arrays. Instead, they learn mathematical curves (quadratic Béziers or straight segments/matchsticks). The backpropagation algorithm physically moves the endpoints of these geometric shapes on a continuous plane, making them 100% resolution-invariant and robust to pixel-level adversarial perturbations.
* *Findings*: [Stroke Neurons](docs/findings_v50_stroke_neurons.md) | [Matchstick Neurons](docs/findings_v51_matchstick_neurons.md)

### 5. Gated Ternary & Control Theory (v251–v274)
* **Ternary Inhibition**: Discovered that weights restricted to $\{-1, 0, 1\}$ are critical to maintain zero-mean signals and enable contrast (edge detection). Gated Ternary CNNs achieved **85.07% accuracy on MNIST with only 394 parameters**.
* **PID Optimizer (v261)**: Replaced statistical optimizers like Adam with a control-theory-based PID controller (`Kp=1, Ki=100, Kd=1`). PID surpassed Adam in both late-stage convergence and speed.
* *Findings*: [Gated Frozen Master Summary](docs/MASTER_SUMMARY_GATED_FROZEN_NETWORKS.md) | [PID Optimizer Sweep](docs/findings_v267_pid_sweep.md)

### 6. Complex Phase, Hyperbolic & Conformal Optics (v275–v287)
* **Complex-Valued Networks (CVNN)**: Employed complex parameters to natively compute 2D rotations and wave interferences.
* **TrueCausalComplexFFT (v281)**: Replaced self-attention with a causal Fourier mixer. It encodes sequence position directly into complex phase shifts ($e^{i\varphi}$), eliminating the need for explicit positional encodings (PE).
* **Ultimate Phase-nGPT (v282)**: Unified nGPT hyper-spherical normalization, causal phase mixers, and NarrowFFN, achieving a **80% parameter reduction** and **2.5x speedup** compared to standard Transformers.
* **Poincaré Attention (v286)**: Applied geodetic distances in Poincaré hyperbolic disks to represent hierarchical tree relationships, outperforming Euclidean attention.
* **Conformal Optics (v287)**: Formulated weight matrices as the projection of a 2D complex texture deformed by a trainable conformal map (complex polynomial function).
* *Findings*: [Phase-nGPT](docs/findings_v282_ultimate_phase_ngpt.md) | [Poincaré Attention](docs/findings_v286_poincare_attention.md) | [Conformal Optics](docs/findings_v287_conformal_optics.md)

---

## 📊 Summary of Experimental Benchmarks

Below is a consolidated summary of key model checkpoints on MNIST and CIFAR-10:

### MNIST Benchmarks
| Model Class | Version | Trainable Params | Total Params | Accuracy | Compression vs. Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard MLP (Baseline)** | - | ~400,000 | ~400,000 | 94.50% | 1.0x |
| **Gated Random (v6b)** | v6b | **7,794** | 400,000 | 94.53% | **51.3x** |
| **High-Rank Random (v18)** | v18 | 1,259,806 | 3,100,000 | **99.09%** | - |
| **Nano Walsh (Spectral)** | v40 | **938** | 4,096 | 92.12% | **426x** |
| **Bézier Stroke (Geometric)**| v50 | **3,200** | Continuous | 97.88% | **125x** |
| **All-DCT MLP (Spectral)** | v63 | **11,914** | 600,000 | 97.59% | **50.3x** |
| **Gated Ternary CNN** | v256 | **394** | 16,384 | 85.07% | **1015x** (CNN prior) |
| **Conformal Optics** | v287 | **3,082** | 100,000 | 39.06% | **32.4x** (Complex shadow) |

### CIFAR-10 & Language Benchmarks
| Model Class / Task | Version | Trainable Params | Metric | Key Advantage |
| :--- | :---: | :---: | :---: | :--- |
| **NavigatorNet (CIFAR-10)** | v19 | **118,238** | 76.76% Acc | Rank-32 gates over frozen random kernels |
| **Complex FFN (Interference)** | v275 | **~800** | **2.63e-6** MSE | 6.1x lower loss than real-valued MLP |
| **CausalPhase-nGPT (WikiText)**| v282 | **116,870** | 5.35 PPL | **80% fewer parameters** and **2.3x faster** than Standard Transformer (4.77 PPL) |
| **Poincaré Attention (Tree Ancestors)**| v286 | **32,797** | **43.49%** Acc | Systematically beats Euclidean attention (+11.6% at $d=64$) |

---

## 🛠️ Library Usage

The core, production-ready modules are packaged under `attention_neuron/`. You can use them directly in PyTorch networks.

### 1. Spatial Gating Layers
`AttentionLinear` applies low-rank gating over a frozen random projection:

```python
import torch
from attention_neuron import AttentionLinear, RosettaLinear

# Input: 784, Output: 2048, Rank-128 Gating Modulator
# Reduces trainable parameters from 1.6M to 362K
layer = AttentionLinear(in_features=784, out_features=2048, rank=128)

x = torch.randn(32, 784)
output = layer(x)
print(output.shape) # torch.Size([32, 2048])
```

`RosettaLinear` mixes multiple random substrates to find optimal features:

```python
# Fuses 4 independent frozen substrates dynamically
rosetta_layer = RosettaLinear(in_features=784, out_features=1024, num_substrates=4)
output_rosetta = rosetta_layer(x)
```

### 2. Spectral Layers (Walsh & DCT)
Used for training directly in frequency space:

```python
from attention_neuron import DCTLinear, WalshLinear

# DCT layer for smooth, semantic modulation (k represents the number of kept frequencies)
dct_layer = DCTLinear(in_features=256, out_features=256, k_in=64, k_out=64)

# Walsh-Hadamard layer for discrete logic gates (requires no multiplications in basis mapping)
walsh_layer = WalshLinear(in_features=512, out_features=512, k_in=128, k_out=128)
```

---

## 📂 Repository Structure

* `attention_neuron/`: The core library with clean, tested layers (`layers/dense.py`, `layers/spectral.py`, `layers/rosetta.py`).
* `scratch/`: The evolutionary workspace containing all 400+ script variations and 287 experimental prototypes (e.g., `prototype_v287_conformal_optics.py`, `prototype_v282_ultimate_phase_ngpt.py`, etc.).
* `docs/`: In-depth analytical documentation, whitepapers, brainstorming files, and experimental findings (`docs/findings_v*.md`).
* `examples/`: Demos demonstrating fast training with small parametrizations.
* `results/`: Visual figures (e.g., Poincaré disk layouts, MNIST delta curves) and raw execution data.

---

## 📜 License
This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.

Developed by **Mario Raúl Carbonell Martínez**.
