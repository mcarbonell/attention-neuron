# Attention Neuron: Intelligence is Learning Where to Look

![Status](https://img.shields.io/badge/Status-Research--Prototype-blue)
![Architecture](https://img.shields.io/badge/Architecture-Spectral%20%2F%20Geometric%20%2F%20Complex-orange)
![Efficiency](https://img.shields.io/badge/Efficiency-up%20to%2099.9%25%20Parameter%20Reduction-green)

> **"Attention is All You Need" — Vaswani et al., 2017**
>
> **This repository asks: What if attention isn't just for Transformers? What if every neuron learns its own attention mask over a fixed substrate?**

---

## 🎯 Core Thesis: Learning *Is* Attention

Standard deep learning treats learning as **weight sculpting**: carve millions of individual parameters $w_{ij}$ until the mapping input→output works.

**This repository explores the opposite thesis:**

> **The intelligence of a neural network does not reside in the values of its weights, but in the attention masks it learns over fixed, structured substrates. Learning is not sculpting—it is *learning where to look*.**

| Paradigm | Standard DL | Attention Neuron |
|----------|-------------|------------------|
| **Unit of learning** | Individual weights $w_{ij}$ | Attention masks over substrates |
| **Substrate** | Learned (random init → optimized) | **Fixed, structured, infinite** (DCT, Walsh, Random, Geometric, Complex) |
| **Parameters** | $O(d^2)$ per layer | $O(k^2)$ or $O(r \cdot d)$ ≪ $d^2$ |
| **Intelligence** | Memorizing mappings | **Learning *what to attend to*** |
| **Analogy** | Sculpting marble | **Pointing a flashlight** |

---

## 🧠 The Attention Neuron Framework

Every layer in this library follows a unified pattern:

```
Fixed Substrate (Dictionary)  ×  Learnable Attention Mask  →  Effective Weight Matrix
```

| Layer | Substrate (Prior) | Attention Mechanism | Budget |
|-------|-------------------|---------------------|--------|
| **`AttentionLinear`** | Frozen random projection (Kaiming) | Dual low-rank: multiplicative gate + additive shift | rank $r$ |
| **`RosettaLinear`** | $K$ independent frozen random substrates | Softmax mixing over $K$ substrates per neuron | $K$ substrates |
| **`DCTLinear`** | Discrete Cosine Transform basis | Spectral coefficients (frequency attention) | $k_{in} \times k_{out}$ |
| **`WalshLinear`** | Walsh-Hadamard basis | Spectral coefficients (logical/parity attention) | $k_{in} \times k_{out}$ |

**The substrate encodes *inductive bias* (what patterns exist). The attention mask encodes *intelligence* (which patterns matter for this task).**

---

## 🔬 Why This Matters

1. **Compression without loss**: 50–1000× fewer trainable params, competitive accuracy
2. **Interpretability**: Attention masks are directly visualizable (heatmaps of frequencies, substrate mixtures, gate activations)
3. **Hardware-friendly**: Walsh = additions only; DCT = frequency-domain sparsity; Geometric = resolution-invariant
4. **Scalability**: Grow substrate resolution (more frequencies, more substrates) without growing parameters—only attention resolution grows
5. **Unified view**: Transformers attend over *tokens*; Attention Neurons attend over *bases*. Same mechanism, different granularity.

---

## 🚀 Six Research Eras (287+ Experiments)

### 1. Multiplicative Gating & Phase Bias (v1–v18)
**Insight**: Gating frozen backbones >> additive tuning (LoRA).  
**Result**: 94.53% MNIST with **7.8k params** (98% compression). Phase bias $\sin(\theta)$ for neuromorphic $[-1,1]$ signals.  
📄 [Phase 1](docs/findings_phase1_consolidation.md) | [Phase 2](docs/findings_phase2_comparison.md)

### 2. Substrate Interference & Spatial Priors (v19–v33)
**Rosetta (v22)**: Attention-like mixing of $K$ substrates via softmax — constructive interference cancels useless frequencies.  
**Perlin substrates**: Correlated spatial frequencies act as natural edge detectors, accelerating learning.  
📄 [Rosetta](docs/findings_v22_rosetta_cifar.md) | [Perlin](docs/findings_v26_perlin_cifar.md)

### 3. Spectral Orthogonal Bases (v35–v67)
**DCT**: Smooth semantic attention (cosine waves) → vision, language.  
**Walsh-Hadamard**: Discrete logic gates ($\pm1$ square waves) → multiplier-free FPGA execution.  
📄 [All-DCT MLP](docs/findings_v63_all_dct_mlp.md) | [Fully-JPEG LLM](docs/findings_v66_fully_jpeg_llm.md)

### 4. Geometric Stroke & Matchstick Neurons (v50–v57)
Neurons learn **mathematical curves** (Bézier, line segments) on continuous plane.  
Backprop physically moves curve endpoints → 100% resolution-invariant, adversarially robust.  
📄 [Stroke](docs/findings_v50_stroke_neurons.md) | [Matchstick](docs/findings_v51_matchstick_neurons.md)

### 5. Gated Ternary & Control Theory (v251–v274)
**Ternary $\{-1,0,1\}$ weights**: Zero-mean signals, native contrast/edge detection. **394 params → 85% MNIST** (CNN prior).  
**PID Optimizer**: Control theory (Kp=1, Ki=100, Kd=1) beats Adam in convergence & speed.  
📄 [Gated Frozen Master](docs/MASTER_SUMMARY_GATED_FROZEN_NETWORKS.md) | [PID Sweep](docs/findings_v267_pid_sweep.md)

### 6. Complex Phase, Hyperbolic & Conformal Optics (v275–v287)
**Complex-valued (CVNN)**: Native 2D rotations, wave interference.  
**TrueCausalComplexFFT**: Causal Fourier mixer — position encoded in phase $e^{i\varphi}$, no positional encodings needed.  
**Phase-nGPT**: nGPT hyperspherical norm + causal phase mixer + NarrowFFN → **80% fewer params, 2.5× speedup** vs Transformer.  
**Poincaré Attention**: Geodesic distances in hyperbolic disk for hierarchical reasoning.  
**Conformal Optics**: Weight matrix = projection of 2D texture deformed by trainable conformal map.  
📄 [Phase-nGPT](docs/findings_v282_ultimate_phase_ngpt.md) | [Poincaré](docs/findings_v286_poincare_attention.md) | [Conformal](docs/findings_v287_conformal_optics.md)

### 7. Complex Delta Phase Holographic State Memory (v300–v301)
**Holographic $O(N)$ Recurrence**: Keys and Queries projected to the complex $U(1)$ unit circle $K = e^{i\theta_k}, Q = e^{i\theta_q}$.  
**Capacity Scaling Law**: Iso-state-memory (iso-floats) benchmarking against Real DeltaNet. Under capacity pressure (64 pairs, $L=512$), Real DeltaNet collapses to **51.69%**, while Complex Delta Phase sustains **99.11% associative recall**.  
**Decisive Rectangular Hypothesis**: Proves intrinsic $U(1)$ phase interference cancellation across square and rectangular state allocations.  
📄 [V300 Capacity Scaling](scratch/prototype_v300_capacity_scaling.py) | [V301 Decisive Benchmark](scratch/prototype_v301_decisive_benchmark.py)

---

## 📊 Key Benchmarks

### MNIST
| Model | Version | Trainable Params | Accuracy | Compression |
|-------|---------|------------------|----------|-------------|
| Standard MLP (Baseline) | — | ~400,000 | 94.50% | 1.0× |
| **Gated Random** | v6b | **7,794** | 94.53% | **51×** |
| **Nano Walsh (Spectral)** | v40 | **938** | 92.12% | **426×** |
| **Bézier Stroke (Geometric)** | v50 | **3,200** | 97.88% | **125×** |
| **All-DCT MLP** | v63 | **11,914** | 97.59% | **50×** |
| **Gated Ternary CNN** | v256 | **394** | 85.07% | **1015×** |
| **Conformal Optics** | v287 | **3,082** | 39.06% | **32×** |

### CIFAR-10, Associative Recall & Language
| Model / Task | Version | Trainable Params | Metric | Advantage |
|--------------|---------|------------------|--------|-----------|
| **NavigatorNet (CIFAR-10)** | v19 | 118,238 | 76.76% Acc | Rank-32 gates over frozen kernels |
| **Complex FFN** | v275 | ~800 | 2.63e-6 MSE | 6.1× lower loss vs real MLP |
| **CausalPhase-nGPT (WikiText)** | v282 | 116,870 | 5.35 PPL | **80% fewer params, 2.3× faster** |
| **Poincaré Attention (Tree Ancestors)** | v286 | 32,797 | **43.49%** Acc | +11.6% vs Euclidean at $d=64$ |
| **Complex Delta Phase (MQAR 64 pairs)** | v300 | ~14,000 | **99.11%** Acc | **+47.42% vs Real DeltaNet (51.69%)** under iso-floats |

---

## 🛠️ Library Usage

```bash
pip install -e .
```

```python
import torch
from attention_neuron import (
    AttentionLinear,   # Spatial: dual low-rank gating over frozen random
    RosettaLinear,     # Spatial: softmax mixture of K frozen substrates
    DCTLinear,         # Spectral: DCT coefficient attention (smooth/semantic)
    WalshLinear        # Spectral: Walsh coefficient attention (logic/discrete)
)

# 1. Spatial Gating — learn WHERE to gate in a random projection
layer = AttentionLinear(in_features=784, out_features=2048, rank=128)
# Params: 784*128*2 + 128*2048*2 ≈ 362k vs 1.6M dense

# 2. Rosetta — learn WHICH substrate to attend to per neuron
rosetta = RosettaLinear(in_features=784, out_features=1024, num_substrates=4)
# Each output neuron learns softmax over 4 frozen random dictionaries

# 3. DCT — learn WHICH FREQUENCIES matter (semantic attention)
dct = DCTLinear(in_features=256, out_features=256, k_in=64, k_out=64)
# Only 64×64=4096 spectral coeffs vs 65k dense — learns frequency mask

# 4. Walsh — learn LOGICAL PARITY patterns (discrete attention)
walsh = WalshLinear(in_features=512, out_features=512, k_in=128, k_out=128)
# Basis is ±1 → synthesis uses only additions, zero multiplications

x = torch.randn(32, 784)
out = layer(x)        # [32, 2048]
out = rosetta(x)      # [32, 1024]
out = dct(x)          # [32, 256]
out = walsh(x)        # [32, 512]
```

---

## 📂 Repository Structure

```
attention_neuron/        # Core library (production-ready layers)
├── layers/
│   ├── base.py          # Shared bases (DCT, Walsh matrices)
│   ├── dense.py         # AttentionLinear (dual low-rank gating)
│   ├── rosetta.py       # RosettaLinear (substrate mixture attention)
│   └── spectral.py      # DCTLinear, WalshLinear (spectral attention)
├── __init__.py          # Public API

scratch/                 # 287+ experimental prototypes (v1–v287+)
docs/                    # Findings, whitepapers, blueprints, brainstorming
examples/                # Runnable demos (MNIST, Spectral GPT)
results/                 # Figures, raw logs, model checkpoints
```

---

## 📖 Key Documents for Deep Dives

| Topic | Document |
|-------|----------|
| **Unified Theory** | [Attention Neuron Theory v2](docs/attention_neuron_theory_v2.md) |
| **Spectral vs LoRA** | [Attention Neuron vs LoRA](docs/attention_neuron_vs_lora.md) |
| **Compression Analysis** | [Attention Neuron Compression](docs/attention_neuron_compression.md) |
| **Phase-nGPT Blueprint** | [Blueprint DCT LLM](docs/BLUEPRINT_DCT_LLM.md) |
| **Conformal Optics** | [Findings v287](docs/findings_v287_conformal_optics.md) |
| **All Findings Index** | `docs/findings_v*.md` (chronological) |

---

## 📜 License

MIT License — see [LICENSE](LICENSE).

---

**Developed by Mario Raúl Carbonell Martínez**  
*Exploring the thesis that intelligence, at every scale, is learning where to direct attention.*