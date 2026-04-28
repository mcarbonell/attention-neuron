# Attention Neuron: Low-Rank Neural Modulation

![Status](https://img.shields.io/badge/Status-Research--Prototype-blue)
![Architecture](https://img.shields.io/badge/Architecture-Rank--k%20Factorization-orange)

**Attention Neuron** is a neural architecture that rethinks the dense weight matrix as a dynamically modulated, low-rank system. Instead of learning millions of individual weights, the network learns a small set of neuron-centric modulation vectors that reconfigure a frozen random substrate.

## About

Most neural networks treat learning as an act of sculpture: you start with a block of parameters and slowly carve them into the right shape, one weight at a time. Attention Neuron proposes the opposite. It treats the weight matrix as a fixed, random dictionary — a static library of possible features — and learns only **how to read from it**.

In this framework, every neuron has a "personality": vectors that dictate how loudly it listens to the previous layer, and how loudly it speaks to the next. By crossing the outgoing voice of Neuron A with the incoming ear of Neuron B, each connection in the network receives a unique, asymmetric modulation. The network reconfigures massively using very few variables.

This shift in the unit of learning — from the individual weight to the neuron's modulation profile — has produced some surprising results. On MNIST, a network with **only 7,794 trainable parameters** (a 98% reduction from a standard MLP) reaches **94.53% accuracy**. With more rank, the same architecture climbs to **99.09%**. On CIFAR-10, a convolutional variant with just **118K trainable parameters** over frozen random kernels achieves **76.76%**. Most strikingly, the **Rosetta** experiments showed that a network can mix *multiple* random substrates simultaneously, learning to synthesize a feature base from pure noise.

The project is young — born from a series of rapid experiments over 48 hours — but the signal is strong. The question it poses is simple but deep: **how much of a neural network's intelligence lives in the specific values of its weights, and how much lives in the pattern of which weights it chooses to use?**

## The Core Idea

Traditional neural networks learn by adjusting every connection weight individually. Attention Neuron asks: **what if the weights are random and frozen, and we only learn how to modulate them?**

The weight matrix evolves through dual low-rank factorization:

```
W_evolved = W_init ⊙ (δ_in_m ⊗ δ_out_m) + (δ_in_a ⊗ δ_out_a)
```

Where:
- **W_init**: Random frozen weights (the "dictionary" of potential features)
- **δ_in/out_m**: Multiplicative modulation (gating: which cables to amplify or silence)
- **δ_in/out_a**: Additive modulation (shifting: breaking symmetries)

This reduces trainable parameters from **O(N × M)** to **O(N + M)** per layer.

---

## Key Innovations

- **Dual Phase Factorization (Rank-k)**: Achieves **~98% parameter reduction** compared to standard dense layers while maintaining competitive expressivity.
- **Frozen Random Substrate**: The base weight matrix is initialized randomly and never updated. On neuromorphic hardware, this could be burned into ROM.
- **Phase Bias (Safe By Design)**: Replaces unbounded linear bias with `sin(θ_bias)`, keeping activations strictly within `[-1, 1]` — ideal for quantization and analog hardware.
- **Multi-Substrate Mixing (Rosetta)**: Multiple random substrates can be mixed via learned attention, reducing dependence on any single initialization.

---

## Results

### MNIST

| Model | Trainable Params | Total Params | Accuracy |
|-------|-----------------|--------------|----------|
| Standard MLP (Baseline) | ~400,000 | ~400,000 | ~94.5% |
| Attention Neuron (v6b, rank-2) | **7,794** | ~400,000 | **94.53%** |
| Attention Neuron (v18, rank-128) | **1,259,806** | ~3,100,000 | **99.09%** |

### CIFAR-10

| Model | Trainable Params | Architecture | Accuracy |
|-------|-----------------|--------------|----------|
| NavigatorNet (v19, rank-32) | **118,238** | 6 Conv + 1 Linear | **76.76%** |
| Rosetta MLP (v22, 4 substrates) | **612,038** | 3-layer MLP | **56.72%** |
| Hybrid Frozen+Plastic (v23) | **2,452,490** | Rosetta sensor + plastic brain | **62.51%** |

*Note: v18-v19 trained with AdamW + OneCycleLR on CPU/GPU.*

---

## The Philosophy: "Learning to Access the Dictionary"

The architecture treats the initial random weights as a frozen dictionary of potential features. The "intelligence" resides in the learned modulation vectors that activate, amplify, or silence these connections. The same random substrate can represent radically different functions depending on how it is modulated.

This perspective connects to several lines of work:
- **Random features literature**: Random projections can be surprisingly expressive
- **LoRA / PEFT**: Low-rank adaptation of frozen pretrained weights
- **Neuromorphic computing**: Fixed physical connections with tunable modulation

---

---

## Getting Started

You can now use `attention_neuron` as a modular library for your projects.

### Installation

Clone the repository and add it to your python path, or simply copy the `attention_neuron/` folder into your project.

```python
from attention_neuron import AttentionLinear, RosettaLinear

# A standard Attention Neuron layer
# 784 -> 2048 using rank-128 modulation of frozen weights
layer = AttentionLinear(784, 2048, rank=128)

# Rosetta layer mixing 4 random substrates
rosetta = RosettaLinear(784, 2048, num_substrates=4)
```

### Spectral Layers

For advanced frequency-domain routing (DCT or Walsh-Hadamard):

```python
from attention_neuron import DCTLinear, WalshLinear

# DCT for smooth semantic attention
dct_layer = DCTLinear(128, 128, k_in=32, k_out=32)

# Walsh for sharp logical FFNs
walsh_layer = WalshLinear(512, 512, k_in=64, k_out=64)
```

---

## Project Structure

- `attention_neuron/`: The core library.
    - `layers/`: Implementation of Dense, Rosetta, and Spectral layers.
- `examples/`: Functional demos (MNIST Compact, Spectral GPT).
- `scratch/`: Experimental history (v1 through v67).
- `docs/`: Technical whitepapers and research findings.

---

## Roadmap

1. **Architecture Consolidation**: Confirm which components (multiplicative, additive, phase bias) are essential.
2. **Fair Baselines**: Compare against standard LoRA and additive low-rank methods on equal footing.
3. **Scaling Beyond CIFAR-10**: Test on CIFAR-100 or small-scale vision transformers.
4. **Transformer / LLM Adaptation**: Apply Attention Neuron layers to fine-tuning pretrained language models.
5. **Multi-Substrate Deep Dive**: The Rosetta finding (v22) suggests mixing substrates is powerful — explore this further.

---

## Acknowledgments

Developed by **Mario Raúl Carbonell Martínez**.

---
**License:** MIT
