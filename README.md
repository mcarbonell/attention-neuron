# Attention Neuron: Low-Rank Neural Modulation

![Status](https://img.shields.io/badge/Status-Research--Prototype-blue)
![Architecture](https://img.shields.io/badge/Architecture-Rank--k%20Factorization-orange)

**Attention Neuron** is a neural architecture that rethinks the dense weight matrix as a dynamically modulated, low-rank system. Instead of learning millions of individual weights, the network learns a small set of neuron-centric modulation vectors that reconfigure a frozen random substrate.

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

## Project Structure

- `scratch/`: Experimental prototypes (v1 through v24).
- `docs/`: Technical whitepapers, findings per version, and research plans.
- `results/`: Saved model checkpoints and training logs.

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
