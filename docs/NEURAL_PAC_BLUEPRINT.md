# Neural-PAC: The Interpretable Spawning Architecture

## Executive Summary
**Neural-PAC** (Neural Purifying Archetype Classifier) is a hybrid architecture that merges the supervised clustering logic of the PAC algorithm with the differentiable power of Neural Networks and the spectral efficiency of Discrete Cosine Transforms (DCT). It eliminates the "Black Box" nature of traditional deep learning by replacing abstract weights with **generative archetypes**.

## Core Principles

### 1. Archetypes as Parameters
In a standard neuron, knowledge is stored as an array of weights $W$. In Neural-PAC, a neuron stores **spectral coefficients** (DCT or Walsh) that generate a human-readable image (the Archetype).
- **Forward Pass**: The neuron generates its archetype $A$ and computes the distance $d = ||x - A||^2$ to the input $x$.
- **Interpretation**: If a neuron activates, it is because the input *looks like* the neuron's drawing.

### 2. Selective Positive Backpropagation
To prevent "blurring" caused by traditional cross-entropy (where a neuron tries to avoid all other classes simultaneously), Neural-PAC uses **Selective Updates**.
- Only the archetype belonging to the **true class** of the input receives a gradient.
- This forces neurons to become pure averages (means) of their class features rather than "anti-class" filters.

### 3. Neurogenesis through Spawning (Dynamic Topology)
Instead of a fixed architecture, Neural-PAC grows based on error detection.
- **Stability Phase**: Initial archetypes capture the global mean of each class.
- **Expansion Phase**: If an input is misclassified or exceeds a distance threshold, the network performs a **Spawn**.
- **Initialization by Reconstruction**: New neurons are initialized using **Forward DCT** on the misclassified image, ensuring they start as a perfect "specialist" for that specific edge case.

## Structural Advantages

| Feature | Traditional Neural Nets | Neural-PAC |
| :--- | :--- | :--- |
| **Interpretabilidad** | Post-hoc (Heatmaps, LIME) | **Intrínseca** (Galería de Arquetipos) |
| **Aprendizaje** | Fine-tuning global | **Aditivo / Localizado** |
| **Olvido Catastrófico** | Alto (Interferencia de pesos) | **Cero** (Nuevas neuronas para nuevos datos) |
| **Eficiencia** | Millones de pesos abstractos | Miles de coeficientes espectrales |
| **Seguridad** | Vulnerable a ruido adversario | Resistente (Filtro DCT natural) |

## Evolutionary Implications
Neural-PAC moves AI closer to biological plasticity. It allows models to:
1. **Learn Forever**: Add new concepts without re-training or forgetting.
2. **Be Auditable**: Every decision can be traced to a specific visual archetype.
3. **Hardware Efficient**: Optimized for spectral transforms and distance-based logic.

---
*Derived from Attention Neuron Research Phase 3 (V85-V86).*
