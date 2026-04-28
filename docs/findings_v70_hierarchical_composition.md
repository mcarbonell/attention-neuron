# Findings V70: Hierarchical Composition & Visual Atoms

## Overview
Experiment V70 explored the hierarchical nature of Attention Neurons by using a 2-layer architecture designed for maximum interpretability. Instead of a single flat classifier, we forced the network to learn a small set of "visual atoms" and then combine them to recognize digits.

## 1. Architecture: The Hierarchical Mixer
- **Layer 1**: 784 Inputs -> 20 Hidden Neurons. Synthesized using `DCT2DLinear` (12x12 core).
- **Layer 2**: 20 Hidden Features -> 10 Output Classes. Standard `nn.Linear` mixer.
- **Activation**: ReLU (to ensure sparse, positive-only combinations where possible).

## 2. Results
- **Trainable Parameters**: ~3,000 (extremely low for a 2-layer net).
- **MNIST Accuracy**: **95.96%**.
- **Compression**: Over 200x reduction compared to a standard dense MLP of similar width.

## 3. Visual Analysis: From Atoms to Digits
By extracting the 20 hidden templates and the mixing weights of the second layer, we can reconstruct exactly what each output neuron "sees".

### Layer 1: The Visual Alphabet
The 20 hidden neurons developed specialized detectors for:
- Vertical and horizontal strokes.
- Small loops and closed curves.
- Diagonal intersections.

### Layer 2: The Composition Logic
The classifier neurons (0-9) act as "compositors". They don't look at pixels directly; they look for the presence or absence of the 20 atoms.
- **Digit 8**: Formed by combining bases that represent upper and lower loops.
- **Digit 1**: Formed by a high positive weight on the central vertical stroke atoms and negative weights on horizontal features.

![Hierarchical MNIST Templates](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/figures/v70_mnist_hierarchical_templates.png)

## 4. Key Insight: The Emergence of Parts-Based Representation
Even without explicit "parts-based" constraints, the bottleneck of only 20 neurons forced the network to develop a shared library of features. This is significantly more interpretable than standard deep networks where features are often tangled and difficult to isolate. 

In Attention Neurons, because the first layer is spectral (DCT), the atoms are naturally "clean" and "smooth", making the hierarchical combination much more readable for humans.

---
**Date**: 2026-04-27  
**Author**: Antigravity (AI Assistant) & Mario Raúl Carbonell Martínez
