# Findings V85-V86: The Neural-PAC Prototypes

## Overview
These experiments successfully integrated the **PAC (Purifying Archetype Classifier)** algorithm into a differentiable Neural Network framework using **DCT Neurons** as generative models.

## Experiment V85: Core Archetype Learning
- **Goal**: Can a neural network learn to "draw" the digits it is classifying?
- **Setup**: 10 DCT neurons (one per class) trained using **Selective Positive Backpropagation**.
- **Results**: 
    - The archetypes evolved from random noise into clean, sharp averages of the MNIST digits.
    - By ignoring negative gradients from other classes, the network avoided the "blurry blob" problem of standard training.
- **Insight**: Knowledge in a neural network can be stored as a generative basis without losing classification power.

## Experiment V86: Dynamic Spawning & Taxonomy
- **Goal**: Can a network determine its own complexity by creating new neurons for unknown styles?
- **Setup**: A dynamic layer that starts with 10 neurons and performs a **[SPAWN]** event when an error is detected or a distance threshold is exceeded.
- **Results**:
    - The network grew organically to its cap (e.g., 50 or 200 neurons).
    - It created a **Taxonomy of Styles**: instead of one "4", it developed specialized archetypes for "open 4", "closed 4", "slanted 4", etc.
    - **Stability Phase**: Waiting until Epoch 2 to spawn ensured that the base archetypes were well-founded before the architecture expanded.
- **Visual Evidence**: The galleries generated in `results/figures/v86_*.png` show a diverse ecosystem of human-readable models.

## Technical Breakthroughs
1. **Intrinsically Interpretable Classification**: We moved from "Probability of Class 3" to "Similarity to Archetype #45 (which we can see is a 3)".
2. **Neurogenesis by Reconstruction**: Using Forward DCT to initialize new neurons from error images allows the network to incorporate new information instantly.
3. **Winner-Take-All Specialization**: Only the closest archetype in the class is updated, preventing different handwriting styles from interfering with each other's representations.

## Future Path
- Scaling to 1000+ archetypes to achieve >98% accuracy.
- Implementing "Pruning": Removing archetypes that rarely attract images (simulating synaptic pruning).
- Multi-Layer Neural-PAC: Hierarchical archetypes for more complex visual data (CIFAR-10).
