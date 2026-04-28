# Findings V71: The Neural Vectorizer (SGD Image Tracing)

## Overview
Following the exploration of interpretable representations, we devised a radical experiment: **Inverting the process**. Instead of a neural network "looking" at an image to classify it, we asked if a set of neurons could be optimized to "draw" a specific image.

We used the Parametric Stroke Neurons (from V50), which use mathematical Quadratic Bezier curves.

## 1. The Experiment Setup
- **Target**: A single MNIST image (a digit '5').
- **Canvas**: An empty 28x28 grid.
- **Model (The "Brush")**: 12 Parametric Stroke Neurons. Each neuron has 3 learnable control points `(P0, P1, P2)` and an opacity weight.
- **Objective**: Optimize the control points and weights using standard SGD (Adam) to minimize the Mean Squared Error (MSE) between the generated canvas and the target pixel image.

## 2. Results & Observations
The vectorization was remarkably fast and successful.

- **Speed**: The model converged to a near-perfect vector representation in under 60 epochs (a few seconds of compute).
- **Physics-like Dynamics**: Initially, the 12 strokes were initialized randomly in the center of the canvas (Epoch 1). Over time, the gradient "pushed" and "stretched" these strokes to snap onto the bright pixels of the target '5'.
- **Resolution Independence**: Because the final image is represented by continuous mathematical points (the Bezier controls) rather than a grid of pixels, the resulting '5' can theoretically be scaled to infinite resolution without pixelation.

![Vector Tracing Evolution](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/figures/v71_vector_tracing_evolution.png)

## 3. Implications for Generative AI (The "Glyph Generator")
This experiment serves as a powerful proof of concept for a novel type of generative model.

Current image generators (like Diffusion models) predict millions of independent pixels. Our approach suggests a **Vectorial Generative Model**:
1. A small neural network takes a concept embedding (e.g., from CLIP for the word "Five").
2. Instead of outputting pixels, the network outputs the `N x 3` coordinates for a set of Bezier strokes.
3. The image is mathematically rasterized.

### Advantages:
- **Tiny Parameter Footprint**: Generating 12 strokes requires predicting only 72 numbers, compared to generating 784 pixels (or millions for HD images).
- **Infinite Scalability**: The output is inherently a vector graphic (SVG).
- **Clean Structure**: By definition, the generated shapes will have smooth, continuous curves, perfectly suited for generating fonts, icons, glyphs, or clean UI elements.

---
**Date**: 2026-04-28  
**Author**: Antigravity (AI Assistant) & Mario Raúl Carbonell Martínez
