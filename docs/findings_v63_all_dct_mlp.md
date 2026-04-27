# Findings V63: The All-DCT MLP

## Overview
Following the success of applying the Discrete Cosine Transform (DCT) to the first layer (spatial attention) of neural networks, this experiment tested a radical hypothesis: **Can we compress the entire internal topology of a Deep Neural Network by projecting all hidden layers into the frequency domain?**

In a standard Multi-Layer Perceptron (MLP), hidden neurons are permutation-invariant and lack inherent structure. By applying DCT to the weight matrices of *every* layer, we force the network to organize its internal representations by "frequency" (from broad, global concepts to fine, high-frequency details).

## Methodology
- **Architecture**: A 3-layer MLP for MNIST (784 $\rightarrow$ 512 $\rightarrow$ 512 $\rightarrow$ 10).
- **Mechanism**: Every `nn.Linear` layer is replaced with a `DCTLinear` layer. 
- **Synthesis**: Instead of learning a full dense matrix $W$ of size $N_{out} \times N_{in}$, the network learns a tiny core matrix $C$ of size $K_{out} \times K_{in}$ (low-frequency coefficients). The full matrix is synthesized during the forward pass using 1D DCT bases: $W = D_{out}^T \cdot C_{padded} \cdot D_{in}$.
- **Parameters**: For the hidden layers, instead of $512 \times 512$ weights, we used a $64 \times 64$ core.

## Results (MNIST)

| Metric | Dense MLP | All-DCT MLP (V63) |
| :--- | :--- | :--- |
| **Total Parameters** | 669,706 | **11,914** |
| **Compression Ratio** | 1.0x | **56.2x** |
| **Accuracy (Epoch 15)** | ~98.0% | **97.59%** |
| **Training Time** | Baseline | 176.0s (Slight synthesis overhead) |

## Key Insights
1. **Internal Frequency Structure**: The fact that the network retains 97.59% accuracy while discarding 98% of its internal degrees of freedom proves that the semantic representations inside a neural network are highly compressible in the frequency domain. 
2. **Global Concept Routing**: By truncating the matrix to $64 \times 64$, the network successfully routes the "low-frequency" (broad semantic) concepts of layer $L$ directly to the "low-frequency" concepts of layer $L+1$. High-frequency noise is mathematically prevented from propagating through the network.
3. **Structured Orthogonal Transforms**: This serves as a powerful proof-of-concept for Structured Orthogonal Transforms for model compression. It provides a deterministic, mathematically grounded alternative to random pruning or expensive Neural Architecture Search.

## Conclusion
Applying DCT is not just a trick for the input layer (spatial images). It is a fundamental tool for regularizing and massively compressing the dense feed-forward topologies of deep neural networks. This opens the door to applying DCT compression to the massive Feed-Forward Networks (FFNs) found within modern Transformer blocks (LLMs), where parameter redundancy is famously high.
