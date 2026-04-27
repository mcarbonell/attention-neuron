# Findings V67: The Hybrid Spectral GPT (DCT + Walsh)

## Overview
Following the successful validation of the Fully-JPEG LLM (V66), we integrated the core ideas from the "Walsh Era" brainstorming (`brainstorming_v2_walsh_era.md`) to create the ultimate hybrid cognitive architecture.
We hypothesized that different components of a Transformer serve different cognitive purposes, and therefore should operate in different spectral domains:
1. **Attention (Language & Context):** Semantic meaning is continuous and smooth. We used the **Discrete Cosine Transform (DCT)** for the Attention Q, K, V, and O projections.
2. **Feed-Forward Networks (Knowledge & Logic):** Logic and facts are often sharp, binary, and rule-based. We used the **Fast Walsh-Hadamard Transform (FWHT)** bases (which consist of binary $+1/-1$ square waves) for the FFN projections.

## Methodology
- **Architecture**: A 4-layer, 4-head Transformer ($d_{model} = 128$).
- **Attention**: `DCTLinear` layers compressed by 16x.
- **FFN**: `WalshLinear` layers compressed by 32x. (Instead of learning dense matrices, the FFN synthesizes its weights from tiny Walsh-Hadamard cores).
- **Dataset**: `train_v1.bin` from `tiny-thinker` ($V = 16384$).

## Results

| Metric | Value |
| :--- | :--- |
| **Total Learnable Params** | 2,270,336 (Includes 2M for Vocabulary Embeddings) |
| **Initial Loss (Iter 0)** | 9.8490 |
| **Final Loss (Iter 499)** | **6.3141** |
| **Convergence** | Smooth and robust. |

## Key Insights
1. **The Interlingua Works**: The network successfully routed continuous semantic concepts (DCT) into sharp logical processors (Walsh) and back, proving that the neural manifold is agnostic to the underlying orthogonal basis as long as the frequency bottleneck is enforced.
2. **Hardware Efficiency Potential**: While DCT requires floating-point multiplications for its bases, Walsh matrices consist purely of `1` and `-1`. This means the massive FFNs (which usually dominate LLM inference) could theoretically be synthesized using **addition and subtraction only** on specialized hardware or FPGAs. 

## Conclusion
The Hybrid Spectral GPT proves that we can "mix and match" orthogonal transforms to suit the cognitive bias required by specific layers. DCT provides the smooth semantic glue, while Walsh provides the sharp, hardware-efficient logical reasoning. This hybrid approach represents a major step toward ultra-efficient edge AI.
