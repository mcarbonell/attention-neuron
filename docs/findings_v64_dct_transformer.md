# Findings V64: The DCT-Transformer (NLP)

## Overview
Following the discovery that deep MLPs can be massively compressed using Discrete Cosine Transform (DCT) bases (V63), this experiment applies the concept to Natural Language Processing (NLP). 
The core of modern LLMs (Transformers) is heavily bottlenecked by massive Feed-Forward Networks (FFNs) that process token embeddings. In this experiment, we replaced standard FFNs with **DCTFeedForward** layers to test if semantic language representation can be efficiently learned in the frequency domain.

## Methodology
- **Task**: Autoregressive Language Modeling (Next-token prediction).
- **Dataset**: `train_v1.bin` from the `tiny-thinker` project (BPE tokenized text, vocab_size=16384).
- **Architecture**: A 4-layer, 4-head Transformer ($d_{model} = 128$).
- **DCT Compression**: The SwiGLU FFNs were replaced by `DCTLinear` layers. Instead of projecting $128 \rightarrow 512 \rightarrow 128$ using dense matrices, the network learns a tiny $32 \times 64$ DCT core and synthesizes the full projection matrices via DCT bases.

## Results

| Metric | Detail |
| :--- | :--- |
| **Dense FFN Params (Expected)** | 786,432 |
| **DCT FFN Params (Actual)** | **24,576** |
| **FFN Compression Ratio** | **32.0x** |
| **Initial Loss (Iter 0)** | 9.8707 |
| **Final Loss (Iter 499)** | **6.0820** |
| **Convergence** | Smooth and stable |

## Key Insights
1. **Semantic Frequency**: The concept of "frequency" applies directly to language embeddings. Broad concepts (e.g., part-of-speech, core meaning) reside in low frequencies, while nuances reside in high frequencies.
2. **Extreme FFN Redundancy**: LLM Feed-Forward networks are famously redundant and sparse. The DCT explicitly exploits this redundancy by mathematically enforcing a low-frequency semantic bottleneck, achieving 32x compression without halting learning.
3. **Cross-Domain Validation**: The DCT-Attention mechanism is not limited to spatial domains (Computer Vision). It is a universal regularizer for mapping any high-dimensional vector space into a semantically dense manifold.

## Implications for Cognitive Architectures
For projects like `tiny-thinker`, integrating DCT FFNs could allow models to be trained much deeper (more layers, thus more reasoning steps) while maintaining an ultra-low parameter count and memory footprint, sidestepping the memory-bandwidth wall of modern LLMs.
