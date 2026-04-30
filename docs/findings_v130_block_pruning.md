# Findings V130: Block-Based Spectral Pruning

## Executive Summary
We attempted to push the spectral pruning limit of **GPT-2** beyond 50% by using localized **Block-Based Walsh Transforms** and **Variance Rescaling**. While the failure mode became "smarter" (retaining some semantic relevance like the word "human"), the model still suffered from catastrophic coherence loss at 25% (4x compression).

## Results: Global vs. Block Pruning (at 25%)

| Method | Error Pattern | Coherence |
| :--- | :--- | :--- |
| **Global Walsh (V129)** | String Repetition | Very Low |
| **Block Walsh (V130)** | Single Word Loops ("is is is") | Extremely Low |
| **Normalized Block (V130c)** | Semantic Noise ("human human no human") | Low (but improved) |

## Key Insights

### 1. The Variance is the Anchor
Without **Variance Rescaling**, the model output quickly decays into empty strings or garbage characters. Restoring the original standard deviation of the weights allows the signal to propagate through the 12 layers, but it doesn't fix the corrupted logic of the Attention heads.

### 2. Locality Matters (But is not enough)
Moving from 64x64 to 16x16 blocks improved the semantic "flavor" of the errors. This confirms that preserving local spectral properties is better than global ones, but zeroing out 75% of the coefficients simply removes too many critical "phase" relationships required for the Transformer's delicate matrix multiplications.

### 3. The "Zero-Shot" Limit
We have established that for GPT-2, the **maximum stable zero-shot compression ratio is 2x (50%)**. To go further, we likely need to either:
- **Fine-tune** the remaining coefficients (Spectral LoRA).
- **Quantize** instead of pruning (keeping all coefficients but at lower bit-depth).

## Conclusion
Block-based spectral pruning is a superior way to compress weights compared to global pruning, but it cannot overcome the fundamental information loss at 4x compression without further adaptation.

**Next Step**: Shift research towards **Spectral Quantization** (keeping 100% of coefficients but at 1-bit or 2-bit precision).
