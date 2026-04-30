# Findings V129: LLM Spectral Pruning

## Executive Summary
We tested the linguistic resilience of **GPT-2** under extreme spectral pruning in the Walsh domain. We discovered a "Free Compression Threshold" at **50%**: the model remains coherent and grammatically correct despite losing half of its spectral coefficients. Beyond this point, the model suffers a rapid structural collapse.

## Pruning Resilience Results

| Compression Ratio | Linguistic Coherence | Examples |
| :--- | :--- | :--- |
| **100% (Baseline)** | Perfect | "The capital of France is the French Republic..." |
| **50% (2x)** | **High (Stable)** | "The capital of France is the city of Paris..." |
| **25% (4x)** | Low (Repetitive) | "France France France France..." |
| **10% (10x)** | Zero (Noise) | "curiosity curiosity curiosity..." |

## Key Insights

### 1. The 50% "Magic" Threshold
GPT-2 can survive a 50% reduction in Walsh coefficients with zero fine-tuning. This confirms that a massive portion of pre-trained weights is redundant in the sequency domain. The model even seems to become "cleaner" or more direct in some answers at 50% (identifying Paris immediately).

### 2. Sudden Phase Transition
There is a sharp "cliff" between 50% and 25%. This suggests that the attention mechanism relies on a minimum spectral density to maintain the long-range dependencies required for grammar. Once we drop below this density, the softmax distributions likely become too flat or too noisy, leading to the observed repetition loops.

### 3. Energy vs. Coherence
While 50% energy is preserved at ~12% coefficients (from V128), the model requires ~50% of coefficients to stay coherent. This means that **low-energy spectral components are NOT just noise**; they carry the "fine-tuning" required for linguistic precision.

## Conclusion
Spectral pruning is a viable zero-shot compression method for LLMs up to **2x**. 

**Next Step**: Implement **Block-based Spectral Pruning** to see if localized Walsh transforms can preserve more information and push the compression limit to 4x (25%) or beyond.
