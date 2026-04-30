# Findings V131: Spectral Quantization

## Executive Summary
We tested **Spectral Quantization** (1-bit and 2-bit Walsh) on **GPT-2** as an alternative to pruning. The results were catastrophic, with the model losing all linguistic ability and generating only punctuation marks. This confirms that for traditionally trained LLMs, the relative magnitudes of spectral coefficients are just as critical as their signs.

## Results: Spectral Quantization vs. Pruning

| Method | Result | Reason for Failure |
| :--- | :--- | :--- |
| **Pruning 50% (V129)** | **Coherent** | Removes noise while keeping signal precision. |
| **1-bit Walsh (V131)** | **Total Collapse** | Noise amplification: zero-ish coefficients are "boosted" to the mean magnitude. |
| **2-bit Walsh (V131)** | **Total Collapse** | Insufficient dynamic range to capture the sharp "keys" in MLP layers. |

## Key Insights

### 1. The Noise Amplification Trap
In the Walsh domain, LLM weights are sparse (as seen in V128). When we apply 1-bit quantization, we are forcing the sparse "background" coefficients to have the same weight as the "signal" coefficients. This effectively turns the weight matrix into a high-entropy noise mask, drowning out the learned associations.

### 2. Spectral Phase is not enough
While preserving the sign of coefficients preserves the "phase," the **dynamic range** of the spectral coefficients in a Transformer is huge. Unlike images where many frequencies have similar importance, LLM weights seem to rely on a few "high-energy" spectral peaks.

### 3. Native vs. Post-Training
This experiment proves that you cannot easily "quantize spectrally" a model that was trained in the spatial domain (float32 dense weights). The optimization paths of standard LLMs do not encourage spectral sparsity or quantization robustness.

## Conclusion
Post-training spectral quantization is NOT a viable zero-shot method for dense LLMs. 

**Next Step**: Return to **Native Spectral Training**. If we want a quantized spectral LLM, we must train it from scratch using spectral layers and optimizers, forcing the model to find a representation that is robust to this domain.
