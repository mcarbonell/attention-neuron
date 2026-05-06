# Findings - Spectral V8.4 "Spectral Residency" Optimization

The Spectral V8.4 architecture implements "Spectral Residency", maintaining the hidden state in the Walsh domain throughout the entire model.

## Performance Comparison (v8.3 vs v8.4)

Test Configuration: 32768 Dim, 16 Layers, 128 Experts (CPU: Ryzen 7 8845HS)

| Version | FWHTs per Token | Tok/s | Speedup |
|---------|-----------------|-------|---------|
| V8.3 (Baseline) | ~160 | 1.42 | 1.0x |
| **V8.4 (Optimized)** | **2** | **4.92** | **3.46x (+246%)** |

## Key Architectural Wins

### 1. Eliminating Domain Switches
By keeping the state in the spectral domain, we eliminated nearly 98% of the Walsh-Hadamard Transforms. The model only transforms back to spatial domain at the very end to project logits.

### 2. Spectral-RMSNorm
We successfully implemented RMSNorm in the spectral domain. By Parseval's theorem, the energy calculation remains identical, allowing for layer normalization without exiting the spectral residency.

### 3. Scaling Potential
With 4.92 Tok/s at 32k dimension, we have reached a "usable" threshold for local LLM experimentation. This optimization allows us to:
- Increase depth to 32+ layers while staying above 2 Tok/s.
- Experiment with even higher dimensions (65k+) for better frequency resolution.

## Remaining Bottlenecks
Even with only 2 FWHTs, the matrix-free MoE routing (`torch.mm` between spectral state and expert signatures) now accounts for a larger percentage of the compute. However, this part is highly parallelizable and less of a bottleneck than the iterative FWHT was.

---
*Results generated on 2026-05-06 with prototype_v163k_v8_4_comparison.py*
