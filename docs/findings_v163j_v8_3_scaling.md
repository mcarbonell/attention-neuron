# Findings - Spectral V8.3 Matrix-Free Benchmark

A comprehensive sweep of 60 configurations was performed on an AMD Ryzen 7 8845HS.

## Key Insights

### 1. The FWHT Bottleneck
The speed (Tok/s) is dominated by the spectral dimension ($D$).
- **Small Scale (2k-4k):** Very fast (20-30 Tok/s), suitable for real-time mobile/edge.
- **Large Scale (32k):** Slows down significantly (< 1 Tok/s at high depth).
- **Reason:** $O(D \log D)$ complexity of the Walsh-Hadamard Transform in pure PyTorch becomes the primary compute consumer as $D$ grows.

### 2. Expert Decoupling (The Matrix-Free Win)
Increasing the number of experts from **32 to 256** has a disproportionately small impact on speed.
- At 32k dim / 24 layers:
    - 32 experts: 0.87 Tok/s
    - 256 experts: 0.70 Tok/s
- **Conclusion:** We can afford a high number of experts (high resolution) without the quadratic memory/compute penalty of traditional MoE.

### 3. Linear Layer Scaling
Speed scales almost perfectly inversely with the number of layers ($L$). This confirms no hidden overheads in the block stacking.

## Scaling Table (Sweet Spots)

| Profile | Dim | Experts | Layers | Params | Tok/s | Note |
|---------|-----|---------|--------|--------|-------|------|
| **Edge-Fast** | 2048 | 128 | 4 | 6.6M | 32.1 | Ultra-responsive |
| **Balanced** | 8192 | 128 | 8 | 22.5M | 7.6 | Good reasoning/speed ratio |
| **High-Res** | 16384 | 256 | 8 | 74.3M | 3.7 | Professional grade resolution |
| **Max-Think** | 32768 | 256 | 24 | 416.6M | 0.7 | Research/Reasoning limit |

## Recommendations

1. **FWHT Optimization:** To unlock 32k+ dimensions at > 5 Tok/s on CPU, we need a vectorized C++/SIMD implementation of the FWHT.
2. **Spectral Residency:** Reduce the number of `fwht` calls by staying in the spectral domain between layers or within the Attention/MoE blocks.
3. **PEI Target:** Focus training on the **Balanced** (22M) profile to maximize intelligence per parameter without sacrificing interactivity.

---
*Results generated on 2026-05-06 with prototype_v163j_v8_3_sweep.py*
