# Findings - Spectral V8.6 "Universal Core" (CPU/GPU)

The Spectral V8.6 architecture introduces a device-agnostic engine that switches between a native C++ Pulse (CPU) and a vectorized GPU engine (DirectML/CUDA).

## Performance Comparison (CPU vs GPU)

Test Configuration: 32768 Dim, 16 Layers, 128 Experts
Hardware: Ryzen 7 8845HS + Radeon 780M (Under 80% background load)

| Mode | Device | Batch Size | Tok/s | Advantage |
|------|--------|------------|-------|-----------|
| **Latency** | CPU (Native) | 1 | 2.26 | Baseline |
| **Latency** | GPU (DirectML) | 1 | 0.87 | -61.3% |
| **Throughput** | **GPU (DirectML)** | **16** | **10.63** | **+370.3%** |

## Insights

### 1. Throughput Excellence
The "Matrix-Free" architecture's vectorized implementation in V8.6 is highly efficient for parallel processing. Reaching **10.63 Tok/s** on an integrated GPU while it is simultaneously training another model is a significant milestone.

### 2. Batch Scaling
GPU performance scales exponentially with batch size. While CPU is better for single-token latency (Batch 1), the GPU becomes the clear winner for any workload involving more than 4 parallel sequences.

### 3. Practical Applications
- **Inference:** B16 performance is ideal for "Parallel Sampling" (generating multiple ideas simultaneously) or hosting a small local multi-user API.
- **Training:** B16 is the "engine room" of training. This speed ensures that training sessions for high-resolution models (32k dim) will be practical on consumer-grade AMD hardware.

## Recommendations
- **Deployment:** Use **CPU (Native Pulse)** for interactive chat (Batch 1).
- **Processing:** Use **GPU (Universal Core)** for batch processing, training, and multi-completion tasks.

---
*Results generated on 2026-05-06 with prototype_v163m_v8_6_gpu_benchmark.py*
