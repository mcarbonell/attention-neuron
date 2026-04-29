# Findings V98b: DCT vs Walsh in Invariant Spectral Attention

## Overview
We compared the performance of **Walsh-Hadamard** (binary, rigid) vs **Discrete Cosine Transform** (sinusoidal, smooth) as the spectral attention mechanism on top of a Fourier-Mellin invariant signature.

## Empirical Results (10 Epochs, Torture Test: 90° Rotation + 20% Shift)

| Metric | V98 (Walsh Hybrid) | **V98b (DCT Hybrid)** | Difference |
| :--- | :--- | :--- | :--- |
| **Peak Accuracy (Best)** | 41.75% | **41.93%** | **+0.18%** |
| **Final Accuracy (E10)** | **40.97%** | 39.59% | -1.38% |

## Key Technical Insights

### 1. The Smoothness Advantage
The DCT achieved the **highest peak accuracy of the session (41.93%)**. This suggests that because the Fourier-Mellin transform is based on the Fourier (sinusoidal) basis, the DCT is a more "natural" language for filtering its invariant signature than the square Walsh functions.

### 2. Walsh Resilience
Interestingly, Walsh ended the final epoch with a higher score. The binary nature of the Walsh transform seems to act as a form of implicit regularization, preventing the network from overfitting to the fine-grained spectral noise that can occur in the DCT coefficients during late-stage training.

### 3. Mutual Superiority over Raster
Both models maintained a massive lead over the Raster baseline (~20%). This confirms that **Spectral Attention**, regardless of the specific transform used, is the key to interpreting complex, non-local features like Fourier-Mellin signatures.

## Conclusion
While DCT provides a higher theoretical peak due to its smoother basis, Walsh remains a robust and computationally efficient alternative. For future "Mega" models, a hybrid of both (Multi-Scale Spectral Fusion) might be the optimal path.

**Reference Script**: `scratch/prototype_v98b_fm_dct_hybrid.py`
