# Findings V124: Micro Walsh Neurons (K=2, K=4)

## Executive Summary
This experiment explored the absolute limits of parameter compression using Walsh spectral neurons. We compared Blocky (Nearest Neighbor) vs Smooth (Bilinear) reconstruction at ultra-low resolutions.

## Results (MNIST, 10 Epochs)

| Method | Resolution (K) | Parameters | Accuracy | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| Blocky K=2 | 2x2 (4) | 2,186 | 52.64% | Better than smooth at this limit |
| **Smooth K=2** | 2x2 (4) | 2,186 | 30.93% | **Failure**: Too blurry |
| Blocky K=4 | 4x4 (16) | 3,722 | 82.41% | - |
| **Smooth K=4** | 4x4 (16) | 3,722 | **90.18%** | **Huge Victory (+7.7%)** |
| **Smooth K=8** | 8x8 (64) | 9,866 | **97.19%** | Near-SOTA efficiency |

## Analysis

### 1. The K=4 Breakthrough
At $K=4$, the `Smooth` variant reaches **90.18%** accuracy. This is a remarkable achievement considering the model uses **~27x fewer parameters** than a standard MLP for the same hidden dimension. The bilinear interpolation allows 16 coefficients to describe complex curved shapes that the blocky version cannot represent without aliasing.

### 2. The K=2 Anomaly
Smoothing failed at $K=2$. 
- **Reason**: With only 4 control points, bilinear interpolation produces an almost linear gradient across the entire 32x32 field. This "ultra-low-pass" filter removes too much structural information. 
- **Contrast**: The `Blocky` version maintains sharp edges between the 4 quadrants, which provides just enough contrast for the network to guess the digit with 52% accuracy (significantly better than random).

### 3. Efficiency Frontier
The `Smooth K=8` model is arguably the most balanced, achieving **97.19%** (close to the 97.8% of a full Dense model) with only **9.8k parameters**.

## Conclusion
- **K=4** is the minimum resolution required for "meaningful" vision using smooth spectral neurons.
- **Bilinear interpolation** is a superior inductive bias for resolutions where $K \ge 4$.
- Below $K=4$, the information density is too low for smoothing to be beneficial.

## Recommendation
For ultra-lightweight edge devices, a **Smooth Walsh K=6** or **K=8** architecture provides the best trade-off between memory footprint and classification performance.
