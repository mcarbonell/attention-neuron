# Findings v117: The Infinite Resolution Paradox

## Experiment Summary
We tested the limits of the spectral architecture by increasing the input resolution from 4,096 to **32,768 samples** per image using the Log-Polar Spiral sampler. Due to the spectral nature of the neurons, the parameter count remained fixed (~2k parameters).

## Results

| Samples | 0° Accuracy | 15° Accuracy | 45° Accuracy |
|---------|-------------|--------------|--------------|
| 4,096   | **79.11%**  | **75.82%**   | **47.02%**   |
| 8,192   | 77.57%      | 74.95%       | 45.85%       |
| 16,384  | 69.22%      | 64.46%       | 35.77%       |
| 32,768  | 74.32%      | 71.45%       | 43.91%       |

## Why did performance degrade with more samples?

1.  **The Spectral Bottleneck**: We kept $k=16$ (spectral coefficients). At 32k samples, each coefficient is trying to represent the average behavior of **2,048 samples**. We are performing an extreme low-pass filtering that destroys all useful detail.
2.  **Redundancy Saturation**: With $q=4.0$ (slow start) and 32k samples, the first few thousand samples are effectively identical (sampling the same central pixels over and over). This creates a "dead zone" in the input vector that provides no information but consumes the network's attention.
3.  **Optimization Complexity**: While the number of parameters is the same, the "landscape" of the spectral projection might become flatter or more complex as the input dimensionality grows, making 3-epoch training insufficient.

## Conclusion
Increasing input resolution only works if you also increase **spectral resolution** ($k$). Otherwise, you are just "blurring" the same 16 basis functions over a larger area. The "Infinite Resolution" theory requires a corresponding "Infinite Spectral Capacity".
