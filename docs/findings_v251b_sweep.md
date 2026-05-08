# Findings v251b: Multiplicative Gating Sweep

## Goal
Analyze the scaling laws of the Frozen-Weight Multiplicative Gating architecture on MNIST. We vary the hidden dimension $D$ while keeping all linear weights frozen, leaving only $D+10$ parameters as trainable gates.

## Results Sweep

| Hidden Dim | Trainable Params | Final Test Acc (%) | PEI |
| :--- | :---: | :---: | :---: |
| 32 | 42 | 34.50 | 21.12 |
| 64 | 74 | 37.78 | 20.15 |
| 128 | 138 | 47.03 | 21.95 |
| 256 | 266 | 59.64 | 24.58 |
| 512 | 522 | 74.32 | 27.34 |
| 1024 | 1034 | 82.46 | 27.35 |
| 2048 | 2058 | 85.32 | 25.75 |
| 4096 | 4106 | 89.30 | 24.71 |

## Scaling Laws
- **Log-Linear scaling**: Accuracy scales almost linearly with the logarithm of the hidden dimension until $D=1024$, where it begins to face diminishing returns.
- **PEI Peak**: The maximum parametric efficiency is found around $D=1024$. Beyond this point, we are adding more gating parameters than the accuracy gain justifies in terms of "efficiency", although total performance continues to rise.
- **Extreme Sparsity**: Achieving ~90% accuracy with only 4k parameters (in a sea of 400k+ frozen ones) proves that the "basis functions" provided by random initialization are sufficiently rich for MNIST.

## Conclusions
The experiment confirms that training only the gating of a frozen random projection is a viable and extremely cheap training method. The next logical step is to determine if multi-layer frozen architectures (Deep Random Networks) can bridge the remaining 10% gap to state-of-the-art MLP performance.
