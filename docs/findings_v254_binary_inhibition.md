# Findings V254: The Importance of Inhibition ({0,1} vs {-1,0,1})

## Overview
To isolate the contribution of negative weights, we benchmarked a version of the gated network using **Binary {0, 1} Weights** instead of Ternary.

## Results (10 Epochs, MNIST)

| Weight Type | Peak Accuracy | Initial Loss (B1) |
| :--- | :--- | :--- |
| Ternary $\{-1, 0, 1\}$ | **94.7%** | **18.8** |
| Binary $\{0, 1\}$ | 41.4% | **357.4** |

## Key Technical Insights

### 1. Cumulative Bias Explosion
Binary weights $\{0, 1\}$ have a positive mean (0.5). In a deep network, this causes activations to accumulate and grow exponentially with each layer. The massive initial loss (357.4) reflects a "saturated" softmax where logits are pushed to extremes.

### 2. Lack of Spatial Contrast
Inhibition (negative weights) is essential for fundamental vision tasks:
- **Contrast**: Detecting the difference between a pixel and its background requires $x_{pixel} - x_{bg}$.
- **Edge Detection**: Requires comparing adjacent signals.
Without the `-1`, the network can only perform "Summation Pooling," losing the ability to define boundaries.

### 3. Failure of "Discovery"
While the ternary network successfully discovered features from a zero-initialized state, the binary network remained unstable. The "signal-to-noise" ratio in a purely additive random system is too low for the gating mechanism to find meaningful structure.

## Conclusion
Negative weights are not just "extra parameters"; they are a structural necessity for zero-mean signals and differential feature extraction.

**Reference Script**: `scratch/prototype_v254_binary_weights.py`
