# Findings v251f: Gate Sparsity Analysis

## Goal
Investigate the sparsity of the learned gates in a frozen-weight architecture. By applying Weight Decay, we force the network to select only the most useful random features.

## Results (D=4096, WD=1e-3)
- **Final Test Acc**: 88.49%
- **Mean Gate Value**: 0.008909
- **Std Dev**: 0.354465

## Sparsity Distribution
| Threshold | % of Gates Below |
| :--- | :---: |
| $< 0.1$ | 33.08% |
| $< 0.01$ | 14.38% |
| $< 0.001$ | 7.06% |
| $< 0.0001$ | 3.00% |

## Analysis
- **The Lottery Ticket**: Unlike traditional dense networks where pruning 90% of weights is often possible, this frozen architecture relies on a vast majority of its neurons. Only ~7% are truly silenced at the 0.001 level.
- **Accuracy Trade-off**: The drop from 93.89% to 88.49% indicates that "weak" features (those pushed toward zero by weight decay) are critical for capturing the nuances required for high accuracy on MNIST.
- **Redundancy**: The high utilization suggests that 4096 neurons is not "too much" for this task when using random projections; rather, it provides the necessary coverage of the feature space.

## Conclusions
The experiment suggests that intelligence in frozen random networks is a collective phenomenon. Instead of a few "master neurons", the model learns to combine thousands of weak random detectors. Forcing sparsity through weight decay significantly harms performance, proving that the "noise" in the reservoir is functionally relevant.
