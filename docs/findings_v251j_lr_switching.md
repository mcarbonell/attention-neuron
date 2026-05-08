# Findings v251j: High-LR Warmup & Switching

## Goal
Maximize training speed and stability by using a high LR for structural gating and a low LR for sequential weight refinement.

## Phase 1: Gating Warmup (LR 0.05)
| Epoch | Accuracy (%) | Note |
| :--- | :---: | :--- |
| 1 | 87.71% | Reached near-peak gating accuracy in one pass. |
| 2 | 88.63% | Slight refinement. |
| 3 | 88.53% | Saturation reached. |

## Phase 2: Rotating Weights (LR 0.001, Gating Frozen)
| Epoch | Layer Active | Accuracy (%) |
| :--- | :---: | :---: |
| 4 | Layer 1 | 95.32% |
| 5 | Layer 2 | 96.34% |
| 6 | Layer 3 | 97.23% |
| 9 | Layer 3 | **97.63%** |

## Analysis
- **Structural Efficiency**: The gating phase (Epoch 1) is incredibly efficient. It performs a "coarse-grained" optimization that places the network in a highly favorable region of the loss landscape.
- **Seamless Switch**: Transitioning from gates to weights caused zero instability. Instead, it triggered a massive accuracy jump (+7% in one epoch).
- **Compute Savings**: During the weight phase, the model achieved near-SOTA performance while only updating 33% of the weights in each step.

## Final Protocol Recommendation
1. **Gating Pass**: 1 Epoch, LR 0.05.
2. **Refinement Pass**: 6-9 Epochs, Layer-wise Rotation, LR 0.001.
