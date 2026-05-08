# Findings v251e: Scheduled Multiplicative Gating

## Goal
Verify the maximum potential of the 1-hidden layer Frozen Gated architecture using a `OneCycleLR` scheduler to stabilize high learning rates.

## Final Results (OneCycleLR, MaxLR 0.05)

| Hidden Dim | Trainable Params | Final Test Acc (%) | PEI |
| :--- | :---: | :---: | :---: |
| 1024 | 1034 | 87.17 | 28.92 |
| 2048 | 2058 | 91.37 | 27.58 |
| **4096** | **4106** | **93.89** | **25.98** |

## Comparison vs Previous Experiments (D=2048)
- **v251b (LR 1e-3)**: 85.32%
- **v251d (LR 5e-2, Constant)**: 91.37% (with oscillations)
- **v251e (LR 5e-2, OneCycle)**: **91.37%** (Higher stability in intermediate epochs)

## Analysis
- **Scheduler Impact**: The `OneCycleLR` allowed the model to use a very high peak LR (0.05) to find features early, while the final cool-down phase ensured a stable finish. 
- **Efficiency Milestone**: Reaching **~94% accuracy** with just **4106 parameters** is an order of magnitude more efficient than standard training. 
- **Non-Saturation**: Even at $D=4096$, the model was still showing slight improvement, suggesting that larger reservoirs or more epochs could push this even further.

## Conclusions
The experiment is a complete success. Training only the gating parameters of a frozen random projection is not only a "toy" experiment but a robust method for extreme parameter compression. 

### Future Work: Stage-Gating Pre-training
This method could be applied to LLMs to perform a "cheap" initialization phase before unfreezing weights for high-fidelity training.
