# Findings v251: Multiplicative Gating Experiment

## Goal
Evaluate the impact of multiplicative gating at the neuron level ($y = (xW + b) \odot g$) on MNIST, comparing standard training vs. training only the gating parameters with frozen weights.

## Experiment Configuration
- **Dataset**: MNIST
- **Architecture**: 2-layer MLP (Input 784, Hidden 512, Output 10)
- **Training**: 10 epochs, Adam (LR 1e-3), Batch Size 64
- **Hardware**: CPU (v3.13)

## Results

| Model | Test Accuracy (%) | Trainable Params | Wall Time (s) | PEI |
| :--- | :---: | :---: | :---: | :---: |
| Standard MLP | 97.93 | 407,050 | 136.83 | 17.46 |
| Gated MLP (Full) | 97.76 | 407,572 | 139.52 | 17.43 |
| Gated MLP (Frozen) | 77.02 | 522 | 120.72 | 28.33 |

## Key Findings
- **Parametric Efficiency**: The frozen variant achieves **77.02% accuracy** with only **522 parameters**. This is a **780x reduction** in trainable parameters compared to the baseline, while maintaining significant performance.
- **PEI Advantage**: The PEI of the frozen variant (28.33) is substantially higher than the standard MLP (17.46), demonstrating much higher "intelligence per parameter".
- **Gating Impact**: Adding gating to a fully trainable model (Gated Full) did not yield significant gains on MNIST, slightly decreasing accuracy (97.76% vs 97.93%).
- **Internal Overhead**: As expected for small models, `internal_overhead_time` represents ~90% of total training time, highlighting the dominance of optimizer and loop logic over actual compute.

## Conclusions
Multiplicative gating on frozen random weights is an extremely efficient way to adapt a projection. While it doesn't match full training accuracy, the **77% accuracy with sub-1k parameters** suggests that the random projections in high dimensions ($d=512$) capture enough variety that simple scaling/gating of these features is sufficient for basic classification. This aligns with the "Random Kitchen Sinks" or "Reservoir Computing" philosophy.
