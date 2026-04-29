# V99: Triangular Attention Neuron (1D)

## Abstract
Experiment with ultra-compressed neurons defined by only 2 parameters: **center** and **width**. These parameters generate a 1D triangular attention mask. We test this on MNIST (flattened) to verify if geometric priors can replace dense weight matrices.

## Topology
- **Input**: 784 pixels.
- **Layer 1**: 1024 Triangular Neurons (2,048 parameters).
- **Layer 2**: 512 Triangular Neurons (1,024 parameters).
- **Layer 3**: 10 Dense Neurons (5,120 parameters).
- **Total Parameters**: ~8,192 (90%+ reduction vs standard MLP).

## Initial Results
| Config | Final Accuracy | Wall Time |
| :--- | :--- | :--- |
| Raster Order | 79.59% | 281.8s |
| Spiral Order | 74.72% | 218.4s |

## Observations
- **Differentiability**: Confirmed. Parameters `raw_center` and `raw_width` receive gradients and update correctly.
- **Accuracy**: Achieving ~80% accuracy with only 11k parameters is a strong result for this architecture.
- **Ordering**: Surprisingly, Raster Order outperformed Spiral Order. This might be because MNIST is already somewhat "line-oriented" and the triangular windows capture horizontal segments well.
- **Stability**: Some instability was observed in validation accuracy during training (dips to ~10-30% before recovering), possibly due to the interaction between width updates and BatchNorm.

## Efficiency Metrics
- `internal_overhead_time`: TBD
- `step_efficiency`: TBD
- `final_objective`: TBD
