# V99b: Multi-View Triangular Attention

## Abstract
Expansion of the V99 experiment. Instead of a single 1D ordering, we concatenate three different perspectives of the same image: **Raster (Rows)**, **Transpose (Columns)**, and **Spiral Scan**. Total input size: 2352 pixels. The number of parameters remains constant (~11k).

## Topology
- **Input**: 2352 pixels (3 views x 784).
- **Layer 1**: 1024 Triangular Neurons (2,048 parameters).
- **Layer 2**: 512 Triangular Neurons (1,024 parameters).
- **Layer 3**: 10 Dense Neurons (5,120 parameters).
- **Total Parameters**: ~11,274.

## Final Results
| Config | Peak Accuracy | Final Accuracy | Stability |
| :--- | :--- | :--- | :--- |
| **v99b (3 Views)** | **78.21%** | 64.14% | Medium-Low |
| v99 (Raster Only) | 79.59% | 79.59% | High |

## Observations
- **Efficiency**: The model handles 2352 inputs with only 11k parameters.
- **View Selection**: The neurons distributed themselves almost equally across Rows (334), Cols (396), and Spiral (294).
- **Instability**: Significant accuracy fluctuations (e.g., dropping to 28% in epoch 9) suggest that the optimization landscape for these geometric parameters is non-convex and sensitive to learning rates.
