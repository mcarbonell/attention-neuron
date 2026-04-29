# V99c: Omni-View Triangular Attention (Final Results)

## Abstract
Expansion of the multi-view experiment to 5 views: Rows, Cols, Diag+45, Diag-45, and Spiral. This test validates the hypothesis that 1D geometric neurons can capture 2D structure if provided with enough spatial perspectives.

## Final Results
| Config | Peak Accuracy | Final Accuracy | Stability |
| :--- | :--- | :--- | :--- |
| **v99c (5 Views)** | **84.21%** | 75.89% | Low |
| v99b (3 Views) | 78.21% | 64.14% | Medium-Low |

## View Distribution (Layer 1)
- **Rows**: 338 neurons (33%)
- **Cols**: 173 neurons (17%)
- **Diag+45**: 205 neurons (21%)
- **Diag-45**: 147 neurons (15%)
- **Spiral**: 161 neurons (16%)
*(Note: Diagonals together account for 36% of the attention).*

## Observations
- **Performance**: Reached the highest accuracy of the series (**84.21%**) proving that multi-view integration is superior even under extreme parameter constraints.
- **The "Cliff" Effect**: The catastrophic drop to 11.35% in epoch 19 indicates a sensitivity to parameter updates. As widths shrink, the neurons can lose signal entirely, causing numerical instability in BatchNorm.
- **Slanted Features**: The high distribution in diagonal views suggests that these are more informative for digit recognition than simple column scans.
