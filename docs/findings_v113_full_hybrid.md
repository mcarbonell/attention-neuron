# Findings v113: Full Morph-Spectral Hybrid (Stability Fix)

## Experiment Summary
We attempted to fuse all features (Islands + Intensity + Pixels) into a single hybrid model. We also addressed the gradient instability of Triangular neurons by enforcing a minimum width (`0.02`).

## Results

| Model | Representation | Parameters | Test Accuracy |
|-------|----------------|------------|---------------|
| **Hybrid v111** | Raster + Islands | 3,850 | **94.20%** |
| **Hybrid v113** | Raster + Islands + Intensity | 5,386 | 93.32% (Train) / 93.01% (Test) |

## Key Insights
1.  **Redundancy Penalty**: Adding "Intensity" features to the structural path didn't improve accuracy, despite increasing the parameter count. This confirms that for ultra-compact models, "less is more". The network seems to struggle when too many heterogeneous features are packed into a single small layer.
2.  **Triangular Stability**: The minimum width constraint successfully eliminated the accuracy jumps (stochastic instability) observed in the initial runs of v113. 
3.  **Spectral Resolution**: Increasing $k$ to 32 didn't yield the expected gains, possibly because $k=16$ already captures the "low-pass" skeleton of the MNIST digits sufficiently.

## Conclusion
The **v111** architecture (Raster + Islands, $k=16$) remains the optimal efficiency-to-performance point.
