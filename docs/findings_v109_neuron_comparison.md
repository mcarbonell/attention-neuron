# Findings v109: Cross-Neuron and Representation Comparison

## Experiment Summary
We conducted a large-scale comparison (16 configurations) between 4 types of neurons (MLP, Triangular, DCT, Walsh) and 4 types of input representations (Intensity, Islands, I+Is, Pixels). All models used 32 hidden units.

## Results Matrix (Test Accuracy %)

| Neuron Type | Intensity (57D) | Islands (56D) | I + Is (113D) | Pixels (784D) | Params (avg) |
|-------------|-----------------|---------------|----------------|---------------|--------------|
| **MLP**     | 85.96%          | 84.57%        | **91.85%**     | **96.13%**    | 2,100-25k    |
| **Triangular**| 62.56%        | **80.02%**    | 70.86%         | 70.35%        | **426**      |
| **DCT**     | 76.46%          | 83.84%        | 85.32%         | 84.07%        | 874          |
| **Walsh**   | 76.93%          | 84.30%        | 84.06%         | **86.71%**    | 874          |

## Key Insights

1.  **Triangular-Island Synergy**: The `Triangular + Islands` configuration is a breakthrough in efficiency, achieving **80.02% accuracy with only 426 parameters**. The local nature of triangular filters perfectly matches the structural local information in Island Signatures.
2.  **Spectral Power**: Walsh neurons outperformed DCT in raw pixel processing, reaching **86.71%** with 874 parameters. This suggests that the binary-like nature of Walsh basis functions is better suited for the "stroke-based" structures of MNIST than the smooth cosines of DCT.
3.  **Representation Complementarity**: While `Intensity + Islands` (113D) is the most efficient dense representation for MLPs (91.85%), it doesn't necessarily translate to better results for specialized neurons like Triangular, which prefer "pure" structural data (Islands).

## Conclusion
Specialized neurons are not just parameter-efficient; they are **representation-sensitive**. To maximize efficiency, we should match the neuron's mathematical bias (e.g., local for Triangular, sequency-based for Walsh) with the appropriate data representation.

## Next Steps: Experiment v110
Implement a **Hybrid Model** that combines:
-   A **Triangular path** for Islands (Structural fast-path).
-   A **Walsh path** for Pixels (Spectral detail-path).
Target: >90% accuracy with <1,500 parameters.
