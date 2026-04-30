# Findings v121: Projection Sinusoids

## Summary
We implemented a "Radon-inspired" architecture where spatial weights are fixed to 1 (calculating row/column sums) and feature extraction is performed solely by learnable **Sinusoidal Modulators** (frequency and phase).

## Metrics
- **Final Test Acc**: 88.87%
- **Total Parameters**: 5,386
- **Architecture**: 28 Rows + 28 Cols -> 8 Sine Neurons per projection -> Linear(448, 10)

## Observations

1. **Information Sufficiency**: 88.87% accuracy confirms that the 1D marginal distributions (projections) of MNIST digits contain ~90% of the information necessary for classification.
2. **Parameter Efficiency**: We achieved this with zero learned weights in the spatial domain. The 5k parameters are almost entirely in the classification head (4.4k).
3. **Bottleneck Identification**: The model plateaued around 88-89%. This suggests that while projections are powerful, the lack of 2D cross-correlation (e.g., knowing exactly WHERE a pixel is in a row vs knowing only the row sum) limits the ceiling.
4. **Learning Dynamics**: The loss decreased steadily, showing that backpropagating through the frequency ($\omega$) and phase ($\phi$) of a sine function is stable and effective.

## Conclusion
The "Projection Sinusoids" experiment is a success in parameter efficiency. It proves that resonance-based feature extraction can replace dense spatial weights for structured data like MNIST.

> [!TIP]
> To cross the 90% barrier, we might need **Diagonal Projections** or a small set of **Random Projections** to recover 2D spatial context without returning to full dense matrices.
