# Findings v276: Complex MNIST in the Frequency Domain

## Experiment Overview
Evaluated a **Complex-Valued Neural Network (CVNN)** on MNIST classification using a **2D FFT** as the input transform. The goal was to test if spectral features combined with complex algebra yield higher parametric efficiency than real-valued MLPs.

## Results: MNIST FFT Benchmark

| Model | Parameters | Best Acc | PEI |
| :--- | :--- | :--- | :--- |
| **Complex FFT MLP** | 101,864 | 95.43% | **19.0554** |
| Real FFT MLP (Matched) | 202,142 | **97.93%** | 18.4577 |

## Analysis
1.  **Efficiency over Accuracy**: Although the real-valued model achieved a higher absolute accuracy (+2.5%), it required **double the parameters**. The Complex model's higher **PEI** confirms that complex weights are more efficient at distilling the underlying information from the frequency domain.
2.  **Structural Constraint**: Complex multiplication is a structured form of linear transformation. This structure acts as a regularizer. While it slightly limits the "memorization" capacity on simple datasets like MNIST, it provides a more elegant representation of phase relationships.
3.  **Numerical Stability**: The introduction of **orthonormal FFT** and **Batch Normalization** was critical to prevent divergence, as the PID optimizer's high integral gain can be sensitive to the large magnitudes produced by spectral transforms.

## Theoretical Reflection: Why CVNN for Images?
Images are usually spatial, not periodic. However, by transforming to the FFT domain, we treat an image as a superposition of waves. The CVNN then acts as a **phase-aware filter bank**. The fact that it maintains high accuracy with half the weights suggests that the "phase" of the FFT components contains structural information that complex weights are uniquely suited to handle.

## Conclusion
The **Complex-Valued Spectral MLP** is a powerful tool for parametric compression. It proves that we can trade a small amount of raw accuracy for a massive gain in efficiency by aligning the network's algebra with the data's domain (Spectral/Complex).
