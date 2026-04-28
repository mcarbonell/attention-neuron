# Findings V82: DCT Image Reconstruction

## Overview
This experiment demonstrates the generative capability of a single "DCT Neuron". Instead of using the neuron for classification, we optimize its internal frequency coefficients to reconstruct a specific MNIST target image.

## Methodology
- **Target**: A single random image from the MNIST dataset.
- **Parameters**: $8 \times 8 = 64$ learnable DCT coefficients (compared to $28 \times 28 = 784$ pixels, a 12.25x compression).
- **Process**:
    1. Initialize 64 random coefficients.
    2. Pad coefficients to $28 \times 28$ with zeros (zero-filling high frequencies).
    3. Perform Inverse 2D-DCT to transform back to pixel space.
    4. Minimize Mean Squared Error (MSE) between the reconstruction and the original.
- **Optimizer**: Adam (LR=0.1) for 500 epochs.

## Results
- **Final MSE Loss**: ~0.0347 (with $8 \times 8$ coefficients).
- **Visualization**: The reconstructed image captures the global structure and "smooth" features of the digit, while losing high-frequency details (sharp edges).
- **Insight**: This confirms that DCT parameters serve as an efficient "recipe" for generating images. In an Attention Neuron context, these coefficients represent the **basis** or **receptive field** of the neuron.

## Conclusion
The DCT basis provides a natural inductive bias for image-like data. By learning only the low-frequency components, the neuron focuses on the most semantic part of the signal, effectively acting as a learned low-pass filter that "knows" how to represent digits.

**Comparison saved in**: `results/dct_reconstruction_test.png`
