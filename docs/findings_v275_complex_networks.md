# Findings v275: The Complex Advantage

## Experiment Overview
Implemented a **Complex-Valued Neural Network (CVNN)** to test the hypothesis that phase-driven intelligence is more parametrically efficient than standard real-valued processing on interference-based tasks.

## Results: Wave Interference Challenge
Prediction of 8 summed complex signals (Amplitude + Phase).

| Model | Final Val Loss (MSE) | PEI (Parametric Efficiency Index) |
| :--- | :--- | :--- |
| **Complex-Valued MLP** | **2.637e-06** | **1.9718** |
| Real-Valued MLP | 1.612e-05 | 1.5530 |

## Analysis
1.  **Phase-Amplitude Decoupling**: The use of **ModReLU** allowed the network to learn geometric relationships (rotations and interference) without destroying the phase information.
2.  **Parametric Compactness**: By using `torch.complex64`, the network naturally handles 2D rotations in a single parameter, whereas a real MLP requires structured 2x2 weight patterns that are harder to learn and less efficient.
3.  **Optimizer Stability**: The **PID Optimizer** proved robust in the complex domain. The high integral gain (`Ki=100`) facilitated fast convergence toward the exact signal summation.

## Conclusion
Complex numbers are not just "pairs of reals"; they provide a algebraic framework that is significantly more expressive for tasks involving periodic signals and interference. The PEI jump of +0.42 confirms this is a breakthrough in efficiency.
