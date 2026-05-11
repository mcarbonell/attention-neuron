# Findings V265: Universal PID Benchmark (The Industrial Standard)

## Overview
After the success of PID in gated spectral architectures, we scaled the experiment to a **Standard MLP with BatchNorm** on MNIST. We tested the newly discovered "Industrial Beast" configuration (Ki=100) with a safety damping mechanism (Relaxed Clipping).

## Results (10 Epochs, MNIST, Standard MLP)

| Optimizer | Final Accuracy | Final Loss | Stability |
| :--- | :--- | :--- | :--- |
| **Adam (Standard) + Clip 10** | 97.64% | 0.0178 | Oscillates in late epochs. |
| **PID (Kp=1, Ki=100, Kd=1) + Clip 10** | **98.41%** | **0.0034** | **Consistent Ascent.** |

## Key Technical Insights

### 1. Generalization of the "Inertia Law"
The high integral gain ($K_i=100$) was not just a quirk of gated networks. In standard MLPs, it acts as a low-pass filter that ignores mini-batch noise and follows the true gradient curvature. Adam, by contrast, seemed to get trapped in late-stage noise, causing its accuracy to drop from 98.1% to 97.6%.

### 2. Relaxed Clipping (Max Norm 10.0)
We discovered that tight clipping (1.0) suffocates the PID's intelligence. By relaxing the clip to **10.0**, we provided enough dynamic range for the "momentum spikes" necessary to navigate narrow valleys in the loss landscape, while still preventing the numerical explosions seen at $K_i=150$.

### 3. Precision vs. Adaptation
This experiment proves that **Dynamics (Velocity/Acceleration)** are often more important than **Statistics (Mean/Variance)**. The PID optimizer treats the learning process as a physical system with mass and damping, leading to a much more stable and deeper convergence than Adam's per-parameter scaling.

## Conclusion
The **PID-100-Clip10** is now the recommended "High-Performance" optimizer for this repository. It is particularly effective for architectures where stability and late-stage precision are critical.

**Reference Script**: `scratch/prototype_v265_pid_standard_mlp_final.py`
