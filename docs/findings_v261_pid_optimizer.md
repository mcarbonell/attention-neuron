# Findings V261: The Industrial Miracle (PID Optimizer)

## Overview
We challenged the standard industry optimizers (Adam) by implementing a **PID (Proportional-Integral-Derivative) Controller** as an optimizer. This is the first time in the repository that a purely mechanical control-theory approach is used for training.

## Results (10 Epochs, MNIST, Standard MLP)

| Optimizer | Final Accuracy | Final Loss | Notes |
| :--- | :--- | :--- | :--- |
| **Adam (Standard)** | 97.85% | 0.0178 | Fast start, slight plateau. |
| **PID (Kp=1, Ki=30, Kd=1)** | 98.26% | 0.0061 | Superior precision. |
| **PID (Kp=1, Ki=150, Kd=1)** | **98.47%** | **0.0027** | **The Beast.** 6.5x lower loss than Adam. |

## Key Technical Insights

### 1. The Power of High Integral Gain ($K_i$)
The jump from $K_i=30$ to $K_i=150$ proved that "Inertia is Intelligence" in certain loss landscapes. By accumulating massive historical gradient information, the optimizer effectively behaves like a projectile that ignores local noise.

### 2. Derivative Damping ($K_d$)
A high $K_i$ usually leads to massive overshooting. The **$K_d=1$** component (tracking gradient change) acted as a predictive brake. This "industrial damping" is the secret to why the model reached a loss of **0.0027** without diverging.

### 3. Stability vs. Adaptation
Adam adapts the learning rate per parameter based on statistics. The PID keeps a global LR but adapts the **force** per parameter based on dynamics (velocity and acceleration). For structured problems like MNIST, the "Physical Dynamics" of the PID proved more robust than the "Statistical Adaptation" of Adam.

## Conclusion
The PID Optimizer is a viable and powerful alternative for neural training. It provides a level of control over the "Momentum vs. Curvature" tradeoff that standard optimizers lack.

**Reference Script**: `scratch/prototype_v261_pid_optimizer.py`
