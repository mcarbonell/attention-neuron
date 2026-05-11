# Findings v267: Systematic PID Hyperparameter Sweep

## Experiment Overview
A systematic grid search was performed over the PID optimizer hyperparameters on a Standard MLP (MNIST) to identify the optimal balance between Proportional, Integral, and Derivative components.

**Hyperparameter Grid:**
- **Kp (Proportional):** [0.1, 1, 10]
- **Ki (Integral):** [1, 10, 100]
- **Kd (Derivative):** [0.1, 1, 10]
- **Total Combinations:** 27

## Results Summary
- **Best PID Configuration:** `PID(Kp=1, Ki=100, Kd=10)`
- **Best PID Accuracy:** **98.27%**
- **Adam Baseline Accuracy:** 97.63%
- **Absolute Improvement:** **+0.64%**
- **PEI (Parametric Efficiency Index):** 17.15 (PID) vs 17.04 (Adam)

## Key Observations

### 1. The Dominance of Integral Gain (Ki)
The most significant factor in performance was the high integral gain. 
- All configurations with **Ki=100** achieved accuracy > 98%.
- Configurations with Ki=1 or Ki=10 struggled to match Adam's baseline in most cases, except when Kp was very high (Kp=10).
- This confirms the "Oligarchy Hypothesis" in a different light: high inertia (memory of past gradients) is essential for escaping local minima in these architectures.

### 2. Damping via Derivative Gain (Kd)
While Ki drove the accuracy up, Kd acted as a stabilizer.
- For the best Ki=100 setting, increasing Kd from 0.1 to 10 provided a consistent boost.
- `PID(Kp=1, Ki=100, Kd=0.1)` -> 98.04%
- `PID(Kp=1, Ki=100, Kd=10)` -> 98.27%
- The derivative component successfully damped the oscillations typically introduced by extreme integral gains.

### 3. Efficiency and Overhead
- **Computational Cost:** The `internal_overhead_time` for PID (~74.3s for the best run) was nearly identical to Adam (~74.29s).
- **Inference:** PID provides a "free" accuracy boost without increasing the number of parameters or significantly slowing down the training loop compared to standard optimizers.

## Visual Breakdown (Top 3)
| Rank | Configuration | Accuracy | PEI |
| :--- | :--- | :--- | :--- |
| 1 | **Kp=1, Ki=100, Kd=10** | **98.27%** | **17.15** |
| 2 | Kp=1, Ki=100, Kd=1 | 98.24% | 17.14 |
| 3 | Kp=0.1, Ki=100, Kd=1 | 98.24% | 17.14 |
| - | Adam (Standard) | 97.63% | 17.04 |

## Conclusion
The Industrial PID approach, specifically with **high integral gain and strong derivative damping**, significantly outperforms Adam on standard MLP architectures. The stability provided by Kd allows for the use of extreme Ki values that would otherwise cause divergence.

**Next Step Recommendation:** Test this specific `(1, 100, 10)` configuration on more complex tasks (CIFAR-10) or more exotic architectures (Spectral/Gated) to see if the superiority holds as task complexity scales.
