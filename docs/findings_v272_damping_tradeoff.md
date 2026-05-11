# Findings v272: The Damping Trade-off

## Experiment Overview
Tested the "Stabilizer" configuration (`Ki=1000, Kd=10`) on the widened StandardCNN (CIFAR-10) to address the mid-training turbulence observed in v271.

## Key Results
- **Peak Accuracy**: **79.27%** (Epoch 11/16)
- **Stability**: Significantly improved. The model stayed consistently within the 78.5% - 79.3% range after the initial convergence phase.
- **Comparison**: While more stable, it failed to reach the 80.41% peak of the less-damped v271.

## Analysis: The Cost of Control
The experiment confirms a fundamental trade-off in the Industrial PID optimizer:
1.  **Damping Efficiency**: Increasing `Kd` from 2 to 10 successfully prevented the "elastic" drops. The recovery from the Epoch 9 dip was nearly instantaneous.
2.  **Peak Suppression**: The high damping acted as a "soft limiter," preventing the optimizer from making the aggressive moves needed to reach the highest local accuracy peaks.
3.  **Stability-Accuracy Frontier**: There is a frontier between maximum speed/accuracy (v271) and maximum reliability/stability (v272).

## Conclusion
A static PID configuration is limited by the trade-off between power and control. To break the 81% barrier, a dynamic approach is required: high gain for initial discovery and high damping for final settlement.

**Next Step**: Implement a hybrid scheduler in v273.
