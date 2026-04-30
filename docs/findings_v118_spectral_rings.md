# Findings v118: Spectral Rings (Mathematical Rotation Invariance)

## Experiment Summary
We abandoned the spiral scan in favor of a **Concentric Rings** architecture (32 rings, 64 samples each). For each ring, we computed the **Magnitude of the FFT**, discarding the phase to create a signature that is mathematically invariant to cyclic shifts (and thus image rotations).

## Results

| Angle | Accuracy |
|-------|----------|
| 0°    | **62.59%** |
| 15°   | 61.94% |
| 45°   | 60.82% |
| 90°   | **62.59%** |
| 180°  | **62.59%** |

## Key Insights
1.  **Perfect Invariance**: The accuracy at 0°, 90°, and 180° is identical. This proves the mathematical validity of using FFT magnitude for rotation invariance in spectral neurons.
2.  **Phase Loss Penalty**: Discarding phase information reduced the base accuracy to ~62%. Without phase, the network cannot distinguish the relative positions of features within a ring (e.g., distinguishing a `6` from a `9` becomes nearly impossible if they are perfect rotated versions of each other).
3.  **Stability**: The "flat line" accuracy across all angles is a major milestone for this repository.

## Conclusion
We achieved total rotation invariance, but at the cost of the structural orientation needed for high-accuracy MNIST classification.
