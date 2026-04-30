# Walkthrough: Efficiency and Invariance Era

## Overview
This session focused on maximizing the classification efficiency of MNIST using ultra-lightweight architectures (<5k parameters) by combining morphological features (PAC) with spectral weight synthesis (Walsh/DCT). We also explored mathematical rotation invariance through circular spectral sampling.

## Key Accomplishments

### 1. Feature Fusion and Disk Caching
Implemented a feature extraction pipeline for **Island Signatures** and **Intensity Profiles**, with a disk caching system to accelerate experimentation.

### 2. The Hybrid Breakthrough (v110-v111)
-   Discovered the synergy between **Triangular Neurons** and **Islands** (Structural Path) and **Walsh Neurons** and **Pixels** (Spectral Path).
-   Achieved **93.03%** accuracy with only **1,290 parameters**.
-   Scaled to **94.20%** with **3,850 parameters**.

### 3. Infinite Resolution Sweep (v117)
Validated the property that spectral neurons do not increase parameter counts with input resolution. Tested up to **32,768 samples** per image, identifying the "spectral bottleneck" where $k$ must scale with resolution to maintain accuracy.

### 4. Rotation Invariance (v118-v119)
-   **Spectral Rings (v118)**: Achieved perfect mathematical rotation invariance (0° to 180°) using FFT Magnitude on concentric rings.
-   **The Invariant King (v119)**: Balanced 92% accuracy with high rotation resistance by fusing invariant, structural, and orientation paths into a single **3.3k parameter** model.

## Experiments & Findings
- [findings_v107_feature_fusion.md](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/docs/findings_v107_feature_fusion.md)
- [findings_v109_neuron_comparison.md](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/docs/findings_v109_neuron_comparison.md)
- [findings_v110_tri_walsh_hybrid.md](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/docs/findings_v110_tri_walsh_hybrid.md)
- [findings_v111_hybrid_scaled.md](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/docs/findings_v111_hybrid_scaled.md)
- [findings_v112_spiral_hybrid.md](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/docs/findings_v112_spiral_hybrid.md)
- [findings_v113_full_hybrid.md](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/docs/findings_v113_full_hybrid.md)
- [findings_v115_invariant_spiral.md](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/docs/findings_v115_invariant_spiral.md)
- [findings_v117_infinite_samples.md](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/docs/findings_v117_infinite_samples.md)
- [findings_v118_spectral_rings.md](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/docs/findings_v118_spectral_rings.md)
- [findings_v119_invariant_hybrid.md](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/docs/findings_v119_invariant_hybrid.md)

## Conclusion
This session successfully transitioned the project from "general spectral exploration" to "high-efficiency morphological hybrids," proving that domain-specific neuron types (Triangular/Walsh) can outperform dense MLPs by factors of 10x-20x in parameter efficiency.
