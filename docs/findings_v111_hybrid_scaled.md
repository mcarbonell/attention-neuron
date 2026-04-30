# Findings v111: Scaled Tri-Walsh Hybrid (H=96)

## Experiment Summary
We scaled the Hybrid Tri-Walsh architecture (v110) from 32 to 96 hidden units per path to assess the scaling potential of the "Cerebro-Cerebelo" approach.

## Results

| Model | Parameters | Test Accuracy | Improvement |
|-------|------------|---------------|-------------|
| Hybrid v110 (H=32) | 1,290 | 93.03% | - |
| **Hybrid v111 (H=96)** | **3,850** | **94.20%** | **+1.17pp** |

## Key Insights
1.  **Diminishing Returns on Width**: Tripling the parameters only yielded a ~1.2% accuracy gain. This suggests that simply making the hybrid paths wider has limits. The structural bottleneck is likely in the representation itself or the spectral core size ($k=16$).
2.  **Stability**: The model is extremely stable during training, reaching 90% accuracy by Epoch 3.
3.  **Efficiency Milestone**: At 3.8k parameters, we are only 1.9% away from the baseline MLP (which uses 25k parameters).
