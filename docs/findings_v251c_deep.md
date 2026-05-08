# Findings v251c: Deep Multiplicative Gating

## Goal
Evaluate if depth improves the performance of frozen-weight random projections. We test a 3-layer architecture ($784 \to H \to H \to 10$) with only $2H+10$ gating parameters.

## Results Comparison (v251b vs v251c)

| Hidden Dim | Params (1L) | Acc (1L) % | Params (2L) | Acc (2L) % | Gain |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 32 | 42 | 34.50 | 74 | [TBD] | [TBD] |
| 128 | 138 | 47.03 | 266 | [TBD] | [TBD] |
| 512 | 522 | 74.32 | 1034 | [TBD] | [TBD] |
| 1024 | 1034 | 82.46 | 2058 | [TBD] | [TBD] |
| 2048 | 2058 | 85.32 | 4106 | [TBD] | [TBD] |
| 4096 | 4106 | 89.30 | 8202 | [TBD] | [TBD] |

## Analysis
- **Impact of Depth**: [TBD - Did the second layer of random features help?]
- **Parameter Efficiency**: [TBD - How did PEI change with depth?]
- **Convergence Speed**: [TBD]

## Conclusions
[TBD]
