# Blueprint: Stage-Gating Pre-training

## Vision
Reduce the massive computational cost of training Large Language Models (LLMs) from scratch by using a phased approach where the model's structure is stabilized by frozen weights while its "attention" is tuned via gating.

## Phase 1: Structural Warmup (Frozen Weights)
- **Architecture**: Standard Transformer layers but with a learnable scalar/vector gate after every linear projection ($y = (xW + b) \odot g$).
- **Training**: 1-2 Epochs. Only Gating parameters are updated.
- **LR**: High (e.g., 0.05).
- **Goal**: Force the gates to discover the "Lottery Ticket" within the random projection. Establish basic feature selection.

## Phase 2: Weight Refinement (Frozen Gating + Layer-wise)
- **CRITICAL RULE**: **Gating must be FROZEN** during this phase. Training both simultaneously causes extreme instability (destructive interference).
- **Action**: Update weights layer-by-layer in a rotating cycle (Epoch-wise Round-Robin).
- **LR**: Low (e.g., 0.001).
- **Goal**: Refine the selected features without disrupting the global structure established by the gating.
- **Benefit**: 60-80% reduction in weight-gradient compute per epoch, high stability, and near-SOTA performance.

## Phase 3: Final High-Res Fine-Tuning
- **Action**: Unfreeze all weights (optional).
- **Goal**: Final polish.

## Synergies with Spectral Architectures
Applying this to the repository's spectral models:
- **Walsh-Hadamard Transforms** act as the frozen, matrix-free core.
- Gating acts as the frequency-domain selector.
- Layer-wise training ensures that spectral components are tuned sequentially.
