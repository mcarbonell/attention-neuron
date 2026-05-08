# Blueprint: Stage-Gating Pre-training

## Vision
Reduce the massive computational cost of training Large Language Models (LLMs) from scratch by using a phased approach where the model's structure is stabilized by frozen weights while its "attention" is tuned via gating.

## Phase 1: Structural Warmup (Frozen Weights)
- **Architecture**: Standard Transformer layers but with a learnable scalar/vector gate after every linear projection ($y = (xW + b) \odot g$).
- **Initialization**: Orthogonal or Spectral initialization for weights.
- **Training**: Only gating parameters are updated.
- **Goal**: Force the gates to discover useful combinations of the initial random "basis functions". Establish basic token-to-token correlations and grammatical structures.
- **Benefit**: 99.9% reduction in optimizer states, 40% reduction in backward pass FLOPs, 0% weight drift.

## Phase 2: Bias & Normalization Tuning
- **Action**: Unfreeze biases and Normalization parameters (LayerNorm/RMSNorm).
- **Goal**: Refine the distribution of activations across the network.
- **Benefit**: Still extremely lightweight, but allows the model to shift the "center" of its representations.

## Phase 3: Full-Res Specialized Training
- **Action**: Unfreeze all weights.
- **Goal**: High-fidelity fine-tuning. The model starts from a coherent state instead of chaos.
- **Benefit**: Potentially faster convergence and better final accuracy due to a "cleaner" starting point.

## Synergies with Spectral Architectures
Applying this to the repository's spectral models:
- **Walsh-Hadamard Transforms** can act as the frozen, matrix-free core.
- Gating acts as the frequency-domain selector.
- This combination would be the ultimate in **Parametric Efficiency (PEI)**.
