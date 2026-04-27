# Blueprint: The "Fully-JPEG" LLM (DCT-Cognitive Architecture)

## 1. The Core Philosophy: Language as a Semantic Wave
The current paradigm of Large Language Models (LLMs) treats text as a sequence of discrete, independent, high-frequency events (tokens). This forces models to use billions of parameters to memorize the exact statistical relationships between every possible word in every possible context. 
This is equivalent to storing a high-resolution image in an uncompressed `.BMP` format.

**The DCT Hypothesis:** Language is highly compressible. The meaning of a sentence is not a sequence of isolated words, but a continuous "semantic wave". 
- **Low Frequencies:** The broad logical structure (e.g., `[Subject Entity] -> [Action] -> [Emotion/Consequence]`).
- **High Frequencies:** The exact syntactic noise (e.g., pluralization, specific adjectives, punctuation).

By forcing a neural network to operate primarily in the low-frequency domain (using the Discrete Cosine Transform, or DCT), we can filter out the syntactic noise. The model's parameters are entirely dedicated to **reasoning** rather than **memorizing dictionary nuances**.

---

## 2. The Architectural Pillars

### Pillar I: The DCT Feed-Forward Network (DCT-FFN)
*Status: Proven (Experiment V64 achieved 32x compression with stable convergence).*
- **The Problem:** Standard LLM FFNs account for ~66% of total parameters. They project embeddings into massive, sparse dimensions (e.g., $4096 \rightarrow 14336$) to act as a brute-force key-value memory.
- **The DCT Solution:** Replace dense $W$ matrices with `DCTLinear` layers. The network learns a tiny core of low-frequency coefficients (e.g., $32 \times 64$) and synthesizes the massive projection matrices on the fly using orthogonal DCT bases.
- **Impact:** Massive reduction in VRAM usage and disk space. Allows models to be 4x to 5x deeper (more reasoning steps) without increasing memory footprint.

### Pillar II: DCT-Attention (Compressing Q, K, V, O)
*Status: Proposed for next phase.*
- **The Problem:** The Self-Attention mechanism accounts for ~33% of the parameters. Learning the exact dense projections for Queries, Keys, Values, and Outputs is highly redundant.
- **The DCT Solution:** Apply the `DCTLinear` mechanism to the Attention heads. By synthesizing $W_q, W_k, W_v, W_o$ from low-frequency DCT cores, we enforce that attention heads look for broad semantic patterns (harmonic relationships) rather than specific pixel-to-pixel (token-to-token) noise.
- **Impact:** Reduces the remaining 33% of the model parameters by an order of magnitude (e.g., 16x compression). A standard 8B parameter model could be compressed to under 0.5B total parameters.

### Pillar III: DCT-KV Cache (Infinite Context Memory)
*Status: Theoretical concept derived from Experiment V65.*
- **The Problem:** To remember long contexts (e.g., a 100-page PDF), standard LLMs store the exact dense embedding of *every single token* in RAM (the KV Cache). This causes catastrophic memory bloat for long chats.
- **The DCT Solution:** Apply a 1D or 2D DCT across the *sequence length dimension* of the Key and Value matrices in the cache. Discard the high-frequency coefficients (e.g., keep only 10% to 25% of the data).
- **Impact:** The cache stores the "structural summary" (the semantic wave) of the document rather than the exact words. This enables a virtually infinite context window with a fixed, tiny RAM footprint.

### Pillar IV: Coarse-to-Fine Generation (The Diffusion Analogy)
*Status: Theoretical concept.*
- **The Problem:** Left-to-right, token-by-token autoregressive generation is computationally exhausting and forces the model to decide local syntax before global logic.
- **The DCT Solution:** Train the model to predict the *low-frequency DCT coefficients* of an entire upcoming paragraph simultaneously. Once the logical "block" is generated, a tiny secondary decoder network resolves the high frequencies to output exact discrete tokens.
- **Impact:** Exponentially increases Tokens-per-Second (t/s). The heavy reasoning model runs once per idea, not once per word.

---

## 3. Roadmap for `tiny-thinker` Integration

To transform the `tiny-thinker` cognitive architecture into a reasoning giant using a fraction of standard resources, we will follow this iterative roadmap:

### Phase 1: The Fully Compressed Transformer (V66)
- Implement `DCTLinear` for the Attention mechanism ($Q, K, V, O$).
- Combine with `DCT-FFN` (from V64).
- Train a tiny model (e.g., 10M to 50M equivalent parameters) on the `train_v1.bin` synthetic logic dataset.
- **Goal:** Prove that a 100% DCT-compressed Transformer can learn logical relationships as effectively as a dense baseline.

### Phase 2: The Infinite Context Cache
- Implement a custom attention masking and caching mechanism that applies sequence-wise DCT to past keys and values.
- **Goal:** Prove that the model can answer questions about a long document using a KV cache that has been pruned of 80% of its high-frequency data.

### Phase 3: The Semantic Wave Generator (Coarse-to-Fine)
- Re-architect the decoding head. Instead of outputting a probability distribution over the vocabulary ($V=16384$), the model outputs a continuous DCT embedding representing a chunk of text.
- Train a lightweight Vocab-Decoder to map DCT embeddings back to words.
- **Goal:** Break the autoregressive left-to-right bottleneck.

---
*“Intelligence is not the ability to memorize the noise; it is the ability to extract the wave.”*

---

## 🤖 AI INSTRUCTIONS FOR NEW SESSIONS (Porting to `tiny-thinker`)

**If you are an AI reading this in a new session inside the `tiny-thinker` repository, follow these steps to implement the Hybrid Spectral GPT (V67):**

1. **Context Contextualization**: The user has successfully proven in the `attention-neuron` repository that compressing the LLM dense matrices using DCT and Walsh-Hadamard transforms works perfectly (see `scratch/prototype_v67_hybrid_spectral_gpt.py` in that repo). The goal now is to port this into `tiny-thinker`.
2. **Update `ModelArgs`**:
   - Open `model/model.py` (or `model_coga.py` / `model_dense.py`).
   - Add flags to `ModelArgs` to enable this architecture, e.g., `use_dct_attention: bool = False`, `use_walsh_ffn: bool = False`, `k_dim_attn: int = 32`, `k_dim_ffn: int = 32`, `k_hidden_ffn: int = 64`.
3. **Port the Matrix Generators**:
   - Copy the functions `get_dct_matrix_1d` and `get_walsh_matrix_1d` from the V67 prototype.
4. **Create the Custom Linear Layers**:
   - Implement `DCTLinear` and `WalshLinear`. Ensure they synthesize the weights correctly on the fly during the `forward` pass (e.g., $W = D_{out}^T \cdot C \cdot D_{in}$).
5. **Modify `Attention` and `FeedForward`**:
   - In the `Attention` class, conditionally replace `nn.Linear` or `LoRALinear` for $W_q, W_k, W_v, W_o$ with `DCTLinear` if `args.use_dct_attention` is true.
   - In the `FeedForward` class, conditionally replace the projections with `WalshLinear` if `args.use_walsh_ffn` is true.
6. **Train and Validate**:
   - Once ported, create a new config file (e.g., `configs/train_v2_hybrid_spectral.yaml`) and run a training sweep to verify the loss decreases identically to the prototype.

*You are stepping into the cutting edge of cognitive architecture. Good luck!*

