# Phase-Coded Delta Memory  
## Online Associative Recall with a Fixed-Norm Trigonometric Feature Map

**Author:** Mario Raul Carbonell Martinez  
**Repository:** [Attention Neuron](https://github.com/attention-neuron)  
**Experiments:** v298–v300  
**Status:** Working research note — synthetic evidence, baseline validation in progress

---

## Abstract

We study an \(O(L)\) recurrent associative memory that combines:

1. learned unit-magnitude complex phasor keys and queries;
2. a fixed-size fast-weight matrix;
3. residual-error writes based on the Delta rule; and
4. a short causal local mixer for forming nearby key-value associations.

In one synthetic multi-query associative recall (MQAR) harness, a three-layer phase-coded Delta model achieved a best observed accuracy of **99.95%**, matching the corresponding causal softmax-attention control while maintaining a recurrent state whose size is independent of sequence length.

The complex recurrence admits an exact real-valued representation. A \(d_v\times d_k\) complex state is equivalent to a \(d_v\times 2d_k\) real Delta memory operating on the fixed-norm feature map

\[
\phi(\theta)=
\begin{bmatrix}
\cos\theta\\
\sin\theta
\end{bmatrix}.
\]

Consequently, the current evidence does not establish that complex arithmetic is intrinsically superior to real-valued memory. A more precise working hypothesis is that the angular parametrization helps by providing bounded, fixed-norm features, a favorable allocation of state capacity between key and value dimensions, and potentially better-conditioned learned Gram matrices.

The primary unresolved experiment is therefore a comparison against a real rectangular \(d_v\times2d_k\) Delta memory with identical local mixing, gating, normalization, parameter budget and optimization.

The reported results establish a promising synthetic mechanism. They do **not yet** establish superiority over modern linear-attention architectures or transfer to natural-language modeling.

---

## 1. Scope of the current claim

### What has been observed

- A phase-coded Delta memory solves the tested short-range MQAR task.
- In the v298 harness, its best observed accuracy matches the softmax-attention control.
- Its recurrent work is linear in sequence length \(L\) for fixed state dimensions.
- Its inference state does not grow with context length.
- Preliminary capacity experiments suggest graceful degradation as the number of stored associations increases.

### What has been established algebraically

- The current complex implementation has an exact real-valued representation.
- Unit phasors induce a bounded, fixed-norm trigonometric feature map.
- A Delta update fits the current association by a known amount and perturbs other associations according to their feature-space similarity.
- Under independent random phases, pairwise interference has variance \(1/(2d_k)\).

### What has not yet been established

- That complex arithmetic itself is responsible for the empirical advantage.
- That phase-coded features outperform equally sized normalized real features.
- That the mechanism outperforms a canonical, properly tuned DeltaNet or Gated DeltaNet.
- That the MQAR result transfers to natural language.
- That the current implementation is faster in wall-clock time than optimized attention kernels.
- That the observed capacity gap survives isoparameter, iso-FLOP and identical-stack controls.

---

## 2. Phase-coded Delta memory

Let

\[
M_t\in\mathbb C^{d_v\times d_k}
\]

be a recurrent fast-weight state. At position \(t\), the network produces:

\[
\theta_t^k,\theta_t^q\in\mathbb R^{d_k},
\qquad
v_t\in\mathbb R^{d_v},
\qquad
\beta_t\in[0,1].
\]

Keys and queries are mapped to unit phasors:

\[
k_t=e^{i\theta_t^k},
\qquad
q_t=e^{i\theta_t^q}.
\]

Every component has unit magnitude:

\[
|k_{t,r}|=|q_{t,r}|=1.
\]

The pre-write prediction for the current key is

\[
\hat v_t
=
\frac{1}{d_k}
\operatorname{Re}\left(M_{t-1}k_t^*\right).
\]

The residual is

\[
e_t=v_t-\hat v_t.
\]

The state is updated using the Delta rule:

\[
M_t
=
M_{t-1}
+
\beta_t e_t k_t^T.
\]

A query is read as

\[
y_t
=
\frac{1}{d_k}
\operatorname{Re}\left(M_tq_t^*\right).
\]

The transpose in the write and the conjugate in the read are intentional. For a matching key,

\[
k_t^T k_t^*
=
\sum_{r=1}^{d_k}|k_{t,r}|^2
=
d_k.
\]

This makes the normalization by \(d_k\) exact.

---

## 3. Exact real-valued representation

The complex formulation can be rewritten exactly in real arithmetic.

Write

\[
k=c+is,
\qquad
M=A+iB,
\]

where

\[
c=\cos\theta,\qquad s=\sin\theta
\]

and

\[
A,B\in\mathbb R^{d_v\times d_k}.
\]

Then

\[
\operatorname{Re}(Mk^*)
=
\operatorname{Re}\left[(A+iB)(c-is)\right]
=
Ac+Bs.
\]

Define the real feature map

\[
\phi(\theta)
=
\begin{bmatrix}
\cos\theta\\
\sin\theta
\end{bmatrix}
\in\mathbb R^{2d_k}
\]

and the rectangular real memory

\[
W=[A\;\;B]
\in\mathbb R^{d_v\times2d_k}.
\]

The complex read becomes

\[
\boxed{
\hat v
=
\frac{1}{d_k}W\phi(\theta).
}
\]

Likewise, the complex update

\[
M\leftarrow M+\beta e k^T
\]

is exactly equivalent to

\[
A\leftarrow A+\beta e\cos(\theta)^T,
\]

\[
B\leftarrow B+\beta e\sin(\theta)^T,
\]

or

\[
\boxed{
W\leftarrow W+\beta e\phi(\theta)^T.
}
\]

Therefore:

> The current phase-coded complex recurrence is exactly equivalent to a rectangular real Delta memory operating on a fixed-norm trigonometric feature map.

Complex arithmetic is a compact and natural notation for the mechanism, but it is not required for its representational capacity.

### 3.1 Feature norm

The trigonometric feature map has constant norm:

\[
\|\phi(\theta)\|^2
=
\sum_{r=1}^{d_k}
\left(
\cos^2\theta_r+\sin^2\theta_r
\right)
=
d_k.
\]

Unlike unconstrained real keys, the model cannot encode information by changing key magnitude. It must use angular relationships.

### 3.2 Effective feature dimension versus intrinsic dimension

The map uses \(d_k\) learned angles to generate \(2d_k\) real coordinates. These coordinates are not independent: every sine-cosine pair satisfies

\[
\cos^2\theta_r+\sin^2\theta_r=1.
\]

Thus, the features lie on the torus

\[
(S^1)^{d_k}
\subset\mathbb R^{2d_k}.
\]

The representation has \(d_k\) intrinsic angular degrees of freedom, while a linear readout can act on \(2d_k\) real coordinates. A collection of such feature vectors may have linear rank as high as \(2d_k\), despite lying on a lower-dimensional nonlinear manifold.

---

## 4. What the Delta update guarantees

Let

\[
\hat v(k)=\frac{1}{d_k}W\phi(k)
\]

and let

\[
e=v-\hat v(k).
\]

After the update

\[
W^+
=
W+\beta e\phi(k)^T,
\]

the prediction for the same key becomes

\[
\begin{aligned}
\hat v^+(k)
&=
\frac{1}{d_k}W^+\phi(k)\\
&=
\hat v(k)
+
\frac{\beta}{d_k}
e\|\phi(k)\|^2\\
&=
\hat v(k)+\beta e.
\end{aligned}
\]

Therefore,

\[
\boxed{
\hat v^+(k)
=
(1-\beta)\hat v(k)+\beta v.
}
\]

If \(\beta=1\), the current association is fitted exactly immediately after the write.

This does not mean that previous associations remain unchanged. For another query \(q\),

\[
\Delta\hat v(q)
=
\beta e\,
\kappa(k,q),
\]

where

\[
\boxed{
\kappa(k,q)
=
\frac{1}{d_k}
\phi(k)^T\phi(q)
=
\frac{1}{d_k}
\sum_{r=1}^{d_k}
\cos(\theta_{k,r}-\theta_{q,r}).
}
\]

The effect of a write on another association is therefore proportional to a learned cosine kernel.

This gives a precise interpretation:

> The Delta rule corrects the current association while perturbing previous associations according to the coherence of their trigonometric key features.

The mechanism is closely related to online normalized least-mean-squares learning, with the normalization simplified by the constant feature norm.

---

## 5. Interference under random phases

Assume, as a reference model, that phase components are independent and uniformly distributed on \([0,2\pi)\).

For two unrelated keys, define

\[
\rho_{ij}
=
\frac{1}{d_k}
\sum_{r=1}^{d_k}
\cos(\theta_{j,r}-\theta_{i,r}).
\]

Then

\[
\mathbb E[\rho_{ij}]=0
\]

and

\[
\operatorname{Var}(\rho_{ij})
=
\frac{1}{2d_k}.
\]

Under additional independence assumptions, interference from \(P\) unrelated associations has root-mean-square scale proportional to

\[
\sqrt{\frac{P}{2d_k}}.
\]

This is random-walk cancellation, but it is **not unique to complex features**. Normalized zero-mean real features in \(m\) dimensions have inner-product variance approximately

\[
\frac{1}{m}.
\]

For \(m=2d_k\), this gives the same second-order scale:

\[
\frac{1}{m}
=
\frac{1}{2d_k}.
\]

The theoretically defensible benefit of the phase representation is therefore not a generic transition from \(O(P)\) to \(O(\sqrt P)\). Both normalized real and phase features can exhibit random-walk cancellation.

The candidate advantages of phase coding are instead:

1. exact and automatic control of feature norm;
2. bounded coordinates;
3. a \(2d_k\)-coordinate nonlinear feature map generated from \(d_k\) learned angles;
4. a restricted angular optimization geometry;
5. potentially improved learned Gram conditioning;
6. a different allocation of a fixed state budget between key and value dimensions.

Because keys are learned rather than random, the random-phase calculation is only a null model. The trained Gram matrix must be measured directly.

---

## 6. Fair interpretation of the state budget

A complex state

\[
M\in\mathbb C^{32\times32}
\]

contains

\[
2\cdot32^2=2048
\]

real floating-point values.

Its exact real equivalent is not a \(45\times45\) square matrix. It is

\[
W\in\mathbb R^{32\times64},
\]

which also contains 2048 floats.

| Representation | Value dimension | Key-feature dimension | State floats |
|---|---:|---:|---:|
| Complex phase | 32 | 32 complex | 2048 |
| Exact realification | 32 | 64 real | 2048 |
| Real square baseline | 45 | 45 real | 2025 |

The existing \(45\times45\) baseline is approximately iso-state, but it allocates the budget differently:

- phase model: 64 real key features and 32 value dimensions;
- square real model: 45 key features and 45 value dimensions.

It is therefore a useful system comparison, but it does not isolate the contribution of phase.

The decisive control is a real rectangular \(32\times64\) Delta memory with the same stack. Further comparisons should separately report:

- iso-state;
- isoparameter;
- iso-FLOP;
- equal key-feature dimension;
- equal value dimension.

No single comparison captures all notions of fairness.

---

## 7. Role of the causal local mixer

The recurrent memory stores associations available at the current position. If a key and its corresponding value occur at different sequence positions, some local causal mechanism is required to combine them before writing.

A causal depthwise Conv1D with kernel size 4 is used in the current model:

```python
class ShortCausalConv1D(nn.Module):
    def __init__(self, d_model, kernel_size=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            padding=kernel_size - 1,
            groups=d_model,
        )
        self.act = nn.SiLU()

    def forward(self, x):
        length = x.size(1)
        y = self.conv(x.transpose(1, 2))
        y = y[:, :, :length].transpose(1, 2)
        return x + self.act(y)
```

The Conv1D is part of the tested architecture, but kernel size 4 has not been established as theoretically necessary. Depending on the exact MQAR serialization, a smaller local mixer may be sufficient.

The relevant claim is:

> A causal local mechanism is required to form associations distributed over nearby positions; the current implementation uses a depthwise Conv1D with kernel size 4.

Required controls include:

- no local mixer;
- kernel sizes 2, 3 and 4;
- an equivalent causal shift-and-linear mechanism;
- verification of the position where each key-value association is written;
- replacement of target tokens by sentinels to rule out leakage;
- explicit visualization of the receptive field used by each evaluated logit.

---

## 8. Experimental evidence

### 8.1 v298: system-level MQAR result

**Setting:** synthetic MQAR, \(L=64\), 8 key-value pairs, \(d_{\text{model}}=64\), 3 layers, approximately 108–118K trainable parameters.

All numbers below are best observed results from the reported learning-rate sweep. They should not be interpreted as seed-averaged estimates or confidence intervals.

| Model | Sequence work | Mechanism | Best LR | Best observed accuracy |
|---|---:|---|---:|---:|
| **DeltaPhaseHolographic** | \(O(Ld^2)\) | Conv1D + phase-coded Delta memory | \(2\times10^{-3}\) | **99.95%** |
| ElementwiseDeltaPhase | \(O(Ld)\) state update | Conv1D + diagonal phase Delta | \(8\times10^{-3}\) | 98.63% |
| CausalAttentionMHA | \(O(L^2d)\) | Conv1D + causal softmax attention | \(4\times10^{-3}\) | **99.95%** |
| PhaseSoftmaxHolographic | \(O(Ld^2)\) | Conv1D + phase scan without Delta write | \(4\times10^{-3}\) | 49.59% |

### Interpretation

This experiment demonstrates that the complete phase-coded Delta system can solve the tested MQAR instance and match the best observed accuracy of the softmax control.

It does not yet demonstrate that:

- phase is better than normalized real features;
- the Delta memory is better than canonical Gated DeltaNet;
- the result transfers to other MQAR generators;
- the result transfers to language;
- the recurrent implementation is faster in wall-clock time.

The large gap between PhaseSoftmax and DeltaPhase suggests that residual-error writing is a critical component of the system.

---

### 8.2 v299: preliminary state-budget comparison

State budget per head:

- complex phase: \(32\times32\) complex \(=2048\) floats;
- square real baseline: \(45\times45=2025\) floats.

| Model | 8 pairs | 16 pairs | 32 pairs | 64 pairs |
|---|---:|---:|---:|---:|
| **ComplexDeltaPhase** | 99.80% | 99.75% | 99.80% | **95.98%** |
| RealDeltaNetVanilla | 99.67% | 99.54% | 94.83% | 73.14% |
| CausalAttentionMHA | 99.63% | 99.77% | 99.63% | 99.73% |

This is evidence of a system-level gap under approximately equal recurrent-state storage. It is not yet evidence that the gap is caused by complex phase because:

1. the phase and real baselines allocate their state budgets differently;
2. their key-feature dimensions differ;
3. the complete architectural stacks must be verified as identical;
4. the results are not yet seed-averaged;
5. an exact real rectangular \(32\times64\) control is missing.

The appropriate current label is therefore **[PRELIMINARY SIGNAL]**, not a confirmed intrinsic advantage of complex arithmetic.

---

### 8.3 v300: preliminary capacity scaling

For the complex model with \(d_k=32\), preliminary best observed accuracies are:

| KV pairs | Sequence length | Complex phase accuracy |
|---:|---:|---:|
| 32 | 256 | 99.74% |
| 64 | 512 | 99.37% |
| 128 | 1024 | 88.47% |
| 256 | 2048 | 56.35% |

These results suggest graceful degradation beyond the short MQAR setting.

The corresponding real comparison is currently excluded from the primary claim because the experiment metadata announces an iso-state \(d_k=45\) model while the runtime output labels the executed model as `d_k=32`. The instantiated shapes and state budgets must be audited before those values are interpreted.

Required metadata for future runs:

- key dimension;
- value dimension;
- state tensor shape;
- number of real floats in the state;
- total trainable parameters;
- estimated FLOPs per token;
- exact stack enabled for each model.

---

## 9. Complexity and memory

For \(H\) heads, value dimension \(d_v\), key dimension \(d_k\), and sequence length \(L\), the recurrent memory performs

\[
O(LHd_vd_k)
\]

work.

Its recurrent inference state contains

\[
2Hd_vd_k
\]

real floats in the complex implementation, independent of \(L\).

By contrast, causal softmax attention performs conventional score computation proportional to

\[
O(L^2Hd_k)
\]

over a full sequence, while its inference KV cache grows linearly with context length.

The correct claim is:

> Phase-coded Delta memory has linear recurrent work in sequence length and a fixed-size inference state.

This asymptotic property does not imply immediate wall-clock superiority:

- the reference implementation uses a sequential Python loop;
- optimized attention kernels can be faster at moderate sequence lengths;
- FlashAttention avoids materializing an \(L\times L\) score matrix during training;
- ordinary autograd through the recurrent scan may retain intermediate states;
- a fused or parallelized scan is needed for a meaningful throughput comparison.

No fixed memory-reduction factor such as “1000×” is claimed without a specified model shape, sequence length, precision and inference implementation.

---

## 10. Current working hypotheses

### H1 — Feature-budget hypothesis

The phase model dedicates a larger fraction of its fixed state budget to key discrimination:

\[
32\text{ value dimensions}\times64\text{ real key features}.
\]

This may explain part or all of its advantage over a \(45\times45\) square real state.

**Prediction:** A real rectangular \(32\times64\) baseline should close the gap if state allocation is the main cause.

---

### H2 — Fixed-norm hypothesis

Phase coding prevents the network from using key magnitude and eliminates key-norm variation:

\[
\|\phi(\theta)\|^2=d_k.
\]

**Prediction:** L2-normalized real keys should close the gap if norm control is the main cause.

---

### H3 — Angular-geometry hypothesis

The paired sine-cosine parametrization constrains optimization to the product of circles \((S^1)^{d_k}\). This may produce more stable or better-conditioned learned key sets than an unconstrained normalized real projection.

**Prediction:** If phase still wins against a normalized \(2d_k\)-dimensional real baseline, it should exhibit at least one of:

- lower off-diagonal Gram coherence;
- a smaller Gram condition number;
- more stable state norms;
- smaller residual growth;
- greater robustness to learning rate or sequence length.

---

### H4 — Delta-rule hypothesis

Most of the improvement may come from residual-error writes rather than phase coding.

**Prediction:** In a factorial comparison, Delta should produce a large gain in both real and phase representations.

---

### H5 — Architectural-stack hypothesis

The observed advantage may result from interactions among Conv1D, gates, normalization and the Delta update rather than from the key representation alone.

**Prediction:** The gap should shrink when all non-representation components are made identical.

---

## 11. Decisive experiments

### 11.1 Exact realification test

Implement the real equivalent using

```python
phi_k = torch.cat([theta_k.cos(), theta_k.sin()], dim=-1)
phi_q = torch.cat([theta_q.cos(), theta_q.sin()], dim=-1)
```

and a state

```python
W.shape == [batch, heads, d_value, 2 * d_key]
```

With equivalent weights, the complex and realified models should agree numerically on:

- pre-write predictions;
- residuals;
- state updates;
- query outputs;
- loss;
- parameter gradients.

This is an algebraic equivalence test, not a performance baseline.

---

### 11.2 Fair state-budget baselines

At \(d_v=d_k=32\), compare:

| Baseline | Key features | Value dim. | State floats |
|---|---:|---:|---:|
| Phase | \([\cos\theta,\sin\theta]\), 64 real coordinates | 32 | 2048 |
| Exact realification | same phase features, real implementation | 32 | 2048 |
| Real normalized rectangular | 64 unconstrained real features + L2 norm | 32 | 2048 |
| Real fixed-norm/Rademacher | 64 bounded real features | 32 | 2048 |
| Existing real square | 45 real features | 45 | 2025 |

All models must use the same:

- local mixer;
- number of layers and heads;
- value dimension where applicable;
- Delta update;
- gate;
- normalization;
- optimizer;
- learning-rate search procedure;
- number of seeds;
- training-token budget.

---

### 11.3 Representation × write-rule factorial

| Representation | Hebbian write | Delta write |
|---|---:|---:|
| Real normalized | required | required |
| Phase-coded | required | required |

This separates the contribution of phase from the contribution of residual-error writing.

---

### 11.4 Mechanistic measurements

At each capacity level, record:

- key-norm distribution;
- pairwise cosine-kernel distribution;
- off-diagonal Gram RMS;
- maximum Gram coherence;
- Gram eigenvalue spectrum;
- Gram condition number;
- recurrent-state norm over time;
- residual norm over time;
- \(\beta_t\) distribution;
- accuracy of old associations after each new write;
- sensitivity to write order.

Accuracy alone cannot identify the mechanism.

---

### 11.5 Robustness and adversarial MQAR

Evaluate:

- at least three seeds;
- larger vocabularies;
- duplicate or overwritten keys;
- correlated keys;
- nonuniform key frequencies;
- more queries per key;
- sequence lengths unseen during training;
- different key-value serializations;
- noisy or partially corrupted queries;
- out-of-distribution numbers of stored pairs.

Random-token MQAR is a necessary first test, but correlated and overwritten associations are more diagnostic of practical memory behavior.

---

### 11.6 Canonical architecture comparison

Before claiming improvement over real Delta networks, compare against a trusted implementation of:

- DeltaNet;
- Gated DeltaNet;
- a conventional linear-attention baseline;
- causal softmax attention.

Comparisons should include:

- accuracy;
- parameters;
- state memory;
- training FLOPs;
- inference throughput;
- latency;
- peak memory.

---

## 12. Corrected reference implementation

The following code expresses the core complex recurrence unambiguously:

```python
def complex_delta_scan(K, Q, V, beta):
    """
    K, Q:  [B, L, H, d_k], complex unit phasors
    V:     [B, L, H, d_v], real
    beta:  [B, L, H, 1, 1], real
    return [B, L, H, d_v], real
    """
    B, L, H, d_k = K.shape
    d_v = V.size(-1)

    M = torch.zeros(
        B, H, d_v, d_k,
        dtype=K.dtype,
        device=K.device,
    )

    outputs = []
    inv_dk = 1.0 / d_k

    for t in range(L):
        k_t = K[:, t]             # [B, H, d_k]
        q_t = Q[:, t]             # [B, H, d_k]
        v_t = V[:, t]             # [B, H, d_v]
        b_t = beta[:, t]          # [B, H, 1, 1]

        # Pre-write prediction for the current key
        v_old = torch.einsum(
            "bhij,bhj->bhi",
            M,
            k_t.conj(),
        ).real * inv_dk

        err = v_t - v_old

        # Outer product: [B,H,d_v,1] * [B,H,1,d_k]
        update = (
            err.to(M.dtype).unsqueeze(-1)
            * k_t.unsqueeze(-2)
        )

        M = M + b_t * update

        # Read after the write
        retrieved = torch.einsum(
            "bhij,bhj->bhi",
            M,
            q_t.conj(),
        ).real * inv_dk

        outputs.append(retrieved)

    return torch.stack(outputs, dim=1)
```

Its exact real-valued equivalent is:

```python
def realified_phase_delta_scan(theta_k, theta_q, V, beta):
    """
    theta_k, theta_q: [B, L, H, d_k], real
    V:                [B, L, H, d_v], real
    beta:             [B, L, H, 1, 1], real
    """
    phi_k = torch.cat(
        [theta_k.cos(), theta_k.sin()],
        dim=-1,
    )
    phi_q = torch.cat(
        [theta_q.cos(), theta_q.sin()],
        dim=-1,
    )

    B, L, H, two_dk = phi_k.shape
    d_k = two_dk // 2
    d_v = V.size(-1)

    W = torch.zeros(
        B, H, d_v, two_dk,
        dtype=V.dtype,
        device=V.device,
    )

    outputs = []
    inv_dk = 1.0 / d_k

    for t in range(L):
        k_t = phi_k[:, t]
        q_t = phi_q[:, t]
        v_t = V[:, t]
        b_t = beta[:, t]

        v_old = torch.einsum(
            "bhij,bhj->bhi",
            W,
            k_t,
        ) * inv_dk

        err = v_t - v_old

        W = W + b_t * (
            err.unsqueeze(-1)
            * k_t.unsqueeze(-2)
        )

        retrieved = torch.einsum(
            "bhij,bhj->bhi",
            W,
            q_t,
        ) * inv_dk

        outputs.append(retrieved)

    return torch.stack(outputs, dim=1)
```

These two implementations should agree up to numerical precision when initialized from equivalent parameters.

---

## 13. Relevance beyond synthetic recall

The result connects several established ideas:

- residual-error fast-weight programming;
- Delta-rule and normalized-LMS updates;
- holographic and phasor representations;
- fixed-norm feature maps;
- recurrent linear attention;
- online associative memory.

The potential practical value is not that quadratic attention has already been replaced. The current result supports a narrower proposition:

> A fixed-size recurrent Delta memory operating on learned trigonometric features can perform near-exact associative recall in a controlled synthetic setting.

The next question is whether the same inductive bias helps when:

- keys are correlated and semantically structured;
- associations are overwritten;
- the number of relevant facts exceeds the nominal state rank;
- retrieval is approximate rather than exact;
- the model is trained on natural language;
- throughput is measured using optimized kernels.

Natural-language transfer is therefore a separate empirical hypothesis, not a consequence of the MQAR result.

---

## 14. Conclusion

The strongest current interpretation of the experiments is not that complex numbers uniquely eliminate recurrent-memory interference.

Instead:

1. phase coding maps \(d_k\) learned angles to a bounded, fixed-norm \(2d_k\)-coordinate real feature representation;
2. the Delta rule exactly corrects the current association according to its learned write gate;
3. interference with previous associations is controlled by the coherence of a learned cosine kernel;
4. the resulting recurrence uses fixed-size inference state and work linear in sequence length;
5. the complete system achieves 99.95% best observed accuracy in the tested short-range MQAR harness.

The main unresolved issue is attribution. The observed advantage may arise from:

- feature dimensionality;
- state-budget allocation;
- fixed-norm keys;
- angular optimization geometry;
- Gram conditioning;
- the Delta update;
- or interactions with the surrounding architectural stack.

A real normalized rectangular baseline with identical architecture is the decisive next control.

Whatever its outcome, it is scientifically informative:

- if the real baseline matches phase, the experiments reveal a useful real trigonometric Delta memory and explain the original result;
- if phase remains superior, the remaining gap becomes stronger evidence for a genuine angular parametrization or optimization advantage.

---

## References

- Arora, S. et al. *Zoology: Measuring and Improving Recall in Efficient Language Models*.
- Plate, T. A. (1995). “Holographic Reduced Representations.” *IEEE Transactions on Neural Networks*.
- Schlag, I., Irie, K., and Schmidhuber, J. (2021). “Linear Transformers Are Secretly Fast Weight Programmers.”
- Widrow, B., and Hoff, M. E. (1960). “Adaptive Switching Circuits.”
- Yang, S. et al. (2024). *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*.
- Yang, S. et al. (2024). *Gated Delta Networks: Improving Mamba2 with Delta Rule*.
