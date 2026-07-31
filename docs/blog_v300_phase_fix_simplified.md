# O(N) Memory That Actually Works: How Complex Numbers Fix the Recall Problem

You've probably heard that linear attention is "as good as" softmax attention. It isn't. Most linear memory variants cap out at ~50-70% recall accuracy on associative tasks, while softmax hits 99%+. The gap is real, and it comes down to one thing: **crosstalk**.

Here's how we closed it — with complex numbers, of all things.

---

## The Problem: Your Memory Is Writing Over Itself

Imagine a notebook where you store key-value pairs. Every time you write a new entry, you also slightly smudge all the previous ones. After a few entries, reading back the first one gives you a blurry mess.

That's what happens in a standard O(N) recurrent memory using the delta rule. The write step is smart — it only stores the *error* between what it predicted and what it should have stored. If a key is already in memory, nothing gets written. Clean.

But the *read* step is broken. When you pull a value out of memory, you also pick up interference from every other key that happens to be somewhat similar. With real-valued keys, some keys are "louder" than others (their norms are bigger), so a few bad keys dominate the interference. The technical term is **heteroscedastic crosstalk**, but you can just think of it as: some keys are bullies.

Result: pure delta rule memories plateau around **23% accuracy** on multi-query recall tasks. The writes are fine. The reads are poisoned.

---

## The Fix: Use Phases Instead of Amplitudes

The insight comes from a 1995 paper on holographic reduced representations (Plate, 1995). Instead of representing keys as real numbers with varying magnitudes, represent them as **phasors** — points on the unit circle of the complex plane.

Think of it like a clock hand. Every key is a direction on the clock face. No key is "louder" than another — they all have the same length. What makes them different is their *angle*.

```
Real key:    [0.73, -1.24, 0.89, ...]    # varying magnitudes
Complex key: [e^(i*0.2), e^(i*3.1), ...]  # unit circle, only angles differ
```

Three things happen when you make this switch:

1. **No more bullies.** Every key has the same magnitude. No single key can dominate the interference.

2. **Interference becomes random noise.** When two phasors have different angles, their product lands at a random point on the circle. The interference from many keys adds up like a drunkard's walk — it grows slowly, not fast.

3. **Reads become exact.** To read back a value, you multiply by the *conjugate* of the key (flip its angle). For a unit phasor, `k * conj(k) = 1` always. No approximation. No residual.

---

## How It Works: 4 Lines of Math

The entire mechanism in pseudocode:

```python
# 1. Turn keys/queries into phasors (unit circle)
K = exp(i * theta_k)
Q = exp(i * theta_q)

# 2. Read: what does memory think this key maps to?
v_old = Re(M @ conj(K)) / d_k

# 3. Compute error
error = v_target - v_old

# 4. Write: store only the correction
M += learning_rate * outer(error, K)
```

If the key is already stored correctly, `error = 0` and the memory doesn't change. If it's partially wrong, the correction lands in a *different phase direction* than existing content. The memory self-corrects without smudging.

---

## One Tricky Detail: The Convolution

In our experiments, sequences look like `[key1, val1, key2, val2, ...]`. Tokens arrive one at a time. When the model sees `key3`, it hasn't seen `val3` yet — they're separate tokens.

We add a tiny causal convolution (window of 4 tokens) before the memory layer. This lets each position see its immediate neighbors, so it can pair `key_i` with `val_i` before feeding them into the memory. Without this, the model is writing keys without their values. It's like filing a letter without its envelope.

---

## The Numbers

We tested on **Multi-Query Associative Recall (MQAR)** — the standard benchmark for testing if a model can store and retrieve key-value pairs from memory.

### Does it match softmax attention?

Yes, exactly:

| Model | Complexity | Accuracy |
|:------|:----------:|:--------:|
| Softmax Attention (MHA) | O(N^2) | 99.95% |
| **Complex Delta Phase (Ours)** | **O(N)** | **99.95%** |
| Real Delta Rule | O(N) | 23% |
| Phase Scan (no delta) | O(N) | 49% |

Same accuracy. Linear cost. The O(N^2) attention matrix is unnecessary for exact recall.

### Does it beat the real-valued version?

At equal memory budgets (~2,048 floats per head):

| KV Pairs | Complex Phase | Real Delta | Gap |
|:--------:|:------------:|:----------:|:---:|
| 8 | 99.80% | 99.67% | +0.13% |
| 32 | 99.80% | 94.83% | +4.97% |
| **64** | **95.98%** | **73.14%** | **+22.84%** |

At low load, they're similar. At high load, the real version collapses. The complex version barely notices.

### Early capacity scaling results (v300, running now)

At `d_k=32` with 2,048 floats per head:

| Pairs | Complex | Real (d_k=45) |
|:-----:|:-------:|:-------------:|
| 32 | 99.74% | 93.36% |
| 64 | 99.37% | **11.11%** |
| 128 | 88.47% | **1.65%** |

The real version is essentially random-guessing at 64+ pairs. The complex version holds strong.

---

## Why This Works (Intuitive Version)

Three intuitions:

1. **Unit circle = fair competition.** No key gets an unfair advantage. Interference is均匀, not dominated by a few loud keys.

2. **Phases are directions, not magnitudes.** Storing 100 keys on the unit circle is like parking 100 needles on a clock face. As long as they point in different directions, they don't collide. In real space, keys can have wildly different "sizes" and the big ones drown out the small ones.

3. **The delta rule + phases = self-correcting memory.** If you write the same key twice, the second write does nothing (error = 0). If you write a similar key, the correction lands orthogonally. The memory doesn't accumulate errors — it fixes them.

---

## The Bottom Line

**You don't need quadratic attention for exact recall.** A fixed-size complex matrix with linear delta updates stores and retrieves associative pairs at 99%+ accuracy. The state doesn't grow with sequence length. The cost is O(N), same as the cheapest linear attention variants, but with the accuracy of the most expensive quadratic ones.

The mechanism is simple: unit-magnitude keys on the complex plane + conjugate unbinding + residual writes. Forty lines of PyTorch. No tricks.

If you're building anything that needs to remember associations over long sequences — retrieval, reasoning, memory-augmented generation — this is worth a look.

---

**Further Reading**
- [Full technical article with math, code, and references](./blog_v300_phase_fix.md)
- Plate, T.A. (1995). Holographic Reduced Representations
- Yang, S. et al. (2024). DeltaNet: Parallelized Linear Attention with Delta Rule

---

*From the [Attention Neuron](https://github.com/attention-neuron) research series. Experiments v298-v300.*
