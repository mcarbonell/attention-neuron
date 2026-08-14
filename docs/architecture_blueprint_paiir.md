# Architectural Blueprint & Audit Report: PAIIR (Parallel Adaptive Infinite Impulse Response)

**Architecture Name:** PAIIR (**P**arallel **A**daptive **I**nfinite **I**mpulse **R**esponse)  
**Status:** `[ANCLA-NEGATIVO]` — Line Closed & Cannibalized into `DeltaPhase`  
**File Location:** `src/models/paiir.py`  
**Audit Date:** August 13, 2026  

---

## ⚠️ Executive Summary & Critical Audit Findings

Following a rigorous double-precision (FP64) numerical equivalence audit and structural capacity analysis, the **PAIIR** architecture line has been classified as `[ANCLA-NEGATIVO]` and **cannibalized into `DeltaPhase`**.

Two fundamental flaws were proven mathematically and empirically:

1. **Numerical Bug in Log-Scan Division:**  
   The parallel log-cumsum scan formulation $\tilde{V}_t = \beta_t / (\Lambda_t + 10^{-6})$ suffers from severe numerical corruption. When decay values drop on signal tokens ($\alpha_t < 0.90$), $\Lambda_t$ drops below $10^{-6}$, causing the division epsilon to dominate and corrupting **71.8% to 92.9% of all tensor positions**.
   - **Empirical FP64 Equivalence Audit Error:** **158.8% to 171.6% Relative Error** vs sequential FP64 reference across $L=128, 256, 512$.

2. **Structural Ceiling of Diagonal Elementwise State:**  
   PAIIR employs a 1D diagonal state $h_t \in \mathbb{R}^{d_{\text{model}}}$ (128 floats) with a fixed linear readout $y_t = W_o h_t$. Unlike matrix outer-product models ($M_t \in \mathbb{R}^{d \times d}$, e.g. DeltaNet / DeltaPhase / Attention), diagonal state has no query-dependent content matching ($M q_t$). Consequently, performance is capped near ground level (~23% Acc vs 100% for matrix models on 8-pair MQAR).

---

## 🛠️ Architectural Cannibalization Plan into DeltaPhase

PAIIR is mathematically the **diagonal degenerate case of DeltaPhase** (collapsing rank-1 matrix delta updates to diagonal lerps). Rather than maintaining PAIIR as an independent line, its two validated inductive biases are transferred directly into **DeltaPhase** (`C:\Users\mrcm_\Local\proj\algorithms\delta-phase`):

1. **Local Causal Conv1D ($k=4$):** Pre-filtering token embeddings to provide local context before computing decay factors.
2. **Coupled Selective Gating:** Parameterizing retention as $\alpha_t = 1 - g_t \cdot \text{decay}_t$ and input as $\beta_t = g_t \cdot (\cdot)$.

---

## 📊 Empirical FP64 Equivalence Audit Results (`scratch/test_log_scan_bug.py`)

| Sequence Length $L$ | Max Absolute Error | Max Relative Error | Corrupted Tensor Positions ($\Lambda_t < 10^{-6}$) |
| :---: | :---: | :---: | :---: |
| **$L = 128$** | $1.0075 \times 10^{1}$ | **158.87%** | **71.83%** |
| **$L = 256$** | $9.9530 \times 10^{0}$ | **124.04%** | **85.91%** |
| **$L = 512$** | $1.0812 \times 10^{1}$ | **171.63%** | **92.96%** |

---

## 📜 Academic Lineage & Prior Art Mapping

PAIIR belongs to the 2023-2024 diagonal gated recurrent network family:
- **Hawk / RG-LRU (De et al., 2024 - Griffin):** Conv1D + $h_t = a_t h_{t-1} + \sqrt{1-a_t^2}(i_t \odot x_t)$.
- **Mamba-1 (Gu & Dao, 2023):** SSM with selective parameters (PAIIR lacks matrix state $D \times N$ and $C_t^\top h_t$ readouts).
- **HGRN / HGRN2 & RWKV-4/5/6:** Gated diagonal linear recurrences.

---

*Report maintained under the **attention-neuron** project framework (`GEMINI.md`).*
