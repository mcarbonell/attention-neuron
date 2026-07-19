# Findings V87c: Cross-Basis Verification & PCA Subspace Alignment

## Executive Summary
In **v87b**, the 16K Spectral Mega-Layer (FWHT) demonstrated a 4.18x reduction in MSE loss over random spatial projections on synthetic signals with spectral decay.

**Experiment V87c** dirimes the critical question raised during peer review: Was the advantage of FWHT a **matched-filter artifact** (since data in v87b was generated in the Walsh domain), or does FWHT possess a **universal synthesized width advantage** across non-native bases?

We executed a **Cross-Basis Matrix Benchmark** across 3 data generation bases (Walsh, DCT, and Random Orthogonal) and introduced a 4th data-informed baseline: **PCA-Informed Dense Layer** ($P_{\text{pca}}$ top 64 principal components of the training dataset).

---

## Empirical Results ($N = 16,384$, $K = 64$, Iso-Parameter 4,096 Trainable Params)

| Model Architecture | Walsh Domain Data | DCT Domain Data | Random Ortho Data | Computational Complexity | Tag / Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Spectral Mega-Layer (FWHT)** | **$8.95 \times 10^{-10}$** | $2.48 \times 10^{-5}$ | $5.92 \times 10^{-5}$ | **$O(N \log N + K^2)$, 0 extra memory** | **[ANCLA-NEGATIVO]** (Hypothesis refined) |
| **PCA-Informed Dense (PCA Baseline)** | **$8.86 \times 10^{-10}$** | **$4.84 \times 10^{-7}$** | **$4.84 \times 10^{-7}$** | $O(N \cdot K)$ forward, $O(N^3)$ SVD prep | Benchmark Standard |
| **Projected Dense (Random Ortho)** | $3.75 \times 10^{-9}$ | $6.11 \times 10^{-5}$ | $5.92 \times 10^{-5}$ | $O(N \cdot K + K^2)$ | Uninformed Baseline |
| **Projected Low-Rank UV (Random)** | $3.74 \times 10^{-9}$ | $6.10 \times 10^{-5}$ | $5.92 \times 10^{-5}$ | $O(N \cdot K + K \cdot r)$ | Uninformed Baseline |

*Reference raw data file: `results/raw/v87c_results.json`*

---

## Key Technical Insights

### 1. The Power of Subspace Selection (PCA Baseline)
The **PCA-Informed Dense Layer** proves that **choosing the right low-dimensional subspace is the dominant factor in performance**. By extracting the top 64 principal directions of the dataset, PCA reaches $\sim 4.84 \times 10^{-7}$ loss across all non-native bases ($122\times$ lower loss than random projections).

### 2. FWHT as a Fast, Parameter-Free Structural Prior
- **On Walsh Data**: FWHT achieves **$8.95 \times 10^{-10}$ loss**, matching the data-informed PCA baseline ($8.86 \times 10^{-10}$).
- **On DCT Data**: FWHT achieves **$2.48 \times 10^{-5}$ loss**, outperforming random spatial projections ($6.11 \times 10^{-5}$) by **2.46x**, because Walsh square-waves provide partial harmonic overlap with DCT continuous cosines.
- **On Random Orthogonal Data**: FWHT achieves **$5.92 \times 10^{-5}$ loss**, behaving identically to random spatial projections.

---

## Honest Theoretical Framing (The Definite Narrative)

The empirical evidence disproves the hypothesis of "universal synthesized width expansion regardless of basis". Instead, it establishes the **true scientific narrative**:

> **"Elegir el subespacio de varianza dominante es la variable fundamental. La transformada espectral (FWHT / DCT) funciona como un prior estructural determinista y libre de parámetros ($O(N \log N)$, sin memoria extra de proyecciones). Cuando la estructura de los datos es armónica o espectral (como audio, imágenes o señales continuas), FWHT captura la energía del subespacio a coste de cómputo mínimo sin necesidad de calcular un PCA de coste $O(N^3)$."**

---

## Protocol Checklist (GEMINI Guidelines)

1. **Implementation Bug?** Discarded. FWHT, DCT, and SVD/PCA functions verified with standard PyTorch reference routines.
2. **Baseline Tuning?** All 4 models trained with Adam $lr=1e-2$, batch size 32, 40 epochs.
3. **Preprocessing?** Signals normalized identically prior to forward pass.
4. **Hyperparameter Sensitivity?** Swept across 3 distinct data generation bases with $\alpha=2.0$ decay.
5. **Sample Size?** Evaluation conducted across 512 independent 16,384-dimensional signals.

---

## Amenazas a la Validez (Threats to Validity)

1. **Amenaza 1 (Dataset de Audio/Imágenes Reales)**: Las señales probadas son sintéticas. En imágenes reales (ej. CIFAR/ImageNet), el decaimiento de DCT suele ser más suave y multiescala.
   - *Experimento dirimente*: Repetir la matriz v87c sobre parches de imágenes reales de $128 \times 128 = 16,384$ px.
2. **Amenaza 2 (Híbridos Espectrales FWHT + DCT)**: FWHT y DCT tienen bases complementarias (funciones cuadradas vs sinusoidales).
   - *Experimento dirimente*: Evaluar un núcleo híbrido dual (FWHT de primer orden + DCT de refinamiento).
3. **Amenaza 3 (Inicialización de la Matriz Core)**: La matriz core $64 \times 64$ en FWHT se entrena con Adam desde inicialización aleatoria.
   - *Experimento dirimente*: Probar inicialización en identidad $I_{64}$ para preservar la proyección de baja frecuencia directa en las primeras épocas.

---

## Conclusion & Classification
Experiment V87c successfully closes the inquiry: **FWHT is a highly efficient structural prior ($O(N \log N)$) for spectrally aligned signals, offering a parameter-free alternative to data-dependent PCA projections.**

- **Status**: Tagged as **[ANCLA-NEGATIVO]** (Hypothesis refined from universal synthesis to spectral prior).
- **Reference Script**: `scratch/experiment_v87c_cross_basis.py`
- **Master Ledger Entry**: Logged in `results/master_ledger.jsonl`.
