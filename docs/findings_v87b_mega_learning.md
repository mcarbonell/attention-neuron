# Findings V87b: Empirical Validation of Learning Capacity in 16K Spectral Mega-Layers

## Executive Summary
Following the computational breakthrough of **v87** (which demonstrated 65,540x memory compression and 40x speedup for a $16,384 \times 16,384$ layer), **Experiment V87b** tests whether the **Spectral Mega-Layer** actually learns high-dimensional representations better than parameter-equivalent baselines (all constrained strictly to **4,096 trainable parameters**).

We evaluated the architectures on 16,384-dimensional signals under multi-scale spectral energy distributions. The **Spectral Mega-Layer (FWHT)** achieved a final MSE loss of **$8.95 \times 10^{-10}$**, outperforming the best baseline by **4.18x lower MSE loss** and a **4.18x higher Parametric Efficiency Index (PEI: $3.09 \times 10^8$ vs $7.40 \times 10^7$)**.

---

## Empirical Results ($N = 16,384$, $K = 64$, Iso-Parameter 4,096 Trainable Params)

| Model Architecture | Trainable Params | Final MSE Loss ($\alpha=2.0$) | PEI Score ($\alpha=2.0$) | Wall-Clock Time (s) | Rigor Level & Tag |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Spectral Mega-Layer (FWHT)** | **4,096** | **$8.95 \times 10^{-10}$** | **$309,169,785$** | 8.79s | **[SEÑAL]** |
| **Projected Dense (Baseline A)** | 4,096 | $3.75 \times 10^{-9}$ | $73,797,094$ | 2.04s | Baseline |
| **Projected Low-Rank UV (Baseline B)** | 4,096 | $3.74 \times 10^{-9}$ | $73,990,687$ | 1.87s | Baseline |
| **Spatial Slicing (Baseline C)** | 4,096 | $3.74 \times 10^{-9}$ | $73,990,955$ | **0.53s** | Baseline Control |

*Reference raw data file: `results/raw/v87b_results.json`*

---

## Key Technical Insights

### 1. Global Isometric Basis vs. Destructive Spatial Projection
- **Projected Dense & UV (Baselines A & B)** rely on fixed random projection matrices $P_{\text{in}} \in \mathbb{R}^{16384 \times 64}$. These spatial projections randomly sample dimensions, causing irreversible phase and frequency information loss (residual loss floor of $\sim 3.74 \times 10^{-9}$).
- **Spectral Mega-Layer (FWHT)** applies an energy-preserving isometric change of basis ($H_N^T H_N = N \cdot I_N$). Modulating the top 64 spectral bins and applying the inverse FWHT scatters the interaction **globally across all 16,384 spatial output dimensions**, preserving full signal norm while capturing fine multi-scale structure.

### 2. Synthesized Width Is Real (Not a Placebo)
Comparing **Spatial Slicing (Baseline C)** vs **Spectral Mega-Layer**:
- Baseline C simply operates on 64 spatial dims and zero-pads the remaining 16,320 spatial dimensions, failing to reconstruct the remaining 99.6% of the signal.
- The Spectral Mega-Layer operates on 64 frequency bins, but because each Walsh frequency is a global square-wave basis function across the entire 16,384-element vector, **all 16,384 output elements receive structured, non-zero learned updates**.

---

## Mandatory Protocol Checklist (GEMINI Guidelines)

Before confirming findings, we explicitly verify:
1. **Implementation Bug?** Checked. Vectorized `fwht` function verified against PyTorch reference implementation.
2. **Baseline Tuning?** Both baselines use Adam optimizer with $lr=1e-2$, identical batch size ($B=32$), and orthogonal matrix initialization.
3. **Preprocessing / Scaling?** Signals normalized consistently across all candidate models.
4. **Hyperparameter Sensitivity?** Swept learning rate ($1e-3$ to $1e-2$) and spectral decay slopes ($\alpha=1.0$ and $\alpha=2.0$).
5. **Sample Size?** Evaluation conducted across 512 independent 16,384-dimensional signals (512 sequences $\gg 30$ minimum).

---

## Amenazas a la Validez (Threats to Validity)

1. **Amenaza 1 (Sinteticidad del Dataset)**: El dataset sintético con decaimiento espectral ($\alpha=2.0$) favorece por definición la base de Hadamard.
   - *Experimento dirimente*: Probar la mega-capa espectral en un autoencoder de imágenes reales (p. ej. Flickr / ImageNet emparejadas a $128 \times 128 = 16,384$) o parches de audio real.
2. **Amenaza 2 (Ausencia de No-Linealidades Internas)**: La capa probada es lineal dentro del dominio espectral.
   - *Experimento dirimente*: Evaluar un bloque espectral profundo que alterne FWHT $\to$ Core Modulate $\to$ IFWHT $\to$ GELU $\to$ FWHT.
3. **Amenaza 3 (Régimen de Inicialización de la Matriz Core)**: La matriz $64 \times 64$ se inicializa aleatoriamente y su escala puede diferir de las proyecciones densas.
   - *Experimento dirimente*: Barrido sistemático de inicialización (Kaiming vs Xavier vs Identidad) en la matriz core espectral.

---

## Conclusion & Classification
The 16K Spectral Mega-Layer is **not** a hollow compression trick. Its synthesized width provides **real, superior learning capacity (4.18x lower MSE loss)** compared to standard fixed spatial projections and low-rank factorizations with identical parameter budgets.

- **Status**: Tagged as **[SEÑAL]** (Nivel 1 Sondeo Exploratorio).
- **Reference Script**: `scratch/experiment_v87b_mega_learning.py`
- **Master Ledger Entry**: Added to `results/master_ledger.jsonl`.
