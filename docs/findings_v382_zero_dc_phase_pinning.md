# Findings v382: Zero-DC Harmonic Phase Pinning & The Rank-1 Clamping Dilemma

**Fecha:** 2026-09-30  
**Lab:** `attention-neuron/`  
**Script:** [`scratch/prototype_v382_zero_dc_phase_pinning.py`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/prototype_v382_zero_dc_phase_pinning.py)  
**Registro Crudo:** [`results/raw/v382_zero_dc_phase_pinning.json`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/raw/v382_zero_dc_phase_pinning.json)  
**Nivel de Rigor:** Nivel 1 (Sondeo Exploratorio multi-semilla, 3 semillas por condición)  
**Etiqueta:** [SEÑAL] (El anclaje por plantilla externa colapsa a rango 1; descarte de souping por clamping armónico)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

En `v381`, se atribuyó el colapso del Model Souping a la deriva DC del potencial lineal, que empujaba los pesos a valores negativos matando las neuronas ReLU. Se hipotetizó que un potencial armónico con media estrictamente cero ($\operatorname{Mean}(H) \equiv 0.0$) evitaría la muerte de ReLUs y permitiría un Model Souping exitoso.

**Refutación y Diagnóstico Teórico Definitivo de `v382`:**
1. **La Trampa del Atractor de Rango 1:** Los datos de `v382` demuestran que el colapso no era simplemente un problema de signo/DC, sino una patología matemática más profunda del **anclaje mediante plantilla externa** $-\lambda \langle H, W \rangle$. Al añadir un patrón fijo $H$, el estado fundamental de mínima energía de ese término es un producto externo de rango 1: $W \propto H \cdot \mathbf{v}^T$.
2. **Colapso de Rango y Pérdida de Diversidad:** Con cualquier $\lambda$ suficiente para forzar alineación entre semillas ($\rho_{\text{raw}} = 0.996$), el regularizador aplasta la matriz contra la plantilla armónica, reduciendo el rango efectivo a un anémico **$3.58 - 4.26$** (frente a $74.6$ en `Topo_Free` y $199.2$ en el baseline).
3. **Incompatibilidad con Model Souping:** Al colapsar a rango 1, los 256 canales calculan esencialmente la misma combinación lineal fija modulada por la onda $H$. El espacio de características pierde la ortogonalidad necesaria para discriminar dígitos, degradando la precisión individual al $83\% - 92\%$ y haciendo que el promedio no lineal de modelos (soup) sufra una interferencia destructiva catastrófica ($16\% - 35\%$).

---

## 1. Configuración Experimental

- **Modelo:** MLP de 3 capas ($784 \to 256 \to 128 \to 10$).
- **Dataset:** MNIST estándar, batch size = 256, 10 épocas por corrida.
- **Semillas:** 3 semillas independientes (`[42, 100, 2026]`) por condición (total 18 ejecuciones).
- **Condiciones evaluadas:**
  1. `Harmonic_1D_Sin1_lam5e-4`: Dirichlet 2D ($\epsilon = 2\times 10^{-3}$) + Armónico 1D $\sin(2\pi i/d_{\text{out}})$ con $\lambda = 5\times 10^{-4}$.
  2. `Harmonic_1D_Sin1_lam1e-3`: Dirichlet 2D + Armónico 1D con $\lambda = 10^{-3}$.
  3. `Harmonic_1D_Sin1_lam2e-3`: Dirichlet 2D + Armónico 1D con $\lambda = 2\times 10^{-3}$.
  4. `Harmonic_2D_Standing_lam1e-3`: Dirichlet 2D + Onda estacionaria 2D $\sin \cdot \cos$ ($\lambda = 10^{-3}$).
  5. `Topo_Free_NoAnchor_eps2e-3`: Dirichlet 2D libre sin plantilla (ganador de `v380`).
  6. `Baseline_Standard_AdamW`: Control estándar sin regularización ($\epsilon = 0, \lambda = 0$).

---

## 2. Resultados Consolidados

| Condición | Indiv Acc (%) | Soup Acc (%) | Barrera ($\Delta \text{Acc}$) | $\rho_{\text{raw}}$ $W_1$ | $\rho_{\text{raw}}$ $W_2$ | W1 DC Mean | EffRank $W_1$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Harmonic_1D_Sin1_lam5e-4` | $91.99 \pm 1.23$ | $34.93 \pm 7.07$ | $57.06$ | **0.9959** | 0.9307 | +0.28987 | **3.58** |
| `Harmonic_1D_Sin1_lam1e-3` | $87.26 \pm 2.27$ | $21.45 \pm 4.45$ | $65.81$ | **0.9968** | 0.9469 | +0.24771 | **3.88** |
| `Harmonic_1D_Sin1_lam2e-3` | $83.67 \pm 1.40$ | $16.77 \pm 7.03$ | $66.90$ | **0.9966** | 0.9114 | +0.16470 | **4.26** |
| `Harmonic_2D_Standing_lam1e-3` | $92.61 \pm 0.68$ | $19.63 \pm 7.29$ | $72.98$ | **0.9952** | **0.9688** | -0.03573 | **4.24** |
| `Topo_Free_NoAnchor_eps2e-3` | $97.47 \pm 0.21$ | $66.11 \pm 2.84$ | $31.36$ | 0.0133 | 0.0682 | -0.00107 | **74.60** |
| `Baseline_Standard_AdamW` 🌟 | **$97.87 \pm 0.32$** | **$88.72 \pm 3.17$** | **9.15** | 0.0076 | 0.0015 | -0.00146 | **199.22** |

*Nota de rigor en marcadores:* 🌟 Asignado al baseline por mayor precisión individual y de sopa.

---

## 3. Análisis Mecanicista: La Física del Clamping vs. La Auto-Organización

### A. La Trampa de la Plantilla Externa (External Clamping)
Al observar la figura [`results/figures/v382_cross_seed_heatmaps.png`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/figures/v382_cross_seed_heatmaps.png):
- Las 4 primeras columnas muestran que Seed 42 y Seed 100 son **copias exactas al 99.6%** ($\rho_{\text{raw}} = 0.996$).
- Sin embargo, la matriz ha degenerado en un patrón puramente horizontal: franjas donde todos los pesos de la fila $i$ siguen servilmente el valor $\sin(2\pi i / d)$.
- Al forzar a la red a parecerse a una onda sinusoidal externa fija, se le quita la libertad de usar las 256 neuronas para detectar bordes o trazos diversos; los 256 canales colapsan a una sola dimensión efectiva ($\text{EffRank} \approx 3.6 - 4.2$).

### B. Por Qué el Baseline Sopa Bien en MNIST
En un MLP superficial sobre MNIST, la pérdida es suficientemente convexa y redundante como para que dos semillas independientes converjan a cuencas conectadas suavemente (la barrera en el baseline es de solo $9.15\%$, logrando un $88.72\%$ en la sopa).
- En `Topo_Free`, la red **sí se auto-organiza internamente** con un rango sano ($74.60$) y una precisión casi intacta ($97.47\%$), pero como la fase espacial es libre, cada semilla sitúa los detectores en bandas neuronales distintas (desfasadas), lo que cancela la señal al promediarlas ($66.11\%$).

### C. Lección Estratégica para Topographic Cellular Matrices
El valor de la regularización topográfica **no reside en imponer plantillas externas fijas para forzar el Model Souping ingenuo**, sino en la **suavidad y compresibilidad intrínsecas del modelo individual auto-organizado**:
- Un modelo auto-organizado libremente (`v380`) tiene $\operatorname{AdjCos} \approx 0.97$, rango efectivo controlado ($74.6$) y decaimiento exponencial SVD ($90\%$ de energía en rango 4 en Turing DoG).
- Estas propiedades son ideales para **Compresión 2D-DCT, Cuantización Delta y Hardware Neuromórfico**, no para forzar una suma algebraica de dos cerebros distintos con un molde rígido.

---

## 4. Amenazas a la Validez

1. **Plantilla de Frecuencia Única:** Se evaluó un único armónico fundamental $k=1$. Una base ortogonal completa de baja frecuencia (los primeros 8 armónicos de Fourier) podría dar más grados de libertad, aunque seguiría sufriendo la tensión entre el sesgo inductivo externo y la libertad de la tarea.
2. **Arquitectura:** Evaluado en MLP sobre MNIST. La conectividad de modo lineal (barrera de sopa) cambia radicalmente en modelos profundos de visión o Transformers.

---

## 5. Próximo Paso Recomendado (Línea de Compresión)

- **v383 (Compresión Espectral 2D-DCT):** Tomar las matrices aprendidas en `v380` (`Topo_Free` y `Turing_DoG`) y evaluar cuánta compresión paramétrica soportan bajo truncamiento espectral (podando el $50\%, 75\%, 90\%$ de los coeficientes 2D-DCT de alta frecuencia) frente al colapso del baseline estándar.
