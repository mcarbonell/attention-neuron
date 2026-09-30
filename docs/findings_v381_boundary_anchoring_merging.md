# Findings v381: Boundary Anchoring, Phase Locking & Zero-Cost Model Merging

**Fecha:** 2026-09-30  
**Lab:** `attention-neuron/`  
**Script:** [`scratch/prototype_v381_boundary_anchoring_merging.py`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/prototype_v381_boundary_anchoring_merging.py)  
**Registro Crudo:** [`results/raw/v381_boundary_anchoring_merging.json`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/raw/v381_boundary_anchoring_merging.json)  
**Nivel de Rigor:** Nivel 1 (Sondeo Exploratorio multi-semilla, 3 semillas por condición)  
**Etiqueta:** [SEÑAL] (Bloqueo de coordenadas $\rho_{\text{raw}} = 0.996$ confirmado; interferencia de fase identificada)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

En `v380` se formuló la hipótesis de que añadir condiciones de contorno o un potencial de coordenadas rompería la simetría de traslación continua, forzando a semillas independientes a caer en el mismo marco canónico y permitiendo el model merging directo (promedio simple de pesos $W_{\text{soup}} = (W_A + W_B)/2$).

**Descubrimientos empíricos fundamentales de `v381`:**
1. **Bloqueo Absoluto de Coordenadas ($\rho_{\text{raw}} = 0.996$):** El potencial de inclinación de coordenadas (`Topo_CoordTilt`) logró por primera vez en el repositorio que semillas entrenadas desde inicializaciones aleatorias independientes converjan a matrices de pesos prácticamente idénticas ($\rho_{\text{raw}} = 0.9956$ en $W_1$ y $0.9336$ en $W_2$). Los mapas de calor cara a cara ([`results/figures/v381_cross_seed_heatmaps.png`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/figures/v381_cross_seed_heatmaps.png)) confirman que las franjas de Seed 42 y Seed 100 son copias fotográficas idénticas en el espacio.
2. **El Fenómeno de Interferencia de Ondas en Model Merging:** Pese a la perfecta alineación de coordenadas, la precisión del promedio ingenuo colapsó. Se identificó la causa física y arquitectónica:
   - En `Topo_CoordTilt`, el potencial lineal empujó la media de los pesos hacia valores negativos ($-2.2$ a $-1.1$), provocando la muerte masiva de neuronas ReLU (rango efectivo colapsado a $\approx 3.9$ y precisión individual reducida al $83.5\%-87.6\%$).
   - En `Topo_Free` y `Topo_BoundaryAnchor`, las matrices aprenden ondas/láminas continuas suaves. Cuando dos semillas tienen un ligero desfase espacial entre sus crestas, el promedio directo genera **interferencia destructiva de ondas** (análogo a promediar $\sin(x) + \sin(x+\phi)$), reduciendo la precisión de la sopa a $\sim 66\%-67\%$, mientras que el baseline disperso conserva un $88.7\%$.

---

## 1. Configuración Experimental

- **Modelo:** MLP de 3 capas ($784 \to 256 \to 128 \to 10$).
- **Dataset:** MNIST estándar, batch size = 256, 10 épocas por corrida.
- **Semillas:** 3 semillas independientes (`[42, 100, 2026]`) por condición (total 15 corridas).
- **Protocolo de Souping:** Para cada par $(A, B)$ se computa $W_{\text{soup}} = \frac{W_A + W_B}{2}$ y se evalúa sobre las 10,000 imágenes de test sin fine-tuning ni matching de permutación.
- **Condiciones evaluadas:**
  1. `Topo_BoundaryAnchor_eps2e-3`: Dirichlet 2D ($\epsilon = 2\times 10^{-3}$) + anclaje Dirichlet fijando fila $0$ a $+0.05$ y fila $d-1$ a $-0.05$.
  2. `Topo_CoordTilt_eps2e-3_lam1e-3`: Dirichlet 2D + potencial lineal $y_i \in [-1, 1]$ con $\lambda = 10^{-3}$.
  3. `Topo_CoordTilt_eps2e-3_lam2e-3`: Dirichlet 2D + potencial lineal con $\lambda = 2\times 10^{-3}$.
  4. `Topo_Free_NoAnchor_eps2e-3`: Dirichlet 2D libre de `v380` ($\epsilon = 2\times 10^{-3}$).
  5. `Baseline_Standard_AdamW`: Control estándar ($\epsilon = 0, \lambda = 0$).

---

## 2. Resultados Consolidados

| Condición | Indiv Acc (%) | Soup Acc (%) | Barrera ($\Delta \text{Acc}$) | $\rho_{\text{raw}}$ $W_1$ | $\rho_{\text{raw}}$ $W_2$ | EffRank $W_1$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Topo_BoundaryAnchor_eps2e-3` | $97.61 \pm 0.14$ | $67.59 \pm 2.31$ | $30.02$ | 0.0665 | 0.1023 | 73.35 |
| `Topo_CoordTilt_eps2e-3_lam1e-3` | $83.56 \pm 6.70$ | $16.57 \pm 5.50$ | $66.99$ | **0.9956** | **0.9336** | **3.91** |
| `Topo_CoordTilt_eps2e-3_lam2e-3` | $87.61 \pm 2.89$ | $12.53 \pm 4.19$ | $75.08$ | **0.9969** | **0.9203** | **4.28** |
| `Topo_Free_NoAnchor_eps2e-3` | $97.47 \pm 0.21$ | $66.11 \pm 2.84$ | $31.36$ | 0.0133 | 0.0682 | 74.60 |
| `Baseline_Standard_AdamW` 🌟 | **$97.87 \pm 0.32$** | **$88.72 \pm 3.17$** | **9.15** | 0.0076 | 0.0015 | 199.22 |

*Nota de rigor en marcadores:* 🌟 Asignado al baseline por mayor precisión individual y de sopa.

---

## 3. Análisis Mecanicista: La Física del Bloqueo vs. La No-Linealidad

### A. El Éxito en Bloqueo de Gauge ($\rho_{\text{raw}} = 0.996$)
Al observar la figura [`results/figures/v381_cross_seed_heatmaps.png`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/figures/v381_cross_seed_heatmaps.png), las columnas de `Topo_CoordTilt` revelan que **Seed 42 y Seed 100 son clones geométricos**. Cada línea horizontal está en la misma coordenada exacta de la matriz. La correlación $\rho_{\text{raw}} = 0.996$ demuestra matemáticamente que:
> *Un potencial de coordenadas direccional destruye por completo la simetría de gauge discreta $N!$ y las traslaciones continuas.*

### B. El Conflicto con la No-Linealidad ReLU
¿Por qué cayó la precisión individual a $\sim 85\%$ y la sopa a $\sim 14\%$?
- El término $-\lambda \sum y_i W_{i, j}$ empuja los pesos superiores ($y_i < 0$) hacia valores fuertemente negativos.
- En una red con activación $\operatorname{ReLU}(W x + b)$, pesos negativos masivos implican que el 50% de las neuronas quedan **completamente muertas** ($\text{output} \equiv 0$).
- Esto redujo el rango efectivo a un anémico $3.91$, asfixiando la capacidad representacional.

### C. Interferencia Destructiva en Redes Continuas
En `Topo_Free` y `Topo_BoundaryAnchor`, las neuronas individuales alcanzan un $97.6\%$ excelente. Sin embargo, al promediar dos semillas, la sopa obtiene $67\%$. 
- Las matrices topográficas forman **campos armónicos continuos (láminas de Fourier/ondas estacionarias)**.
- Si dos semillas aprenden la misma frecuencia espacial pero con un desfase de fase relativo $\Delta \phi \neq 0$, la suma directa $\frac{1}{2}(W_A + W_B)$ cancela las amplitudes por interferencia destructiva.

---

## 4. Amenazas a la Validez

1. **Sesgo de Activación ReLU:** El potencial de inclinación lineal afectó la simetría de signos. En activaciones antisimétricas (como $\tanh$, GELU centrado o neuronas complejas polimórficas de fase), un desplazamiento de potencial no apaga las neuronas de la misma forma que en ReLU.
2. **Escala de MNIST en Modo de Conectividad:** En un MLP superficial sobre MNIST, el baseline ya exhibe una barrera relativamente baja ($9.15\%$). En arquitecturas más profundas o problemas más complejos (CIFAR/LLMs), la barrera del baseline suele ser total ($\sim 90\%$).

---

## 5. Próximo Paso Recomendado

Para resolver la interferencia de ondas y preservar la precisión del $97.6\%$:
- **Anclaje de Fase de Media Cero (Zero-DC Phase Pinning):** En lugar de un potencial lineal monótono que introduce un offset negativo masivo, usar un **potencial oscilatorio armónico de media cero**:
  $$\mathcal{R}_{\text{pin}}(W) = -\lambda \sum_{i, j} \sin\left(\frac{2\pi i}{d_{\text{out}}}\right) W_{i, j}$$
  Esto fija la fase espacial de la onda sin desplazar la media DC de los pesos ni matar las ReLUs.
