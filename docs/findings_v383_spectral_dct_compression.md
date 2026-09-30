# Findings v383: Compresión Espectral 2D-DCT en Redes Topográficas (TCR)

**Fecha:** 2026-10-01  
**Lab:** `attention-neuron/`  
**Script:** [`scratch/prototype_v383_spectral_dct_compression.py`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/prototype_v383_spectral_dct_compression.py)  
**Registro Crudo:** [`results/raw/v383_spectral_dct_compression.json`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/raw/v383_spectral_dct_compression.json)  
**Nivel de Rigor:** Nivel 1 (Sondeo Exploratorio multi-semilla, 3 semillas por condición)  
**Etiqueta:** [SEÑAL] (Validación de compresión 50x post-entrenamiento sin re-entrenamiento)  

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

En `v381` y `v382`, la evaluación de TCR se centró en *Model Souping* (fusión algebraica de modelos independientes), donde se observó interferencia destructiva por desfasaje espacial o colapso a rango 1 al forzar anclaje externo. Aquellos resultados podían sugerir la lectura pesimista de que la regularización celular topográfica era meramente una restricción que cobraba peaje en precisión individual sin utilidad práctica tangible.

**Reconciliación y Nueva Evidencia de `v383`:**
1. **La Utilidad Real de TCR es la Compresión Espectral, no el Souping:** `v383` clarifica que el verdadero retorno de la regularización celular Dirichlet no es la alineación de fase entre semillas distintas, sino la **concentración extrema de energía en frecuencias espaciales bajas (2D-DCT)** dentro de cada modelo.
2. **Diferenciación entre Rango SVD (v380) y Soporte DCT (v383):** En `v380`, `Turing_DoG` exhibía un 90% de energía espectral en rango 4 vía SVD. Sin embargo, en `v383`, bajo poda por magnitud 2D-DCT, `Turing_DoG` colapsa tempranamente al 75% de esparsidad (61.29% de precisión). Esto ocurre porque los patrones de Turing son fenómenos resonantes de banda pasante (bandpass), mientras que `Dirichlet_2D` es un filtro pasobajo estricto de mínima tensión superficial.
3. **Resistencia a la Poda Extrema:** Mientras que los pesos del optimizador estándar (AdamW) se comportan como ruido blanco isotrópico ($\operatorname{AdjCos} \approx 0.0114$) y colapsan a 98% de poda ($38.82\%$), el modelo `Dirichlet_2D_eps1e-2` retiene un **$96.28\%$ de precisión con el 98% de coeficientes a cero (compresión 50x)**, con una pérdida de apenas $-0.50\%$ respecto al modelo sin podar ($96.78\%$).

---

## 1. Configuración Experimental

- **Modelo:** MLP de 3 capas ($784 \to 256 \to 128 \to 10$, total 235,146 parámetros).
- **Transformada:** Base ortonormal completa 2D-DCT tipo II / III calculada analíticamente ($D \in \mathbb{R}^{M \times M}$, $W_{\text{dct}} = D_{\text{out}} W D_{\text{in}}^T$).
- **Poda:** Poda por magnitud global sobre los coeficientes espectrales de $W_1$ y $W_2$ sin ningún fine-tuning posterior.
- **Niveles de esparsidad evaluados:** $0\%$ (1.0x), $50\%$ (2.0x), $75\%$ (4.0x), $90\%$ (10.0x), $95\%$ (20.0x), $98\%$ (50.0x).
- **Dataset:** MNIST estándar (60k train / 10k test), batch size = 256, 10 épocas por corrida.
- **Semillas:** 3 semillas independientes (`[42, 100, 2026]`) por condición (total 12 ejecuciones).
- **Condiciones evaluadas:**
  1. `Turing_DoG_eps2e-3`: Regularizador Turing DoG ($\epsilon = 0.002$).
  2. `Dirichlet_2D_eps2e-3`: Regularizador Dirichlet 2D moderado ($\epsilon = 0.002$).
  3. `Dirichlet_2D_eps1e-2`: Regularizador Dirichlet 2D fuerte ($\epsilon = 0.01$).
  4. `Baseline_Standard_AdamW`: Control estándar sin acoplamiento espacial ($\epsilon = 0.0$).

---

## 2. Resultados Consolidados

### A. Precisión en Test vs. Nivel de Esparsidad 2D-DCT

| Condición | 0% (1.0x) | 50% (2.0x) | 75% (4.0x) | 90% (10.0x) | 95% (20.0x) | 98% (50.0x) | $\operatorname{AdjCos}$ $W_1$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Turing_DoG_eps2e-3` | $97.72 \pm 0.10$ | $95.56 \pm 0.77$ | ⚠️ $61.29 \pm 8.27$ | ⚠️ $23.72 \pm 8.28$ | ⚠️ $12.42 \pm 1.65$ | ⚠️ $10.46 \pm 1.15$ | 0.0916 |
| `Dirichlet_2D_eps2e-3` | $97.47 \pm 0.17$ | $97.47 \pm 0.16$ | $97.47 \pm 0.13$ | 🌟 **$97.25 \pm 0.12$** | $96.46 \pm 0.12$ | $90.65 \pm 2.73$ | 0.9103 |
| `Dirichlet_2D_eps1e-2` | $96.78 \pm 0.23$ | $96.78 \pm 0.23$ | $96.80 \pm 0.22$ | $96.85 \pm 0.17$ | 🌟 **$96.80 \pm 0.22$** | 🌟 **$96.28 \pm 0.24$** | **0.9696** |
| `Baseline_Standard_AdamW` | 🌟 **$97.87 \pm 0.26$** | 🌟 **$97.79 \pm 0.28$** | 🌟 **$97.38 \pm 0.48$** | $94.45 \pm 1.96$ | ⚠️ $80.14 \pm 4.95$ | ⚠️ $38.82 \pm 6.40$ | 0.0114 |

*Convención de rigor:* 🌟 Mejor valor numérico por columna. ⚠️ Degradación anómala o colapso.

### B. Retención de Energía Espectral en $W_1$ (%)

| Condición | 0% (1x) | 50% (2x) | 75% (4x) | 90% (10x) | 95% (20x) | 98% (50x) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Turing_DoG_eps2e-3` | 100.0% | 97.65% | 87.45% | 65.36% | 48.39% | 30.39% |
| `Dirichlet_2D_eps2e-3` | 100.0% | 99.98% | 99.79% | 98.72% | 96.24% | 88.38% |
| `Dirichlet_2D_eps1e-2` 🌟 | 100.0% | **100.00%** | **99.96%** | **99.75%** | **99.22%** | **96.75%** |
| `Baseline_Standard_AdamW` | 100.0% | 96.74% | 86.71% | 70.24% | 57.60% | 40.72% |

### C. Retención de Energía Espectral en $W_2$ (%)

| Condición | 0% (1x) | 50% (2x) | 75% (4x) | 90% (10x) | 95% (20x) | 98% (50x) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Turing_DoG_eps2e-3` | 100.0% | 95.61% | 80.91% | 55.92% | 39.21% | 22.90% |
| `Dirichlet_2D_eps2e-3` | 100.0% | 99.92% | 99.27% | 95.40% | 87.74% | 70.22% |
| `Dirichlet_2D_eps1e-2` 🌟 | 100.0% | **100.00%** | **99.97%** | **99.68%** | **98.70%** | **93.28%** |
| `Baseline_Standard_AdamW` | 100.0% | 93.25% | 73.62% | 45.92% | 29.93% | 16.11% |

### D. Coste Computacional y Sobrecarga

| Condición | Wall-Clock Medio (s) | Sobrecarga Regularizador (s) | % Sobrecarga |
| :--- | :---: | :---: | :---: |
| `Turing_DoG_eps2e-3` | 109.51 | 5.06 | 4.62% |
| `Dirichlet_2D_eps2e-3` | 93.75 | 1.08 | 1.15% |
| `Dirichlet_2D_eps1e-2` | 118.68 | 1.27 | 1.07% |
| `Baseline_Standard_AdamW` | 133.16 | 0.12 | 0.09% |

---

## 3. Análisis de Hallazgos

### A. La Curva Plana de Dirichlet 2D hasta 50x
Al revisar [`results/figures/v383_dct_compression_curve.png`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/figures/v383_dct_compression_curve.png):
- `Dirichlet_2D_eps1e-2` mantiene una curva prácticamente plana desde el $0\%$ hasta el $95\%$ de poda ($96.78\% \to 96.80\%$).
- Al alcanzar el $98\%$ de coeficientes eliminados (factor de compresión 50x, donde solo quedan activos 4,703 coeficientes de un total de 235,146), la precisión se mantiene en **$96.28 \pm 0.24\%$**.
- En contraste, el baseline estándar experimenta una caída catastrófica no lineal a partir del $90\%$ de poda, desplomándose a **$80.14 \pm 4.95\%$** al $95\%$ y a **$38.82 \pm 6.40\%$** al $98\%$.
- La diferencia neta en el régimen de 50x compresión es de **$+57.46\%$** a favor de Dirichlet ($\Delta = 57.46\% \gg 2 \times \text{SE}$, con $\text{SE} \approx 3.70\%$).

### B. Concentración Espectral Exponencial
En [`results/figures/v383_dct_energy_spectra.png`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/figures/v383_dct_energy_spectra.png):
- En `Dirichlet_2D_eps1e-2`, el **$96.75\%$ de la norma Frobenius total** de $W_1$ reside en apenas el **$2\%$ de los modos DCT de menor frecuencia**.
- En el baseline estándar, el 2% de los modos DCT contiene solo el **$40.72\%$ de la energía**. Al podar el 98% restante, se cercena el 60% de la energía de la matriz, provocando la pérdida de la capacidad discriminativa de la red.

### C. Por Qué Colapsa Turing DoG en DCT
`Turing_DoG` fue diseñado para inducir patrones celulares reactivos (morfogénesis local tipo activador-inhibidor).
- A diferencia del acoplamiento Dirichlet (que penaliza $(\nabla W)^2$ forzando gradientes nulos y superficies ultralisas de baja frecuencia), Turing favorece una **frecuencia resonante intermedia no nula** $k_{\text{crit}} > 0$.
- La poda estándar por magnitud corta coeficientes pequeños distribuidos en armónicos intermedios necesarios para mantener la resonancia de Turing, causando un colapso prematuro al 75% de esparsidad ($61.29\%$).
- Por lo tanto, para compresión DCT estándar, la tensión superficial de Dirichlet es estrictamente más adecuada que los filtros de morfogénesis reactiva.

---

## 4. Amenazas a la Validez

1. **Poda por Magnitud Global sin Re-entrenamiento (One-Shot):** No se aplicó fine-tuning posterior a la poda. Es posible que el baseline estándar recupere parte de la precisión perdida si se le permite re-entrenar con los coeficientes podados fijos.
2. **Escalado de Arquitectura y Complejidad del Dataset:** Experimento realizado en MLP superficial sobre MNIST ($d = 256$). La compresibilidad 2D-DCT debe verificarse en proyecciones lineales densas de Transformers ($W_q, W_k, W_v, W_o$) y con datos de texto (TinyStories / OpenWebText).
3. **Estructura Hardware del Almacenamiento DCT:** La poda por magnitud genera una matriz esparsa de coeficientes DCT. Para aprovechar esto en hardware neuromórfico o microcontroladores de borde (MCUs), es preferible emplear una máscara de corte rectangular/triangular fija (guardar únicamente el cuadrante $K \times K$ de bajas frecuencias), eliminando la sobrecarga de índices esparsos (CSR/COO).

---

## 5. Próximo Paso en el Roadmap

- **v384 (Spectral-LoRA / Finetune 100% Espectral):** Probar el ajuste fino de modelos pre-entrenados congelando la topografía espacial base y aprendiendo exclusivamente una matriz compacta $K \times K$ de coeficientes DCT en el dominio espectral.
- **v385 (Topographic Transformers):** Evaluar el acoplamiento celular Dirichlet 2D en las matrices de proyección de atención ($W_q, W_k, W_v$) de un Transformer autoregresivo pequeño, para analizar si la suavidad cortical reduce la perplejidad y acelera la inferencia.
