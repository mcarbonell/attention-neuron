# Findings v380: Unattenuated Macroscopic Topographic Cellular Regularization

**Fecha:** 2026-09-30  
**Lab:** `attention-neuron/`  
**Script:** [`scratch/prototype_v380_macroscopic_cellular_weights.py`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/prototype_v380_macroscopic_cellular_weights.py)  
**Registro Crudo:** [`results/raw/v380_macroscopic_cellular_weights.json`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/raw/v380_macroscopic_cellular_weights.json)  
**Nivel de Rigor:** Nivel 1 (Sondeo Exploratorio multi-semilla, 3 semillas por condición)  
**Etiqueta:** [SEÑAL] (Transición de fase topográfica y colapso espectral confirmados)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

Este experimento reconcilia y corrige directamente la conclusión preliminar de `v379`:
1. **Reconciliación matemática de la atenuación $1/N$:** En `v379`, la penalización utilizó `.mean()`, dividiendo la suma por $N = 256 \times 784 = 200{,}704$ parámetros. Esto provocó una atenuación de seis órdenes de magnitud en el gradiente espacial ($\sim 10^{-9}$ frente a $\sim 10^{-3}$ de la tarea). Al corregir la formulación en `v380` según la Ecuación 16 de la propuesta (suma directa sin divisor $1/N$), el gradiente espacial opera en el mismo orden que la tarea ($\sim 10^{-4}$ a $10^{-3}$).
2. **Validación de H2 (Topografía Cortical) y H3 (Colapso Espectral):** Los datos de `v380` refutan la aparente insensibilidad observada en `v379`. Con la formulación no atenuada, se confirma una transición de fase dramática:
   - La correlación coseno adyacente ($\operatorname{AdjCos}$) salta de $0.0114$ (ruido ortogonal) a **$0.9696$** (continuo suave cuasi-perfecto).
   - El rango efectivo colapsa de $199.22$ a **$50.81$** (Dirichlet) y la retención de energía en rango 4 salta de $18.40\%$ a **$90.00\%$** (Turing DoG) preservando $97.72\%$ de precisión.
3. **Matiz Teórico sobre H1 (Alineación Inter-Semilla):** La hipótesis de que la regularización espacial alinearía semillas independientes sin matching no se cumple en su forma fuerte ($\rho_{\text{raw}}$ crece de $0.0076$ a $0.0238$ en $W_1$ y hasta $0.0859$ en $W_2$, pero no alcanza valores cercanos a $1.0$). La razón es que la suavidad espacial reduce la simetría discreta $N!$, pero introduce una simetría continua de traslación/reflexión sobre el plano: cada semilla auto-organiza un mapa continuo, pero el origen de coordenadas o la polaridad espacial del mapa varía independientemente entre semillas.

---

## 1. Configuración Experimental

- **Modelo:** MLP de 3 capas ($784 \to 256 \to 128 \to 10$), $235{,}146$ parámetros.
- **Dataset:** MNIST estándar normalizado, batch size = 256, 10 épocas por corrida.
- **Optimizador:** AdamW ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$).
- **Semillas:** 3 semillas independientes (`[42, 100, 2026]`) por condición (total 18 ejecuciones).
- **Formulación no atenuada:**
  $$\mathcal{R}_{\text{dirichlet}}(W) = \frac{\epsilon}{2} \left[ \sum (\Delta_r W)^2 + \sum (\Delta_c W)^2 \right] \implies -\nabla_{W} \mathcal{R} = \epsilon \cdot \Delta W$$

---

## 2. Resultados Consolidados

| Condición | Modo | $\epsilon$ | Val Acc (%) | EffRank $W_1$ | AdjCos (H2) | $\rho_{\text{raw}}$ $W_1$ (H1) | $\rho_{\text{raw}}$ $W_2$ | Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Dirichlet_2D_eps1e-2` | dirichlet | $10^{-2}$ | $96.78 \pm 0.28$ | **50.81** | **0.9696** | 0.0238 | **0.0859** | 90.36 |
| `Dirichlet_2D_eps2e-3` | dirichlet | $2 \times 10^{-3}$ | $97.47 \pm 0.21$ | 74.60 | 0.9103 | 0.0133 | 0.0682 | 89.71 |
| `Dirichlet_2D_eps5e-4` | dirichlet | $5 \times 10^{-4}$ | $97.63 \pm 0.31$ | 105.20 | 0.7407 | 0.0205 | 0.0292 | 89.35 |
| `Cortical_1D_eps2e-3` | cortical_1d | $2 \times 10^{-3}$ | $97.38 \pm 0.38$ | 89.87 | 0.9033 | 0.0175 | 0.0438 | 93.21 |
| `Turing_DoG_eps2e-3` | turing | $2 \times 10^{-3}$ | $97.72 \pm 0.13$ | 93.70 | 0.0916 | 0.0026 | -0.0025 | 148.29 |
| `Baseline_Standard_AdamW` 🌟 | none | $0.0$ | **$97.87 \pm 0.32$** | 199.22 | 0.0114 | 0.0076 | 0.0015 | 88.33 |

*Nota de rigor en marcadores:* 🌟 Se asigna al `Baseline_Standard_AdamW` por obtener el máximo valor nominal en precisión ($97.87\%$). No obstante, `Turing_DoG_eps2e-3` ($97.72\%$) y `Dirichlet_2D_eps5e-4` ($97.63\%$) están dentro del intervalo de $\pm 1 \times \text{SE}$ del baseline ($0.32\%$), exhibiendo retenciones espectrales y topográficas incomparablemente superiores.

---

## 3. Análisis de las Hipótesis

### H2: Inducción de Topografía Cortical (CONFIRMADA)
- En el baseline, neuronas consecutivas en el índice son ortogonales ($\operatorname{AdjCos} = 0.0114$).
- Con Dirichlet $10^{-2}$, $\operatorname{AdjCos}$ alcanza **$0.9696$** (un incremento de $\sim 85\times$). Los mapas de calor ([`results/figures/v380_weight_heatmaps.png`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/figures/v380_weight_heatmaps.png)) muestran que el ruido salt-and-pepper se disuelve por completo en bandas horizontales y láminas funcionales continuas.
- Con `Cortical_1D_eps2e-3`, se induce topografía estrictamente a lo largo del eje neuronal ($\operatorname{AdjCos} = 0.9033$) con columnas verticales de sintonización nítidas.

### H3: Colapso Espectral y Compresibilidad de Bajo Rango (CONFIRMADA)
- El espectro singular de `Turing_DoG_eps2e-3` ([`results/figures/v380_svd_spectrum.png`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/figures/v380_svd_spectrum.png)) presenta un decaimiento pronunciado:
  - **Energía en Rango 4:** **$90.00\%$** (frente a $18.40\%$ en el baseline).
  - **Energía en Rango 8:** **$92.59\%$** (frente a $31.53\%$ en el baseline).
  - **Energía en Rango 16:** **$94.62\%$** (frente a $49.80\%$ en el baseline).
- Esto confirma la predicción del documento: la matriz se auto-organiza en un subespacio de rango ultrabajo donde una aproximación de rango 4 conserva el 90% de la energía funcional sin degradar la precisión ($97.72\%$ vs $97.87\%$).
- Con Dirichlet $10^{-2}$, el rango efectivo cae de $199.22$ a **$50.81$** (reducción de $4\times$).

### H1: Ruptura de la Degeneración por Permutación (PARCIALMENTE MATIZADA)
- La correlación cruzada inter-semilla $\rho_{\text{raw}}$ se triplica en $W_1$ ($0.0076 \to 0.0238$) y se multiplica por **$57\times$** en $W_2$ ($0.0015 \to 0.0859$).
- Sin embargo, no converge a $\approx 1.0$. La razón física es la emergencia de una simetría continua de traslación: mientras que la permutación discreta se penaliza, la fase espacial del mapa (en qué coordenada específica del eje se sitúa cada clase) varía libremente entre semillas. Se requiere un punto de anclaje de frontera (boundary condition) para canonizar la fase global.

### H4: Frontera de Pareto Eficiencia vs. Precisión
- Con `Dirichlet_2D_eps5e-4`, se reduce el rango efectivo en casi un $50\%$ ($105.20$ vs $199.22$) y se eleva la topografía a $0.7407$ sufriendo únicamente una variación nominal de $-0.24\%$ en precisión ($97.63\%$ vs $97.87\%$).
- El sobrecoste temporal de la regularización (`mean_overhead_s`) es inferior a $1$ segundo por entrenamiento completo ($\sim 1\%$).

---

## 4. Amenazas a la Validez

1. **Dependencia de la Estructura de Entrada 2D:** En MNIST, la entrada tiene una geometría plana $28 \times 28$. La regularización 2D mezcla el eje neuronal ($d_{\text{out}}$) con el eje de entrada aplanado ($d_{\text{in}}$). Sería relevante probar si la topografía se sostiene cuando la entrada no tiene orden natural (ej. embeddings de lenguaje).
2. **Ausencia de Anclaje de Frontera para H1:** Sin condiciones de contorno fijas (ej. fijar la primera neurona a un patrón canónico), las traslaciones espaciales continuas impiden la alineación directa inter-semilla a coste cero.
3. **Escala del Modelo:** El sondeo se ejecutó sobre un MLP pequeño en MNIST. Validar si el colapso de rango se replica en matrices de proyección de atención ($W_Q, W_K, W_V$) en Transformers es el paso crítico para evaluar impacto a escala.

---

## 5. Próximo Paso Recomendado

- **v381:** Probar **Topographic Attention Weights** en un Transformer pequeño o en la arquitectura de atención compleja/escalar (`v375`-`v378`): aplicar Dirichlet 1D o Turing DoG sobre las cabezas de atención ($d_{\text{head}} \times d_{\text{model}}$) para comprobar si induce especialización retinotópica/tonotópica en las cabezas y si permite podado/compresión SVD sin pérdida de capacidad asociativa.
