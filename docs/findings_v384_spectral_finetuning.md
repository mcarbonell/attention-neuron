# Findings v384: Finetuning Espectral (Spectral-LoRA vs LoRA de Bajo Rango)

**Fecha:** 2026-10-01  
**Lab:** `attention-neuron/`  
**Script:** [`scratch/prototype_v384_spectral_finetuning.py`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/prototype_v384_spectral_finetuning.py)  
**Registro Crudo:** [`results/raw/v384_spectral_finetuning.json`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/raw/v384_spectral_finetuning.json)  
**Nivel de Rigor:** Nivel 1 (Sondeo Exploratorio multi-semilla, 3 semillas por condición)  
**Etiqueta:** [SEÑAL] (Validación de adaptación espectral 11x más ligera que LoRA)  

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

En `v380` y `v383`, se demostró que la regularización celular Dirichlet induce representaciones de pesos concentradas casi exclusivamente en frecuencias espaciales bajas (96.75% de energía en el 2% de modos 2D-DCT), lo que habilitó compresión 50x post-entrenamiento. Sin embargo, quedaba abierta la hipótesis de si la adaptación de parámetros en *fine-tuning* (PEFT) requería obligatoriamente un modelo base preentrenado con Dirichlet o si las actualizaciones espectrales ($\Delta W = D_{\text{out}}^T C D_{\text{in}}$) eran viables en modelos estándar.

**Reconciliación y Nuevos Hallazgos de `v384`:**
1. **Spectral-LoRA es Agnóstico a la Base:** Contrario a la expectativa de que Spectral-LoRA solo funcionaría sobre bases topográficas, `Spectral_k16_StandardBase` (512 parámetros de adaptador sobre una base estándar AdamW) alcanza un **$90.29 \pm 0.20\%$** en MNIST, superando a `Spectral_k16_DirichletBase` ($87.71 \pm 0.51\%$) en precisión absoluta. Esto ocurre porque la base estándar retiene una mayor capacidad inicial sin adaptar (Linear Probe: $79.24\%$ vs $69.95\%$).
2. **Mayor Plasticidad Macroscópica en Bases Dirichlet:** Aunque la precisión absoluta es mayor en la base estándar, la **ganancia neta de adaptación** ($\Delta \text{Acc} = \text{TestAcc} - \text{LinearProbe}$) es significativamente superior en la base Dirichlet: $+17.76\%$ ($k=16$) y $+19.96\%$ ($k=24$) frente a $+11.05\%$ en la base estándar. Esto sugiere que las redes con suavidad topográfica son más sensibles y receptivas a la modulación armónica de baja frecuencia.
3. **Captura del 84% al 93% de la Capacidad de LoRA con 11x Menos Parámetros:** `Spectral_k16_StandardBase` requiere únicamente 512 parámetros (frente a 5,696 en LoRA $r=4$), quedando a solo $2.09\%$ de precisión de LoRA. El 84.1% del salto de adaptación sobre el linear probe se obtiene usando el 9% del presupuesto de parámetros de LoRA.

---

## 1. Configuración Experimental

- **Modelo Base:** MLP ($784 \to 256 \to 128 \to 10$, total 235,146 parámetros).
- **Tarea de Preentrenamiento:** Fashion-MNIST (60k train / 10k test), 5 épocas, AdamW ($\text{lr} = 10^{-3}$).
  - *Base Dirichlet:* Entrenada con $\mathcal{R}_{\text{Dirichlet}}$ ($\epsilon = 0.01$).
  - *Base Estándar:* Entrenada con $\epsilon = 0.0$.
- **Tarea de Transferencia / Fine-Tuning:** MNIST (60k train / 10k test), 5 épocas, AdamW ($\text{lr} = 2\times 10^{-3}$).
  - *Backbone Congelado:* Las capas $W_1$ ($256 \times 784$) y $W_2$ ($128 \times 256$) están estrictamente congeladas (`requires_grad = False`).
  - *Cabeza de Clasificación:* Capa lineal $W_3$ ($128 \to 10$) re-inicializada y entrenable en todas las condiciones (1,290 parámetros).
- **Semillas:** 3 semillas independientes (`[42, 100, 2026]`) por condición (total 24 corridas de fine-tuning).
- **Formulación de Adaptadores:**
  - *Spectral-LoRA:* $\Delta W = D_{\text{out}, :k}^T C D_{\text{in}, :k}$, donde $C \in \mathbb{R}^{k \times k}$ es el único parámetro entrenable ($C=0$ en $t=0$).
  - *LoRA Estándar:* $\Delta W = B A \cdot (\alpha / r)$, donde $A \in \mathbb{R}^{r \times d_{\text{in}}}$, $B \in \mathbb{R}^{d_{\text{out}} \times r}$.
  - *Linear Probe:* $\Delta W \equiv 0$ (solo se entrena la cabeza $W_3$).

---

## 2. Resultados Consolidados

### A. Rendimiento de Transferencia y Eficiencia Paramétrica

| Condición | Base Preentrenada | Tipo Adaptador | Parámetros Adaptador | Test Acc (%) | Error Estándar (SE) | $\Delta \text{Acc}$ vs Probe | PEI ($\Delta / \log_{10}(P+1)$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Spectral_k8_DirichletBase` | Dirichlet | Spectral ($k=8$) | **128** | $82.35 \pm 0.68$ | 0.39% | $+12.40\%$ | 5.87 |
| `Spectral_k16_DirichletBase` | Dirichlet | Spectral ($k=16$) | **512** | $87.71 \pm 0.51$ | 0.29% | $+17.76\%$ | 🌟 **6.55** |
| `Spectral_k24_DirichletBase` | Dirichlet | Spectral ($k=24$) | **1,152** | $89.91 \pm 0.11$ | 0.06% | $+19.96\%$ | 6.52 |
| `Spectral_k16_StandardBase` | Estándar | Spectral ($k=16$) | **512** | **$90.29 \pm 0.20$** | 0.12% | $+11.05\%$ | 4.08 |
| `LoRA_r4_DirichletBase` | Dirichlet | LoRA ($r=4$) | 5,696 | $91.36 \pm 1.12$ | 0.65% | 🌟 **$+21.41\%$** | 5.70 |
| `LoRA_r4_StandardBase` | Estándar | LoRA ($r=4$) | 5,696 | 🌟 **$92.38 \pm 0.34$** | 0.20% | $+13.14\%$ | 3.50 |
| `LinearProbe_DirichletBase` | Dirichlet | Ninguno (Probe) | 0 | ⚠️ $69.95 \pm 1.70$ | 0.98% | $0.00\%$ | — |
| `LinearProbe_StandardBase` | Estándar | Ninguno (Probe) | 0 | $79.24 \pm 0.95$ | 0.55% | $0.00\%$ | — |

*Convención de rigor:* 🌟 Mejor valor numérico por métrica. ⚠️ Cota inferior sin adaptación de representación.

### B. Desglose de Parámetros en $W_1$ y $W_2$

| Adaptador | Parámetros $W_1$ ($784 \to 256$) | Parámetros $W_2$ ($256 \to 128$) | Total Adaptador | Reducción vs LoRA $r=4$ |
| :--- | :---: | :---: | :---: | :---: |
| `Spectral_k8` | $8 \times 8 = 64$ | $8 \times 8 = 64$ | **128** | **44.5x menos** |
| `Spectral_k16` | $16 \times 16 = 256$ | $16 \times 16 = 256$ | **512** | **11.1x menos** |
| `Spectral_k24` | $24 \times 24 = 576$ | $24 \times 24 = 576$ | **1,152** | **4.9x menos** |
| `LoRA_r4` | $4 \times (784+256) = 4,160$ | $4 \times (256+128) = 1,536$ | **5,696** | 1.0x (Baseline) |

---

## 3. Análisis de Hallazgos

### A. Frontera de Pareto Parámetros vs. Precisión
Al observar la figura [`results/figures/v384_spectral_finetuning_curves.png`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/figures/v384_spectral_finetuning_curves.png):
- El gráfico de dispersión logarítmico (Panel 1) muestra una curva de rendimientos decrecientes clara:
  - Pasar de 0 a 128 parámetros (`Spectral_k8`) eleva la precisión de $69.95\%$ a $82.35\%$ ($+12.40\%$).
  - Pasar de 128 a 512 parámetros (`Spectral_k16`) eleva la precisión a $87.71\%$ ($+5.36\%$).
  - Pasar de 512 a 1,152 parámetros (`Spectral_k24`) eleva la precisión a $89.91\%$ ($+2.20\%$).
  - Multiplicar por 5 los parámetros hasta 5,696 (`LoRA_r4`) solo añade $+1.45\%$ adicional ($91.36\%$).
- En términos de índice de eficiencia paramétrica (PEI), `Spectral_k16_DirichletBase` obtiene la puntuación más alta (**6.55**), superando al baseline LoRA (**5.70** en Dirichlet, **3.50** en Estándar).

### B. Invariancia del Forward Pass y Ausencia de Sobrecarga de Memoria
- En LoRA, las matrices $A$ y $B$ requieren optimizador Adam con sus respectivos estados de primer y segundo momento para $5,696 \times 3 = 17,088$ variables de estado en memoria.
- En Spectral-LoRA ($k=16$), el tensor entrenable $C$ solo ocupa $512$ elementos ($1,536$ variables de estado).
- Las matrices de proyección $U_{\text{in}}$ y $U_{\text{out}}$ son tensores estáticos precomputados (buffers), lo que reduce la memoria dinámica de entrenamiento en un **factor de 11.1x**.

### C. Por Qué Funciona en Bases Estándar
Se observa que `Spectral_k16_StandardBase` logra **$90.29 \pm 0.20\%$**.
Esto sugiere que, aunque los pesos de una red estándar tengan componentes de ruido blanco espacial, las *actualizaciones de adaptación* ($\Delta W$) entre dominios visuales emparentados (ropa $\to$ dígitos) residen predominantemente en modificaciones de baja frecuencia (filtros de orientación global y reescalado de umbrales). Filtrar las altas frecuencias en $\Delta W$ actúa como un regularizador implícito de suavidad que previene el sobreajuste a la nueva tarea.

---

## 4. Amenazas a la Validez

1. **Similitud Estructural de los Dominios:** Fashion-MNIST y MNIST comparten dimensiones espaciales ($28 \times 28$) y naturaleza de imagen. En tareas lingüísticas (Transformer autoregresivo), la matriz de proyección ($W_q, W_k, W_v$) opera sobre espacios semánticos latentes cuya estructura espacial intrínseca podría requerir permutación previa para beneficiarse de la base DCT.
2. **Barrido de Rangos en LoRA:** Se evaluó LoRA con rango fijo $r=4$. Un barrido exhaustivo con $r=1$ ($1,424$ parámetros) y $r=2$ ($2,848$ parámetros) aportaría una comparación punto a punto más densa en la curva de Pareto.
3. **Escala del Modelo:** Verificado en un MLP de 3 capas. El comportamiento en redes convolucionales profundas o capas de atención densas multi-cabeza debe contrastarse empíricamente.

---

## 5. Próximo Paso en el Roadmap

- **v385 (Topographic Attention en Transformers):** Llevar el acoplamiento Dirichlet 2D a las matrices de proyección ($W_q, W_k, W_v, W_o$) de un nano-Transformer autorregresivo (TinyStories / OpenWebText). Evaluar si la regularización topográfica induce mapas corticales semánticos y permite aplicar Spectral-LoRA para finetuning eficiente en modelos de lenguaje.
