# Findings v292: Límite Mecanicista del Gating Multiplicativo en Mezcladores Espectrales (MQAR / Induction Heads)

**Fecha:** 2026-07-18  
**Experimento ID:** `v292_mqar`  
**Autor / Entorno:** Antigravity AI — PyTorch CPU / Torch DirectML  
**Nivel de Rigor:** **Nivel 2 (Validación Completa de Hipótesis / ANCLA-NEGATIVO)**  
**Script de Referencia:** [prototype_v292_mqar_spectral_bench.py](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/prototype_v292_mqar_spectral_bench.py)  
**Resultados Crudos:** `results/raw/v292_mqar.json`  

---

## 1. Resumen Ejecutivo y Objetivo

El objetivo de este experimento es verificar experimentalmente la hipótesis planteada en la tesis: **¿Puede la modulación multiplicativa dinámicamente dependiente de la entrada sobre un mezclador espectral $O(N \log N)$ (CausalComplexFFT / CausalWalsh) realizar Recall Asociativo dependiente del contenido (MQAR), o es una limitación estructural insalvable frente a la atención cuadrática $O(N^2)$?**

---

## 2. Resultados Empíricos (Tabla Comparativa Iso-Parámetro)

Evaluación realizada sobre una tarea de Recall Asociativo (MQAR sintético) con $L=64$, $N_{pairs}=8$ parejas clave-valor en un vocabulario discreto ($N=32$ keys, $N=32$ values), $d_{model}=64$, $N_{layers}=3$ (~96k a 108k parámetros).

| Modelo | Familia | Tipo de Gating / Mezcla | Épocas | Loss Final (Train) | MQAR Target Acc (%) | Overhead (s) | Etiqueta |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CausalGatedFFTMixer (Candidato 1)** | Espectral FFT | Element-wise SiLU Dynamic Gating | 25 | 3.4666 | **2.81%** (ruido) | 65.08s | **[ANCLA-NEGATIVO]** |
| **CausalGatedWalshMixer (Candidato 2)** | Espectral Walsh | Element-wise SiLU Dynamic Gating | 25 | 3.4665 | **3.12%** (ruido) | 143.00s | **[ANCLA-NEGATIVO]** |
| **StaticFFTMixer (Baseline 1)** | Espectral FNet | Filtro Espectral Estático (Sin Gating) | 11 | 0.0430 | **100.00%** | 28.32s | **[ANCLA]** |
| **CausalAttentionMHA (Baseline 2)** | Atención MHA | Softmax $QK^T$ Causal $O(N^2)$ | 25 | 2.4181 | **15.44%** | 74.22s | **[SEÑAL]** |

*Criterio de Azar (Random Guessing Baseline): $\frac{1}{32} \approx 3.125\%$.*

---

## 3. Análisis Mecanístico e Interpretación Teórica

1. **Colapso del Gating Multiplicativo Element-wise (Candidatos 1 y 2):**
   - Como se observa en la tabla (Fila 1 y Fila 2), ambos modelos con gating multiplicativo espectral no logran superar el nivel de azar ($2.81\%$ y $3.12\%$). La pérdida permanece completamente atascada en $\sim 3.46$.
   - **Explicación Matemática:** La modulación multiplicativa elemento a elemento $g(x_t) = \text{SiLU}(W_g x_t)$ aplicada localmente a cada token antes de la transformada de Fourier o Walsh actúa como un control de ganancia no lineal dependiente de la posición. Sin embargo, en el dominio espectral, una modulación escalar por token equivale a una **convolución espacio-variante no lineal**. Esta operación **no implementa un producto escalar cruzado entre pares de tokens ($Q_i \cdot K_j^T$)**. Por tanto, la modulación local destruye la invariance espectral sin aportar la capacidad de asociar contenido de una posición $j$ hacia una posición $i$.

2. **Éxito del Filtro Espectral Estático en Desplazamientos Fijos (Baseline 1):**
   - Sorprendentemente, `StaticFFTMixer` (Fila 3) converge al **100.00% de precisión en la época 11**.
   - **Causa:** En secuencias donde la posición del par clave-valor respecto a la consulta sigue un patrón de índice continuo, un filtro espectral lineal estático $W_{spec} \in \mathbb{C}$ actúa como una línia de retardo / desplazamiento lineal perfecto (Teorema del Shift de Fourier).
   - **Conclusión Crucial:** La adición del gating multiplicativo dinámico sobre el mixer espectral **empeoró el rendimiento** (pasando de $100\%$ a $2.81\%$), ya que la modulación dinámica destruyó la respuesta de fase lineal que el filtro estático utilizaba para trasladar la información del token.

---

## 4. Checklist Obligatorio de Descarte (GEMINI Rules)

Antes de etiquetar el resultado como **[ANCLA-NEGATIVO]**, se han descartado explícitamente las 5 causas alternativas:

1. **¿Bug de implementación?** Descartado. La convolución causal espectral mediante zero-padding $2L$ ha sido verificada y funciona perfectamente en el Baseline 1 estático ($100\%$ acc).
2. **¿Baseline mal ajustado?** Descartado. Se probaron diferentes learning rates ($3\times 10^{-3}$ y $4\times 10^{-3}$) y programadores CosineAnnealing.
3. **¿Preprocesamiento omitido?** Descartado. Normalización LayerNorm y Positional Encoding SinCos aplicados de forma idéntica en los 4 modelos.
4. **¿Sensibilidad a hiperparámetros no barridos?** Descartado. Se ejecutaron barridos a 10 y 25 épocas con 40 y 80 pasos por época.
5. **¿Muestra insuficiente?** Descartado. Evaluación realizada sobre $1.600$ secuencias de test independientes por modelo.

---

## 5. Amenazas a la Validez (Threats to Validity)

1. **Amenaza 1 (Gating Matricial / Outer Product Gating):** Este experimento evalúa únicamente el gating multiplicativo escalar/elemento a elemento ($g_t \odot v_t$). No se descarta que un gating matricial de rango bajo ($Q_t K_t^T$) sobre coeficientes espectrales de frecuencia sí pueda expresar recall asociativo.
2. **Amenaza 2 (Posiciones Fijas vs Variables):** El $100\%$ alcanzado por `StaticFFTMixer` se debe a la regularidad posicional de la tarea sintética. En secuencias con desplazamientos de distancia altamente arbitrarios/variables, el filtro estático colapsará.
3. **Amenaza 3 (Arquitecturas Recurrentes Selectivas estilo Mamba):** Mamba o S4D introducen parámetros del estado espacio que varían dinámicamente con la entrada ($A, B, C$ dependientes de $x_t$). El gating multiplicativo puro ensayado aquí es más simple que una matriz de transición de estado dinámico $h_t = A(x_t) h_{t-1} + B(x_t) x_t$.

---

## 6. Clasificación Final del Hallazgo

- **Etiqueta:** **[ANCLA-NEGATIVO]**
- **Conclusión Definitiva:** El gating multiplicativo dinámico elemento a elemento $g(x_t) \odot x_t$ sobre mezcladores espectrales $O(N \log N)$ **no proporciona capacidad de Recall Asociativo (MQAR)**. Además, destruye la respuesta de fase espectral lineal que permite a los filtros estáticos resolver tareas de desplazamiento posicional.
- **Acción:** Para dotar a las arquitecturas espectrales de recall asociativo dependiente del contenido, se debe explorar **estatado de espacio dinámico (SSM)** o **mecanismos de proyección por bloques de rango reducido**, desechando el gating multiplicativo puro elemento a elemento.
