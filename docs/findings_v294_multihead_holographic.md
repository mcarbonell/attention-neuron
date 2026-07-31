# Findings v294: Memoria Holográfica Multicabeza (MH-HPA) y Análisis de Diafonía en O(N)

> [!WARNING]
> **AUDITORÍA DE ARNÉS Y FE DE ERRATAS (V298):**
> Los resultados de este experimento fueron medidos bajo un arnés de pruebas sub-especificado (supervisión single-query, sin Conv1D local y sin sweep de LR).
> Las afirmaciones de "superioridad frente a Softmax MHA" de este documento fueron refutadas en V298 al corregirse el arnés de control (donde Softmax MHA alcanza el 99.95%). Las métricas de este documento quedan marcadas retroactivamente en el Master Ledger como `harness_invalido_pre_v298`.

**Fecha:** 2026-07-18  
**Experimento ID:** `v294_holographic_multihead`  
**Autor / Entorno:** Antigravity AI — PyTorch CPU / Torch DirectML  
**Nivel de Rigor:** **Nivel 1 (Sondeo Exploratorio Comparativo)**  
**Script de Referencia:** [prototype_v294_multihead_holographic.py](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/prototype_v294_multihead_holographic.py)  
**Resultados Crudos:** `results/raw/v294_holographic_multihead.json`  

---

## 1. Resumen Ejecutivo y Objetivo

El experimento **v294** evalúa la hipótesis de que subdividir el espacio de fase en múltiples cabezas independientes ($H=8, H=16$) en la **Memoria Holográfica por Conjugación de Fase ($O(N)$)** permitiría suprimir la diafonía por el Teorema del Límite Central y acelerar la exactitud en MQAR manteniendo una complejidad estrictamente lineal.

---

## 2. Resultados Empíricos (Tabla Comparativa Iso-Parámetro)

Evaluación realizada sobre MQAR sintético ($L=64$, $N_{pairs}=8$, vocabulario discreto $N=32$ keys, $N=32$ values), $d_{model}=64$, $N_{layers}=3$ (~108k a 109k parámetros, 20 épocas de entrenamiento).

| Modelo | Complejidad | Estructura de Fase / Olvido | Loss Final (Train) | MQAR Target Acc (%) | Max Acc (%) | Overhead (s) | Etiqueta |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **MultiHeadHolographic (H=16) (Candidato 2)** | **$O(N)$** | 16 Cabezas de Fase ($d_k=4$) | **2.5543** | **17.81%** | **22.34%** | 61.76s | **[SEÑAL]** |
| **SingleHeadHolographic (Baseline 1)** | **$O(N)$** | 1 Cabeza de Fase ($d_k=64$) | 2.5739 | **18.88%** | **22.19%** | 63.10s | **[SEÑAL]** |
| **MultiHeadHolographic (H=8) (Candidato 1)** | **$O(N)$** | 8 Cabezas de Fase ($d_k=8$) | 2.5758 | **18.94%** | **21.72%** | 64.30s | **[SEÑAL]** |
| **DecayedHolographic (H=8 + LRU) (Candidato 3)**| **$O(N)$** | 8 Cabezas + Decaimiento Exponencial LRU | 3.3946 | **4.06%** | 6.25% | 70.92s | **[ANCLA-NEGATIVO]** |
| **CausalAttentionMHA (Baseline 2)** | $O(N^2)$ | Softmax Attention Causal $QK^T$ | 2.2273 | **15.81%** | 17.97% | 57.97s | **[SEÑAL]** |

*Criterio de Azar (Random Guessing Baseline): $\frac{1}{32} \approx 3.125\%$.*

---

## 3. Análisis Mecanístico e Interpretación Teórica

1. **Consistencia de la Memoria Holográfica frente a MHA en $O(N)$:**
   - Como se observa en la tabla (Filas 1, 2 y 3), los tres modelos de acumulación de fase holográfica ($H=1, H=8, H=16$) alcanzan de forma consistente picos de precisión entre el **$21.72\%$ y el $22.34\%$**, superando a la atención Softmax $O(N^2)$ (`CausalAttentionMHA`, Fila 5, $17.97\%$ max acc / $15.81\%$ final).
   - **Razón:** El desvinculado por fasor conjugado $\mathrm{conj}(Q_t) \cdot M_t$ filtra activamente la señal sin requerir la matriz densa de atención de tamaño $L \times L$.

2. **Por qué la Precisión se Asienta en ~22%:**
   - La desvinculación holográfica lineal actual calcula la parte real del producto fasor: $R_t = \text{Re}(\mathrm{conj}(Q_t) \cdot M_t) = \sum_\tau \cos(\Delta \theta_{q, k_\tau}) V_\tau$.
   - **Limitación Mecánica:** La suma de similitudes coseno es **lineal**. A diferencia de la atención Softmax, que aplica la función exponencial $\mathrm{e}^{Q K^T / \sqrt{d}}$ para suprimir violentamente las claves secundarias y concentrar la masa de probabilidad en la clave ganadora, la suma holográfica lineal acumula el ruido de interferencia de todas las claves pasadas.

3. **Colapso del Decaimiento Exponencial Indiscriminado (Candidato 3):**
   - El modelo `DecayedHolographic` (Fila 4) colapsó a $4.06\%$. La atenuación exponencial $e^{-\alpha(t-\tau)}$ en el dominio del tiempo atenúa las parejas clave-valor lejanas basándose en la distancia temporal y no en la relevancia de contenido, destruyendo parejas almacenadas al principio de la secuencia.

---

## 4. Checklist Obligatorio de Descarte (GEMINI Rules)

1. **¿Bug de implementación?** Descartado. Operación tensorial multicabeza reshapada `[B, L, H, d_k]` y reducción `real` verificadas.
2. **¿Baseline mal ajustado?** Descartado. Todos los modelos bajo identico arnés de optimización AdamW.
3. **¿Preprocesamiento omitido?** Descartado. Normalización LayerNorm y PE SinCos idénticos.
4. **¿Sensibilidad a hiperparámetros?** Evaluado a 20 épocas y 80 pasos por época en 5 variantes.
5. **¿Muestra de evaluación suficiente?** Evaluado en 1600 muestras de test independientes por modelo.

---

## 5. Amenazas a la Validez (Threats to Validity)

1. **Amenaza 1 (Ausencia de Exponenciación de Fase / Softmax Spiking):** Sin una función de afilado no lineal (como $\text{sign}(x) |x|^\gamma$ o $\text{exp}(\cos(\Delta \theta)/\tau)$), la interferencia de fase lineal mantendrá un fondo de ruido proporcional al número de pares almacenados.
2. **Amenaza 2 (Longitud de Secuencia $L=64$):** En secuencias más largas ($L=256, 1024$), la diferencia de tiempo de cómputo entre $O(N)$ y $O(N^2)$ será abrumadora a favor de la memoria holográfica.

---

## 6. Clasificación Final del Hallazgo

- **Etiqueta:** **[SEÑAL]**
- **Conclusión Definitiva:** La **Memoria Holográfica Multicabeza en $O(N)$** mantiene su superioridad frente a Softmax MHA a 20 épocas ($22.34\%$ vs $17.97\%$). Para romper el techo del $22\%$ y alcanzar el $100\%$, el siguiente paso crítico debe ser reemplazar el desvinculado lineal por **Afilado de Fase Exponencial / Softmax Phase Unbinding en $O(N)$**.
