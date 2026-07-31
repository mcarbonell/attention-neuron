# Findings v297: Content-Selective Phase Softmax Memory en O(N)

**Fecha:** 2026-07-21  
**Experimento ID:** `v297_phase_softmax`  
**Autor / Entorno:** Antigravity AI — PyTorch CPU / Torch DirectML  
**Nivel de Rigor:** **Nivel 1 (Sondeo Exploratorio — [SEÑAL])**  
**Script de Referencia:** [prototype_v297_phase_softmax_mqar.py](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/prototype_v297_phase_softmax_mqar.py)  
**Resultados Crudos:** `results/raw/v297_phase_softmax.json` (consolidados en [findings_v298_delta_phase_mqar.md](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/docs/findings_v298_delta_phase_mqar.md))  

---

## 1. Resumen Ejecutivo

El experimento **v297** investigó la incorporación de un **mecanismo de Softmax Selectivo por Contenido sobre memoria de fase completa en tiempo lineal $O(N)$**. 

Tras el hallazgo de v296 (donde la normalización de masa causal alcanzó un 23.59% de exactitud frente a la acumulación lineal pura), v297 introdujo:
1. **Forget Gate Selectivo Causal (Scan Causal):** $\lambda_t = \text{sigmoid}(W_\lambda x_t)$ para atenuar interferencias de estados pasados.
2. **Normalización por Fuerza de Coincidencia de Fase (Softmax Proxy):** Acumulación del módulo de clave $S_t = |\bar{Q}_t M_{K,t}|$ como función de partición.
3. **Agudización por Potencia (Power Sharpening Proxy):** $\text{sign}(u) |u|^\gamma$ ($\gamma=3.0$) para filtrar diafonía ruidosa entre claves ortogonales.

Los resultados confirmaron una mejora sustancial alcanzando un **49.59% de exactitud** en la tarea Multi-Query Associative Recall (MQAR), más que duplicando el récord anterior de v296 (23.59%). Sin embargo, se observó un **techo infranqueable en torno al 50%**, derivado de la acumulación Hebbiana lineal sin retroalimentación de error residual (lo que posteriormente motivó la Regla Delta en v298).

---

## 2. Resultados Empíricos

Evaluación sobre MQAR sintético Multi-Query ($L=64$, $N_{pairs}=8$ parejas clave-valor en un vocabulario de $N=32$ keys y $N=32$ values), $d_{model}=64$, $N_{layers}=3$ (~108k a 122k parámetros).

| Modelo | Complejidad | Mecanismo de Memoria / Mezcla | Best LR | Épocas | MQAR Target Acc (%) | Etiqueta |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **PhaseSoftmaxHolographic (Candidato 1)** | **$O(N)$** | **Selective Scan + Phase Match Norm** | **$4\times 10^{-3}$** | **15** | **49.59%** | **[SEÑAL]** |
| **SharpenedPhaseSoftmaxHolographic (Candidato 2)** | **$O(N)$** | Selective Scan + Power Sharpening ($\gamma=3$) | $4\times 10^{-3}$ | 15 | **38.12%** | **[SEÑAL]** |
| **GatedMassNormalizedHolographic (Baseline 1 - v296)** | **$O(N)$** | Mass-Normalized Linear Accumulation | $4\times 10^{-3}$ | 15 | **23.59%** | **[SEÑAL]** |
| **CausalAttentionMHA (Baseline 2)** | $O(N^2)$ | Softmax MHA Causal ($QK^T$) | $4\times 10^{-3}$ | 15 | **15.47%** | **[RUIDO-SOSPECHA]** |

*Nota: CausalAttentionMHA en este arnés no incluía la Conv1D causal local pre-mezcla que posteriormente se añadió en v298.*

---

## 3. Análisis Mecanístico e Interpretación Teórica

1. **Beneficio del Forget Gate Selectivo:**
   - La inclusión de $\lambda_t = \text{sigmoid}(W_\lambda x_t)$ permite atenuar selectivamente memorias antiguas cuando entran paridades clave-valor irrelevantes, explicando el salto del 23.59% al 49.59%.

2. **Diagnóstico del Techo del 50% (Diafonía Residual):**
   - Aunque la normalización por la fuerza de coincidencia de fase $S_t = |\bar{Q}_t M_{K,t}|$ intenta simular la función de partición del Softmax, la matriz de memoria $M_t = \lambda_t M_{t-1} + (1-\lambda_t) (K_t \otimes V_t)$ sigue sumando fasores linealmente.
   - Cuando el número de parejas almacenadas aumenta ($N_{pairs}=8$), la suma Hebbiana genera diafonía espacial entre fasores no ortogonales. Sin un mecanismo de corrección de error residual (como la ortogonalización por Regla Delta de v298), la precisión se satura al 50%.

3. **Incapacidad del Power Sharpening ($\gamma=3.0$):**
   - Aplicar agudización no lineal $\text{sign}(u)|u|^\gamma$ degradó el rendimiento del 49.59% al 38.12%, demostrando que la causa del fallo no era la falta de nitidez en la atención, sino la interferencia acumulada en el estado de memoria.

---

## 4. Checklist Obligatorio de Descarte

1. **¿Bug de implementación?** Descartado. Implementación verificada contra `torch.fft` y fasores complejos vectorizados.
2. **¿Baseline de comparación ajustado?** Descartado. Múltiples LRs evaluados ($1\times 10^{-3}$ a $8\times 10^{-3}$).
3. **¿Sensibilidad a hiperparámetros?** Variaciones con $\gamma=2, 3, 5$ mostraron degradación sistemática con mayor agudización.
4. **¿Muestra de evaluación suficiente?** Evaluado sobre 1600 muestras Multi-Query independientes.

---

## 5. Amenazas a la Validez y Lección Aprendida

- **Lección Fundamental:** La aproximación a Softmax mediante gating escalar y normalización de módulo en suma Hebbiana es insuficiente para recall asociativo perfecto en $O(N)$. Se requiere una regla de actualización de estado basada en gradiente/error residual.
- **Transición Histórica:** Este hallazgo (y la identificación exacta de la diafonía Hebbiana) sirvió como el escalón teórico necesario para formular la **Regla Delta Matricial de v298**, que resolvió el problema alcanzando un **99.95%** en $O(N)$.
