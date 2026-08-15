# Findings v349: Complex Beta-Gated Householder Core (\beta_t = 1 + e^{i\varphi_t}) & Z_k Group Audit

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento
- **Prueba del Autovalor en el Círculo Unitario ($S^1$):** El experimento confirma la hipótesis teórica sugerida en la revisión externa: la parametrización $\beta_t = 1 + e^{i\varphi_t}$ garantiza autovalores unitarios $|\lambda| = 1.0$ (matriz de Householder generalizada en el plano complejo $U(d)$).
- **Evaluación en Aritmética Modular ($\mathbb{Z}_7$):** Se realizó la primera prueba empírica de adición acumulativa modular $\mathbb{Z}_7$ ($L=64$).

## 1. Resumen Ejecutivo
Se comparó `Real Beta DeltaNet` (autovalores reales $1 - \beta \in (-1, 1)$, paridad $\mathbb{Z}_2$) frente a `Complex Beta DeltaPhase` ($\beta_t = 1 + e^{i\varphi_t}$, autovalores complejas $-e^{i\varphi_t} \in S^1$, representación de grupos cíclicos $\mathbb{Z}_k$).

### Tabla de Resultados en Aritmética Modular ($\mathbb{Z}_7$)
| Modelo | Autovalores | Formulación $\beta_t$ | Accuracy Final (%) | Nivel de Azar (%) | Etiqueta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Real Beta DeltaNet** | Reales ($1-\beta \in (-1,1)$) | $\beta_t = 2.0 \cdot \text{sigmoid}(W_\beta x_t)$ | 18.59% | 14.29% | [SEÑAL] |
| **Complex Beta DeltaPhase** 🌟 | Complejos ($-e^{i\varphi_t} \in S^1$) | $\beta_t = 1.0 + e^{i\varphi_t}$ | **19.88%** | 14.29% | [SEÑAL] |

*(Brecha a favor de Complex Beta: +1.30% en prueba preliminar de 600 pasos en CPU).*

## 2. Inventario de Arquitectura y Parámetros
* **Dimensiones:** $d_{\text{model}} = 64$, $n_{\text{layers}} = 2$, $n_{\text{heads}} = 2$, $d_k = 16$.
* **Embeddings:** Token Embedding ($7 \times 64 = 448$ params).
* **Bloques Recurrentes ($2 \times$):**
  - Real Beta: Proyecciones $K, Q, V$ ($3 \times (64 \times 32 + 32) = 6,240$), $\beta$ ($64 \times 2 + 2 = 130$), Out ($32 \times 64 + 64 = 2,112$), Norms ($256$). FFN SiLU ($64 \to 128 \to 64 = 16,576$). Total por capa: $25,314$ params.
  - Complex Beta: Proyecciones $\theta_K, \theta_Q, V$ ($6,240$), $\varphi_\beta$ ($130$), Out ($2,112$), Norms ($256$), FFN ($16,576$). Total por capa: $25,314$ params.
* **Head:** Linear($64 \to 7$, $455$ params).
* **Total Parámetros:** **51,531 parámetros** (Iso-paramétrico estricto entre Real y Complejo).

## 3. Amenazas a la Validez
1. **Profundidad del Modelo:** Prueba de 2 capas sin convolución causal ni encodings posicionales.
2. **Número de Pasos:** 600 pasos en CPU es el umbral inicial de convergencia.

## 4. Conclusión
La parametrización $\beta_t = 1 + e^{i\varphi_t}$ es numéricamente estable por construcción ($|\lambda| = 1.0$) y extiende la regla delta al grupo unitario $U(d)$ en 1 solo paso por token con exactamente los mismos parámetros que el baseline real.

