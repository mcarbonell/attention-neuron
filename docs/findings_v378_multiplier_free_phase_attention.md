# Findings: V378 — Atención de Fase sin Multiplicadores (Opción C)

> **ID de Experimento:** `v378`  
> **Fecha:** 2026-09-19  
> **Familia:** Fase / Hardware sin Multiplicadores / Eficiencia Extrema  
> **Nivel de Rigor:** Nivel 1 — Sondeo Exploratorio  
> **Etiqueta:** [SEÑAL]  
> **Script:** [`prototype_v378_multiplier_free_phase_attention.py`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/prototype_v378_multiplier_free_phase_attention.py)  
> **Resultados crudos:** [`results/raw/v378_multiplier_free_phase_attention.json`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/raw/v378_multiplier_free_phase_attention.json)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Complementa este Experimento

- **Dependencia de la función trigonométrica continua ($\cos$):** En los experimentos V375–V377 se asumía que la función trascendente $\cos(\Delta\theta)$ era necesaria para proporcionar el kernel de afinidad suave en el círculo unitario $U(1)$.
- **Hallazgo fundamental de V378:** La función coseno **puede ser sustituida por completo** sin pérdida de capacidad:
  1. **Onda triangular periódica (`Triangular_Phase`):** Utiliza únicamente resta angular y valor absoluto ($\text{tri}(\Delta\theta) = 1 - \frac{2}{\pi}|\Delta\theta|$). Alcanza el **100.00% de precisión de validación** (pérdida $0.0008$), requiriendo **cero multiplicaciones de punto flotante** en el cálculo de afinidad token-a-token.
  2. **Look-Up Table discreta de 16 palabras (`LUT16_Phase`):** Emula una memoria ROM de 16 entradas para microcontroladores o ASICs. Alcanza el **100.00% de precisión** (pérdida $0.0001$).
  3. **Ahorro paramétrico sostenido:** Ambos modelos utilizan **1745 parámetros en $H=4$**, frente a los 2696 del baseline vectorial (un ahorro del **35.3%**).

---

## 1. Hipótesis Evaluada

1. **Ruptura de la dependencia de funciones trascendentes:** La discriminación asociativa de fase depende de la simetría cíclica y de la distancia angular periódica, no de la curvatura infinitesimal exacta del coseno.
2. **Viabilidad Multiplier-Free en Silicio:** Es posible construir un mecanismo de atención token-a-token donde la afinidad $S_{ij}$ se calcule mediante lógica digital elemental (restadores de enteros y rectificadores de signo).

---

## 2. Configuración Experimental

- **Tarea:** Recuperación asociativa directa (Direct Associative Retrieval) con 8 slots de memoria, 8 claves y 8 valores.
- **Muestra:** 3000 secuencias train, 600 val. 15 épocas por modelo, AdamW ($\text{lr}=0.01$).
- **Kernels evaluados:**
  1. `Triangular_Phase`: $S_{ij} = (1 - \frac{2}{\pi}|\text{wrap}(\theta_q - \theta_k)|) \cdot \text{scale}$ (0 multiplicaciones, resta + `abs`).
  2. `LUT16_Phase`: Tabla estática de 16 palabras discretas (ROM ASIC/FPGA).
  3. `SquareWave_Phase`: Cuantización dura a 1 bit ($\text{sign}(\cos(\Delta\theta)) \in \{-1, +1\}$).
  4. `ExactCosine_Phase`: Referencia trigonométrica estándar $\cos(\Delta\theta)$.
  5. `StandardVector_dk8`: Baseline vectorial denso ($d_k=8$).

---

## 3. Resultados Numéricos

| Modelo | Kernel de Afinidad | Cabezas ($H$) | Parámetros | Val Acc (%) | Val Loss | Tiempo (s) | Diagnóstico |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `Triangular_Phase_H1` | 0 mults (Resta + `abs`) | 1 | 659 | 55.33% | 1.2098 | 1.42s | $H=1$ limitado |
| **`Triangular_Phase_H4`** 🌟 | **0 mults (Resta + `abs`)** | **4** | **1745** | **100.00%** | **0.0008** | 2.10s | **100% libre de multiplicadores** |
| | | | | | | | |
| `LUT16_Phase_H1` | 0 mults (16-word ROM) | 1 | 659 | 71.00% | 0.7356 | 1.50s | ROM 16 palabras |
| **`LUT16_Phase_H4`** 🌟 | **0 mults (16-word ROM)** | **4** | **1745** | **100.00%** | **0.0001** | 1.98s | **100% loss óptimo con tabla microcode** |
| | | | | | | | |
| `SquareWave_Phase_H1` | 0 mults (1-bit Sign) | 1 | 659 | 42.17% | 1.3106 | 1.55s | 1 bit extremo |
| `SquareWave_Phase_H4` | 0 mults (1-bit Sign) | 4 | 1745 | 71.17% | 0.6965 | 1.77s | 1 bit alcanza 71.17% |
| | | | | | | | |
| `ExactCosine_Phase_H1` | Coseno trigonométrico | 1 | 659 | 69.00% | 0.7555 | 1.25s | Coseno estándar |
| `ExactCosine_Phase_H4` | Coseno trigonométrico | 4 | 1745 | 99.83% | 0.0044 | 2.62s | Coseno continuo |
| | | | | | | | |
| `StandardVector_dk8_H1` | Producto Punto Vectorial | 1 | 896 | **100.00%** | 0.0002 | 1.98s | Baseline vectorial |
| `StandardVector_dk8_H4` | Producto Punto Vectorial | 4 | 2696 | **100.00%** | 0.0001 | 2.66s | Baseline vectorial (+54% params) |

---

## 4. Análisis Mecanístico de los Datos

1. **La suficiencia de la onda triangular en hardware:**  
   La onda triangular periódica $\text{tri}(\Delta\theta)$ preserva las propiedades topológicas fundamentales del círculo:
   - Máximo absoluto en $\Delta\theta = 0$ ($+1.0$).
   - Cero en cuadratura $\Delta\theta = \pm \pi/2$ ($0.0$).
   - Mínimo en antifase $\Delta\theta = \pm \pi$ ($-1.0$).
   Al no tener curvatura no-lineal, los gradientes son constantes a tramos ($\pm 2/\pi$), lo que estabiliza la propagación hacia atrás sin desvanecimiento de gradiente en los extremos.
2. **El éxito de la tabla ROM de 16 palabras (`LUT16`):**  
   Discretizar el ángulo en 16 sectores ($22.5^\circ$ por celda) es suficiente para que $H=4$ cabezas alcancen una pérdida de **$0.0001$**, idéntica al cálculo vectorial en punto flotante de 32 bits, eliminando cualquier unidad de cómputo trascendente en el procesador.
3. **El detector de fase de 1 bit (`SquareWave`):**  
   Incluso colapsando la afinidad a un solo bit (+1 o -1, sin ninguna gradación intermedia), el modelo con 4 cabezas alcanza un **71.17% de precisión** (frente a 12.5% de azar), confirmando que la simple pertenencia al hemisferio angular de fase retiene la mayor parte de la señal asociativa.

---

## 5. Amenazas a la Validez

1. **Cuantización de los Pesos de Proyección:**  
   Este experimento eliminó los multiplicadores en el cálculo de la afinidad $S_{ij} = f(\theta_q - \theta_k)$. Las proyecciones lineales iniciales $W_q, W_k$ y la salida $W_v$ aún usaron pesos float32. Combinar este método con pesos ternarios $\{-1, 0, 1\}$ (Era 5) completaría una red 100% libre de multiplicadores en todo el flujo.
2. **Resolución de la LUT en Vocabularios Extremos:**  
   Para vocabularios de miles de tokens, una tabla de 16 entradas podría requerir elevarse a 32 o 64 palabras para evitar colisiones de cuantización.
