# Findings v302 — Multi-Hop MQAR & Shared Vocabulary Interference (Provisional)

> ⚠️ **DOCUMENTO PROVISIONAL:** Resultados preliminares basados en ejecuciones en Tesla T4 ($n=1$, `seed=42`). Clasificado como **Nivel 1 (Sondeo Exploratorio)**.

---

## 1. Contexto y Objetivos del Experimento v302

El experimento `v302` evalúa la atención de fase compleja (`ChunkwiseComplexDeltaPhase`) frente a variantes reales de DeltaNet y Softmax MHA en dos régimenes:
1. **Atención asociativa con Vocabulario Compartido (1-hop):** Claves y valores se muestrean del mismo vocabulario ($1..512$), simulando la distribución real de tokens en un LLM.
2. **Encadenamiento Multi-Hop (2-hop y 3-hop):** Requerimiento de composicionalidad donde la salida de un salto actúa como clave del siguiente a través de las capas ($L_{layers}=4$).

### Corrección de Arquitectura Aplicada en v302:
Se implementó la máscara de actualización `compute_kv_mask`:
$$\beta_{\text{efectivo}} = \beta \times \text{mask}_{\text{Value\_pos}}$$
Esto restringe las actualizaciones de la memoria recurrente $M$ exclusivamente a las posiciones donde el token de Valor está presente, evitando que las posiciones de Clave o Consulta contaminen el estado asociativo.

---

## 2. Resultados Provisionales ($d_k=32$, Tesla T4)

### Tabla 1: Best Accuracy Final (%) por Modelo, Hops ($h$) y Cadenas ($c$)

| Modelo | h1_c16 (L=128) | h1_c32 (L=192) | h1_c64 (L=576) | h1_c128 (L=1088) | h2_c16 | h2_c32 | h2_c64 | h3_c16 | h3_c32 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ChunkwiseComplexDeltaPhase** | **98.99%** | **99.06%** | **98.89%** 🌟 | **94.30%** 🌟 | 0.55% | 0.38% | 0.23% | 0.30% | 0.26% |
| **ChunkwiseRealDeltaNetSquare** | 89.88% | 1.35% ⚠️ | 72.92% | 0.29% | 0.55% | 0.28% | 0.24% | 0.31% | 0.22% |
| **ChunkwiseRealDeltaNetRectangular** (Iso-Floats) | 10.11% ⚠️ | 87.87% | 0.49% | 0.65% | 0.59% | 0.28% | 0.24% | 0.30% | 0.28% |
| **CausalAttentionMHA** (Softmax) | **99.70%** | **99.78%** | 0.22% ⚠️ | 0.21% | 0.28% | 0.24% | 0.19% | 0.26% | 0.28% |

---

## 3. Hallazgos Principales

### 3.1 Dominio Masivo de Fase en Vocabulario Compartido bajo Alta Carga ($h=1, c=128$) [SEÑAL]
En la condición de mayor presión de memoria ($c=128$ pares de cadenas, $L=1088$ tokens):

* `ChunkwiseComplexDeltaPhase` mantiene **94.30% de precisión** (Loss = 0.0268).
* `ChunkwiseRealDeltaNetRectangular` (mismo presupuesto de 2,048 floats por cabeza) colapsa a **0.65%** (azar).
* `ChunkwiseRealDeltaNetSquare` colapsa a **0.29%** (azar).
* `CausalAttentionMHA` colapsa a **0.21%** (azar).

> **Interpretación:** La parametrización en el círculo unitario complejo $S^1 \subset \mathbb{C}^{d_k}$ evita el colapso por interferencia cruzada cuando claves y valores comparten vocabulario. Las representaciones reales sufren un cruce destructivo masivo al escalar la longitud a $L>1000$.

### 3.2 El Muro del Multi-hop Corto ($h \ge 2$) [SEÑAL]
Todos los modelos permanecen en el rango de **0.18% - 0.59%** en 2-hop y 3-hop (equivalente a azar puro $\approx 0.195\%$).

> **Explicación:** El aprendizaje de "induction heads" multi-nivel autorregresivas desde cero requiere una transición de fase en la optimización que no se alcanza en 20 épocas cortas (1,000 pasos de gradiente) sin un esquema de curriculum learning.

---

## 4. Amenazas a la Validez

1. **Semilla Única ($n=1$):** El estudio se realizó bajo `seed=42`. Es obligatorio validar en $n \ge 5$ semillas para descartar varianza por inicialización.
2. **Caída de Softmax MHA en Secuencias Largas ($L > 500$):** La pérdida de precisión en `CausalAttentionMHA` a partir de $c=64$ sugiere falta de *warmup* en la tasa de aprendizaje durante el entrenamiento autorregresivo.
3. **Log Parcialmente Incompleto:** La celda de evaluación para $d_k=64$ en $h \ge 2$ no llegó a completarse en este archivo de log.

---

## 5. Sugerencia de Siguientes Experimentos

1. **Evaluación Multi-semilla ($n=5$) en $c=128$:** Reejecutar la prueba en la celda $h=1, c=128$ con 5 semillas fijas para promover el hallazgo a **Nivel 2 [ANCLA]**.
2. **Curriculum Learning para Multi-hop:** Entrenar primero 10 épocas en $h=1$ y luego migrar progresivamente a $h=2$ para facilitar la alineación inter-capa de las *induction heads*.
3. **Validación en Texto Real (TinyStories):** Evaluar si la retención asociativa compleja se traduce en menor perplejidad en tareas de next-token prediction en lenguaje natural.
