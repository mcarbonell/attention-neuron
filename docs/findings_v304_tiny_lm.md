# Findings v304 — Tiny Language Modeling & Perplexity (Provisional)

> ⚠️ **DOCUMENTO PROVISIONAL:** Resultados preliminares basados en ejecuciones en Tiny Shakespeare (`v304_log.txt`, $n=1$, `seed=42`). Clasificado como **Nivel 1 (Sondeo Exploratorio)**. No citar como evidencia definitiva sin pasar los criterios de Nivel 2.

---

## 1. Contexto y Objetivos del Experimento v304

El experimento `v304` evalúa el comportamiento de **`ChunkwiseComplexDeltaPhase`** frente a **`ChunkwiseRealDeltaNetRectangular`** (control Iso-Floats) y **`CausalAttentionMHA`** (Softmax MHA) en una tarea de **Language Modeling en texto real** (Next-Token Prediction autorregresivo a nivel de caracteres sobre *Tiny Shakespeare*, 1,115,394 caracteres).

El objetivo es determinar si la atención de fase compleja en el círculo unitario $\mathbb{C}^{d_k}$ se generaliza a lenguaje natural reteniendo baja perplejidad de validación (PPL) en comparación con los controles reales y la atención estándar Softmax.

---

## 2. Resultados Provisionales ($d_k=32$, 15 Épocas, $L=256$)

### Tabla 1: Pérdida y Perplejidad (PPL) de Validación en Tiny Shakespeare

| Modelo | Parámetros | Val Loss | Val PPL ($e^{\text{Loss}}$) | Best LR |
| :--- | :---: | :---: | :---: | :---: |
| **`ChunkwiseRealDeltaNetRectangular`** (Iso-Floats) | 175,675 | **1.7811** | **5.94** | 0.004 |
| **`ChunkwiseComplexDeltaPhase`** | 144,331 | **1.7913** | **6.00** 🌟 | 0.004 |
| **`CausalAttentionMHA`** (Softmax MHA) | 141,883 | **1.8506** | **6.36** | 0.004 |

---

## 3. Hallazgos Principales

### 3.1 Superioridad de los Modelos Delta frente a Softmax MHA en Texto [SEÑAL]
Ambos modelos recurrentes DeltaNet (`ComplexDeltaPhase` con PPL **6.00** y `RealRectangular` con PPL **5.94**) superan la perplejidad de la atención Softmax estándar (`CausalAttentionMHA` con PPL **6.36**) en un margen de **-0.36 a -0.42 puntos de PPL**.

### 3.2 Paridad Competitiva Compleja vs Real bajo Iso-Floats [SEÑAL]
`ComplexDeltaPhase` (PPL **6.00**) logra un rendimiento prácticamente equivalente al control real `RealRectangular` (PPL **5.94**, diferencia de solo 0.06 nats de PPL), demostrando que la aritmética de fase compleja es **completamente estable y funcional en modelado de lenguaje natural**, sin sufrir divergencia ni inestabilidades numéricas.

---

## 4. Amenazas a la Validez

1. **Escala de Modelo y Vocabulario Pequeño:** El benchmark se ejecutó a nivel de caracteres ($Vocab=67$) con modelos mini de ~144k parámetros y 15 épocas corta.
2. **Semilla Única ($n=1$):** El experimento utiliza `seed=42`. La diferencia entre 5.94 y 6.00 es $< 0.1 \text{ nats}$ y requiere evaluación multi-semilla para verificar significancia estadística.
3. **Longitud de Contexto Acotada ($L=256$):** La longitud de contexto evaluada fue de 256 tokens. En secuencias más largas ($L \ge 1024$), la ventaja de la memoria asociativa de fase observada en MQAR ($v302$) podría hacerse más pronunciada.

---

## 5. Sugerencias de Siguientes Experimentos

1. **Escalado a Subwords / BPE:** Probar `ComplexDeltaPhase` con tokenización BPE (ej. Tiktoken / GPT-2 vocab de 50k tokens) en TinyStories.
2. **Evaluación de Contexto Largo en Texto ($L=1024$):** Medir la PPL en secuencias largas para evaluar si la ventaja de capacidad observada en $v302$ beneficia a los modelos de lenguaje en contextos extensos.
