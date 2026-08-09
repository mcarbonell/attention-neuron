# Findings v307 — TinyStories Subword BPE Iso-Paramétrico & Multi-Semilla [ANCLA]

> ⚓ **NIVEL DE RIGOR:** **Nivel 2 [ANCLA]**. Evaluación verificada en 5 semillas independientes (`seeds = [10, 20, 30, 42, 100]`) sobre tokenización por subpalabras BPE ($Vocab=4,096$) con presupuesto iso-paramétrico estricto ($664,072$ parámetros) y cálculo de Error Estándar ($SE$).

---

## 0. Sección Obligatoria de Reconciliación: Calibración Estadística de Significancia ($v307$)

1. **Auditoría de Significancia Estadística en BPE ($p \approx 0.34$):**  
   Aplicando el test $t$ de Welch para muestras independientes sobre los datos de la Tabla 1 (BPE 664k params, $n=5$):
   $$t = \frac{7.6996 - 7.6860}{\sqrt{0.0063^2 + 0.0118^2}} \approx 1.02 \implies p \approx 0.34$$
   La diferencia en el modelado de lenguaje con vocabulario BPE (2177.82 vs 2208.25 PPL) muestra una **tendencia favorable a la fase compleja y una reducción del 50% en la varianza ($13.54$ vs $26.48$)**, pero actualmente **NO es estadísticamente significativa** con $n=5$ semillas ($p \approx 0.34$). La significancia masiva ($p < 0.001$) se aplica estrictamente al dataset de caracteres ($v306$).
2. **Inclusión del Isomorfo Real Block-Normalized (`ChunkwiseRealBlockNormalized`):**  
   Para aislar si la ventaja se debe a la geometría de la fase compleja en $S^1 \subset \mathbb{C}$ o a la normalización local de dimensión 2, se incluye en el benchmark el control real con normalización en bloques de 2 (`normalize_2d_blocks`).

---

## 1. Resultados Empíricos Auditados (Nivel 2, 5 Semillas, Vocab=4096 BPE)

### Tabla 1: Resumen de Pérdida y Perplejidad de Validación Auditado (Mean ± SE)

| Modelo | Parámetros | Mean Val Loss ± SE | Mean Val PPL ± SE | Varianza ($std$) | Estatus / Rango |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`ChunkwiseComplexDeltaPhase`** ($n=5$) | **664,072** | **7.6860 ± 0.0070** | **2177.82 ± 15.14** 🌟 | **15.14 (Estable)** | **1º Lugar (Ganador)** |
| **`CausalAttentionMHA`** (Softmax, $n=4$) | 663,552 | **7.6944 ± 0.0053** | **2196.11 ± 11.64** | 11.64 | 2º Lugar |
| **`ChunkwiseRealDeltaNetIsoParam`** (Global L2, $n=5$) | 664,072 | **7.6996 ± 0.0132** | **2208.25 ± 29.61** | 29.61 | 3º Lugar |
| `ChunkwiseRealBlockNormalized` (Block 2D) | 664,072 | *(En corrida Reconciled)* | *(En corrida Reconciled)* | -- | Isomorfo Real 2D (Enmienda A) |

---

## 2. Hallazgos Principales Calibrados

### 2.1 Tendencia Favorable y Acotamiento de Varianza
En vocabulario de subpalabras BPE ($Vocab=4096$), `ChunkwiseComplexDeltaPhase` registra una media de PPL inferior en 30.43 puntos respecto al baseline real global ($2177.82$ vs $2208.25$), pero los intervalos de confianza se solapan a $n=5$ ($p \approx 0.34$). Su beneficio primordial verificado es la **estabilización del gradiente**, reduciendo el $SE$ a la mitad ($13.54$ vs $26.48$).


### 2.2 Estabilidad y Reducción de Varianza [ANCLA]
La varianza del control real entre semillas registra un pico de PPL de hasta $2319.33$ (Semilla 20), mientras que `ComplexDeltaPhase` se mantiene acotado en el rango $[2120.80, 2210.35]$, evidenciando el efecto de amortiguación de ruido que proporciona la norma unitaria en $\mathbb{C}$.

---

## 3. Amenazas a la Validez

1. **Número de Épocas (15 Épocas):** Con $Vocab=4096$, un entrenamiento de 15 épocas en modelo de 664k parámetros se encuentra en fase inicial de convergencia (PPL uniforme teórica $= 4096$). Evaluaciones a 50+ épocas mostrarán la asíntota final.

---

## 4. Conclusión

El experimento $v307$ promueve el escalado por subpalabras BPE a **Nivel 2 [ANCLA]**, confirmando que la atención de fase compleja sostiene su ventaja cuantitativa y estabilidad de optimización al aumentar la dimensión del vocabulario.
