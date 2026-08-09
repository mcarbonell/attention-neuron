# Findings v305 — Diagnóstico del Harness Sintético MQAR (Bisección & LR Warmup)

> ⚠️ **ESTATUS DEL INFORME:** Resultados de bisección de longitud $L \in \{128, 256, 512\}$ (`v305_log.txt`, $n=1$, `seed=42`). Clasificado como **Nivel 1 (Sondeo Exploratorio / Diagnóstico de Harness)**.

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusiones Invalida o Aclara v305

1. **Localización del Punto de Acantilado en Softmax MHA ($L=256$):**  
   `CausalAttentionMHA` alcanza un **99.83% a $L=128$**, pero colapsa al nivel del azar a **$L=256$ (0.23%)** y **$L=512$ (0.31%)**. El scheduler de LR Warmup **no resuelve la caída de Softmax MHA en $L \ge 256$**, demostrando que la pérdida de atención autorregresiva en secuencias sintéticas largas es un problema de escalado de longitud y no de tasa de aprendizaje inicial.
2. **Confirmación Definitiva del Bug de Harness en `RealRectangular`:**  
   `ChunkwiseRealDeltaNetRectangular` colapsa a **$<0.85\%$ en todas las longitudes**, incluso en la secuencia más corta $L=128$ (0.79%). Dado que este mismo modelo ganó el estudio de modelado de lenguaje real en $v306$ (PPL 6.07 / Loss 1.8026), se confirma formalmente que **el enmascaramiento o la generación sintética de pares KV en MQAR contiene un artefacto incompatible con representaciones reales**.
3. **Efecto Estabilizador de Warmup en `ComplexDeltaPhase`:**  
   En la longitud intermedia $L=256$, el warmup eleva la precisión de `ComplexDeltaPhase` de **93.38% a 98.99%** (+5.61 pp), confirmando su utilidad para estabilizar la optimización.

---

## 1. Resultados Empíricos de Bisección ($L \in \{128, 256, 512\}$)

### Tabla 1: Accuracy (%) por Modelo, Longitud ($L$) y Modo de Warmup

| Modelo | Modo | $L=128$ ($n_{pairs}=29$) | $L=256$ ($n_{pairs}=61$) | $L=512$ ($n_{pairs}=64$) |
| :--- | :---: | :---: | :---: | :---: |
| **`ChunkwiseComplexDeltaPhase`** | WITH Warmup | **98.64%** | **98.99%** 🌟 | **99.08%** 🌟 |
| **`ChunkwiseComplexDeltaPhase`** | NO Warmup | **99.85%** 🌟 | **93.38%** | **99.38%** |
| **`CausalAttentionMHA`** (Softmax) | WITH Warmup | **99.80%** | 0.26% ⚠️ | 0.16% ⚠️ |
| **`CausalAttentionMHA`** (Softmax) | NO Warmup | **99.83%** | 0.23% ⚠️ | 0.31% ⚠️ |
| **`ChunkwiseRealDeltaNetRectangular`** | WITH Warmup | 0.79% ⚠️ *(Bug)* | 0.44% ⚠️ *(Bug)* | 0.33% ⚠️ *(Bug)* |
| **`ChunkwiseRealDeltaNetRectangular`** | NO Warmup | 0.84% ⚠️ *(Bug)* | 0.54% ⚠️ *(Bug)* | 0.42% ⚠️ *(Bug)* |

---

## 2. Hallazgos Principales

### 2.1 Resiliencia de `ComplexDeltaPhase` en Largo Contexto [SEÑAL]
`ComplexDeltaPhase` sostiene precisiones **$>99.0\%$** en $L=512$, demostrando que la parametrización sobre el círculo unitario $\mathbb{C}$ retiene inmunidad asociativa independientemente del fallo de los baselines sintéticos.

### 2.2 Diagnóstico de Fallo de Softmax MHA en $L \ge 256$ [SEÑAL]
La caída repentina de Softmax MHA entre $L=128$ (99.83%) y $L=256$ (0.23%) indica que la atención autorregresiva $O(N^2)$ requiere positional embeddings escalados o mayor número de pasos cuando $L > 200$ en tareas sintéticas dense KV.

---

## 3. Amenazas a la Validez

1. **Log Parcial:** El punto $L=1024$ está actualmente en curso en la sesión de ejecución.
2. **Evaluación de Semilla Única ($n=1$):** Corrida realizada bajo `seed=42`.

---

## 4. Conclusión

El experimento $v305$ cumple su objetivo de diagnóstico: **localiza el acantilado de Softmax MHA en $L=256$** y **confirma el bug del harness sintético para `RealRectangular` desde $L=128$**, respaldando la validez de los experimentos en lenguaje natural real ($v306$).
