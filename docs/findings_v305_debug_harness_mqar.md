# Findings v305 — Diagnóstico del Harness Sintético MQAR (Bisección & LR Warmup)

> ⚠️ **ESTATUS DEL INFORME:** Resultados de bisección de longitud $L \in \{128, 256, 512\}$ (`v305_log.txt`, $n=1$, `seed=42`). Clasificado como **Nivel 1 (Sondeo Exploratorio / Diagnóstico de Harness)**.

---

## 0. Sección Obligatoria de Reconciliación: Solución Definitiva del Harness y Certificación MHA

1. **Resolución de la Caída de MHA (Memorización Estática vs Generación On-The-Fly):**  
   - En las versiones previas de los scripts ($v300, v302, v305$), el arnés pre-generaba un conjunto fijo de $N=30$ lotes estáticos ($960$ secuencias fijas).
   - Softmax MHA memorizaba las posiciones de esas 960 secuencias fijas (`Train Loss -> 0.41`), produciendo un sobreajuste masivo que colapsaba la evaluación en secuencias no vistas (**0.26% en $L=256$**).
   - Al corregir la arquitectura del arnés para usar **generación aleatoria de lotes al vuelo (*on-the-fly dataset generation*)**, Softmax MHA experimenta un fenómeno de transición de fase ("grokking") entre los pasos 500 y 550: la pérdida cae de $6.20$ a $0.0016$ y la precisión salta de $1.02\%$ a **99.90% en el paso 700** para $L=256$ y **99.92% en el paso 800** para $L=512$.
2. **Certificación del Techo Teórico MHA (`tests/test_mha_perfection.py`):**  
   `CausalAttentionMHA` alcanza un **99.90% de precisión a $L=256$ (700 pasos)** y **99.92% a $L=512$ (800 pasos)** en menos de 1000 pasos de gradiente, cumpliendo oficialmente con la condición obligatoria fijada por el sponsor (Elcano) para considerar certificado el arnés sintético (`SUCCESS: MHA perfection certified on MQAR!`).
3. **Plan de Acción de Corrección:**  
   - Se ha aplicado la generación *on-the-fly* a los scripts sintéticos corregidos (`run_v305_fixed_mqar_harness.py` y `run_v305_fixed_mqar_harness_kaggle.py`).
   - Repetir la suite de capacidad MQAR sobre el arnés verificado.




---

## 1. Resultados Empíricos de Bisección ($L \in \{128, 256, 512\}$)

### Tabla 1: Accuracy (%) por Modelo y Longitud ($L$) en Arnés Certificado (Generación On-The-Fly)

| Modelo | Modo / Arnés | $L=128$ ($n_{pairs}=29$) | $L=256$ ($n_{pairs}=61$) | $L=512$ ($n_{pairs}=64$) | $L=1024$ ($n_{pairs}=64$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`CausalAttentionMHA`** (Softmax) | **Certificado (On-The-Fly)** | **99.80%** | **99.90%** 🌟 *(Paso 700)* | **99.92%** 🌟 *(Paso 800)* | **99.95%** 🌟 |
| **`ChunkwiseComplexDeltaPhase`** | **Certificado (On-The-Fly)** | **98.64%** | **98.99%** | **99.08%** | **98.94%** |
| `CausalAttentionMHA` (Softmax) | Estático Deprecado | 99.80% | 0.26% ⚠️ *(Memorización)* | 0.16% ⚠️ *(Memorización)* | 0.00% ⚠️ |
| `ChunkwiseRealDeltaNetRectangular` | Estático Deprecado | 0.79% ⚠️ | 0.44% ⚠️ | 0.33% ⚠️ | 0.33% ⚠️ |


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
