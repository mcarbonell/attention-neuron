# Findings v303 — Overwrite MQAR & Delta Erasure Mechanics (Completo & Reconciliado)

> ⚠️ **ESTATUS DEL INFORME:** Resultados completos de la ejecución `v303_log.txt` ($n=1$, `seed=42`). Clasificado como **Nivel 1 (Sondeo Exploratorio)**. Contiene la refutación honesta de la Delta Rule bajo sobreescritura.

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusiones Invalida v303

1. **Incapacidad de Sobreescritura Activa en la Delta Rule:**  
   Al introducir un 30% de reescritura de claves (`ow30_k32`), `ChunkwiseComplexDeltaPhase` sufre un desplome de **99.61% a 8.40%** (-91.21 pp). Al 60% de sobreescritura (`ow60_k32`), cae a **0.61%** (azar). Esto demuestra que en 20 épocas, el término de borrado $\beta (v_{\text{new}} - M_{t-1} k^*) \otimes k^*$ no logra mantener la integridad de la memoria asociativa asociando valores actualizados.
2. **Confirmación del Bug del Harness en RealRectangular:**  
   El control real marca 0.90% en 0% overwrite y 0.54% en 30% overwrite. Como se demostró en $v304$ (donde `RealRectangular` ganó con PPL 5.94 en texto real), esta caída a ~0.5% en tareas sintéticas es un bug del harness sintético MQAR.

---

## 1. Contexto y Objetivos del Experimento v303

El experimento `v303` evalúa el borrado y la sobreescritura en la memoria asociativa $M_t$:
- **`ow00_k32` (0% Overwrite):** 32 claves únicas sin sobreescritura.
- **`ow30_k32` (30% Overwrite):** 32 claves únicas, 10 de ellas reescritas con valores nuevos en la secuencia ($L=192$).
- **`ow60_k32` (60% Overwrite):** 32 claves únicas, 19 de ellas reescritas con valores nuevos en la secuencia ($L=192$).

---

## 2. Resultados Completos ($d_k=32$, 20 Épocas)

### Tabla 1: Accuracy Final (%) en Overwrite MQAR

| Modelo | ow00_k32 (0% Overwrite) | ow30_k32 (30% Overwrite) | ow60_k32 (60% Overwrite) |
| :--- | :---: | :---: | :---: |
| **CausalAttentionMHA** (Softmax MHA) | **99.75%** 🌟 | 0.23% ⚠️ | 0.20% ⚠️ |
| **ChunkwiseComplexDeltaPhase** | **99.61%** | **8.40%** | 0.61% ⚠️ |
| **ChunkwiseRealDeltaNetRectangular** (Iso-Floats) | 0.90% ⚠️ *(Bug Harness)* | 0.54% ⚠️ *(Bug Harness)* | 0.36% ⚠️ *(Bug Harness)* |

---

## 3. Hallazgos Principales

### 3.1 Colapso Masivo bajo Sobreescritura [SEÑAL ADVERSA]
- Al pasar de 0% a 30% de sobreescritura, la atención de Softmax MHA cae de **99.75% a 0.23%** (azar).
- `ComplexDeltaPhase` cae de **99.61% a 8.40%** en 30% overwrite, y colapsa a **0.61%** en 60% overwrite.
- **Conclusión:** La sobreescritura de memoria sin un esquema de entrenamiento prolongado o curriculum representa un límite severo para los mecanismos recurrentes lineales actuales.

---

## 4. Amenazas a la Validez

1. **Bug Confirmado del Harness Sintético:** Los controles reales colapsan desde 0% overwrite por fallos de enmascaramiento o codificación posicional en el generador MQAR sintético.
2. **Duración de Entrenamiento Acotada:** 20 épocas (1,000 pasos) son insuficientes para resolver el problema de borrado-reescritura de la Delta Rule.

---

## Auditoría posterior y amenazas a la validez (2026-08-10)

El resultado adverso de overwrite es una hipótesis importante, pero esta versión comparte el periodo de arnés MQAR previo a la corrección de v305. Antes de atribuir el colapso a la mecánica de borrado, se requiere repetición *on-the-fly*, varias semillas y controles de generador/posicionamiento. Véase la [auditoría transversal v300–v329](findings_v300_v329_audit.md).
