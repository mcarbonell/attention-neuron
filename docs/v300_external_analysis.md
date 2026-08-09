# Análisis V300 — Resultados Parciales

> **Nivel de rigor:** 1 (Sondeo Exploratorio). n=1, grid de LR estrecho (2e-3, 4e-3, 8e-3). Todos los hallazgos se etiquetan [SEÑAL] o [RUIDO-SOSPECHA].

## 1. Baseline primero: CausalAttentionMHA (O(N²) ceiling)

| d_k | 32 pairs | 64 pairs | 128 pairs | 256 pairs |
|-----|----------|----------|-----------|-----------|
| 32  | 99.97%   | 100.00%  | 100.00%   | 100.00%   |
| 64  | 99.92%   | 99.98%   | --        | --        |

Softmax attention resuelve MQAR-256 de forma esencialmente perfecta en ambas escalas. **Este es el techo.** Comportamiento esperado y coherente con literatura.

---

## 2. Tabla completa de candidatos lineales

### d_k = 32 (d_model = 64, 2048 floats/head complex, 2025 floats/head real-sq, 2048 floats/head real-rect)

| Modelo | 32 pairs | 64 pairs | 128 pairs | 256 pairs |
|--------|----------|----------|-----------|-----------|
| ComplexDeltaPhase | 99.66% | 99.32% | 95.61% | 72.29% |
| RealDeltaNet Square | 94.82% | 86.63% | 71.56% | 0.86% |
| RealDeltaNet Rectangular | **3.93%** ⚠️ | 88.93% | 62.62% | 3.20% |

### d_k = 64 (d_model = 128, 8192 floats/head complex, 8100 floats/head real-sq, 8192 floats/head real-rect)

| Modelo | 32 pairs | 64 pairs | 128 pairs | 256 pairs |
|--------|----------|----------|-----------|-----------|
| ComplexDeltaPhase | 99.96% | 100.00% | -- | -- |
| RealDeltaNet Square | 92.64% | 97.32% | -- | -- |
| RealDeltaNet Rectangular | 95.06% | 97.11% | -- | -- |

---

## 3. Observaciones

### 3.1 Complex domina consistentemente los lineales a d_k=32 [SEÑAL]

La ventaja de ComplexDeltaPhase sobre AMBOS baselines reales es grande y monótona en el régimen `d_k=32`:

| Pairs | Complex | Mejor Real | Δ (pp) |
|-------|---------|-----------|--------|
| 32    | 99.66%  | 94.82% (Sq) | +4.84 |
| 64    | 99.32%  | 88.93% (Rect) | +10.39 |
| 128   | 95.61%  | 71.56% (Sq) | +24.05 |
| 256   | 72.29%  | 3.20% (Rect) | +69.09 |

La diferencia se **amplifica con la carga**: a 256 pairs, Complex retiene 72.29% mientras ambos reales colapsan (<4%). Esto es consistente con la hipótesis de que la fase compleja proporciona mejor condicionamiento de la memoria bajo alta carga.

> Sin embargo, con n=1, las diferencias absolutas no son confirmables. **Se requiere Nivel 2 (≥5 semillas) en el punto 128 pairs para promover a [ANCLA].**

### 3.2 A d_k=64, la brecha se estrecha [SEÑAL]

Con más capacidad (4x más floats de estado), los tres lineales se acercan:

| Pairs | Complex | Rect | Sq | Max Δ (pp) |
|-------|---------|------|-----|-----------|
| 32    | 99.96%  | 95.06% | 92.64% | +7.32 |
| 64    | 100.00% | 97.11% | 97.32% | +2.68 |

Esto sugiere que la ventaja de la fase compleja es más pronunciada en **régimen de capacidad limitada** (d_k pequeño, alta carga). A d_k=64 con pocas pairs, todos los modelos tienen capacidad de sobra.

> **Faltan los puntos críticos** (128 y 256 pairs a d_k=64) que mostrarían si la brecha se reabre bajo carga.

### 3.3 ⚠️ Anomalía: Rectangular 3.93% a d_k=32, pairs=32

Rectangular colapsa a random en la configuración **más fácil** del sweep, pero funciona a 88.93% en la configuración siguiente (64 pairs). Esto es una **curva no monótona** que necesita explicación.

**Causas probables (ordenadas por plausibilidad):**
1. **Fallo de LR:** El grid [2e-3, 4e-3, 8e-3] puede no cubrir el rango óptimo de Rectangular a d_k=32 pequeño. Con d_model=64 y d_key=64 (Rectangular usa 2×d_k como key dim), las proyecciones son cuadradas → la dinámica de gradientes puede ser diferente y necesitar un LR más bajo.
2. **Bug de inicialización específico a esa combinación de dimensiones** (menos probable, ya que d_k=64 funciona bien con 32 pairs).
3. **Inestabilidad numérica con la normalización L2** cuando d_key = d_model (ambos 64 en este caso).

> **Etiqueta: [RUIDO-SOSPECHA].** Este punto no debe usarse para concluir nada sobre Rectangular. Requiere reejecutar con LR más amplio (añadir 1e-3 al grid) y al menos 3 semillas.

### 3.4 La pregunta decisiva: ¿Es la fase o es la forma?

Con los datos disponibles a d_k=32, **Complex supera a Rectangular de forma contundente** en todas las configuraciones donde Rectangular no colapsa:

| Pairs | Complex | Rectangular | Δ (pp) |
|-------|---------|-------------|--------|
| 64    | 99.32%  | 88.93%      | +10.39 |
| 128   | 95.61%  | 62.62%      | +32.99 |

Ambos tienen exactamente 2048 floats/head de estado y la misma forma rectangular de M (d_k × 2×d_k). La única diferencia es la aritmética compleja. Esto apunta a que **la geometría de fase aporta algo que la forma rectangular real no reproduce**.

> **Pero:** n=1 + anomalía en pairs=32 + grid estrecho de LR. La señal es fuerte pero no confirmable sin Nivel 2.

---

## 4. Amenazas a la Validez

1. **n=1.** Todas las diferencias podrían ser artefactos de la semilla 42. Este es el riesgo dominante.
2. **Grid de LR estrecho (3 valores).** Los modelos reales podrían beneficiarse de un LR que no está en el grid. La anomalía de Rectangular a pairs=32 apunta en esta dirección.
3. **d_k=64 incompleto.** Los puntos 128 y 256 pairs a d_k=64 son los que resolverían la pregunta "¿la ventaja persiste con más capacidad?". Sin ellos, la conclusión se apoya solo en d_k=32.

---

## 5. Recomendación de siguiente paso

Dado que los créditos de GPU son escasos, la asignación óptima para cerrar la pregunta binaria es:

| Configuración | Semillas | LR grid | Coste (runs) |
|---|---|---|---|
| d_k=32, **128 pairs** | 5 | [1e-3, 2e-3, 4e-3] | 5 × 3 × 3 modelos = **45** |

Un solo punto del acantilado, con semillas y un LR bajo adicional. Si Complex mantiene Δ > 2×SE sobre Rectangular en esa celda, es [ANCLA]. Si no, [RUIDO-SOSPECHA] y hay que buscar la explicación en el LR o en la implementación.
