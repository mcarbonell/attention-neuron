# Findings v300 — Capacity Scaling & Holographic Phase Advantage (Provisional)

> ⚠️ **DOCUMENTO PROVISIONAL:** Resultados preliminares basados en ejecuciones parciales ($n=1$, `seed=42`). Clasificado como **Nivel 1 (Sondeo Exploratorio)**. No citar como evidencia definitiva sin pasar los criterios de Nivel 2.

---

## 1. Contexto y Objetivos del Experimento v300

El experimento `v300` evalúa la hipótesis de la **Atención de Fase Compleja Causal (DeltaPhase)** frente a variantes de DeltaNet real en el benchmark MQAR (*Multi-Query Associative Recall*).

El objetivo primario es dirimir si la aritmética compleja en el círculo unitario $\mathbb{C}^{d_k}$ aporta una ventaja geométrica real o si sus beneficios pueden igualarse mediante un control real con el mismo presupuesto de floats libres por cabeza de atención (**iso-floats**).

### Controles probados:
1. **`ChunkwiseComplexDeltaPhase`:** Claves complejas en el círculo unitario $S^1 \subset \mathbb{C}^{d_k}$. Estado $M \in \mathbb{C}^{d_k \times d_k}$ ($2 d_k^2$ floats/head).
2. **`ChunkwiseRealDeltaNetSquare`:** Claves reales L2-normalizadas en $\mathbb{R}^{d_{k,real}}$. Estado $M \in \mathbb{R}^{d_{k,real} \times d_{k,real}}$.
3. **`ChunkwiseRealDeltaNetRectangular` (Control Iso-Floats Principal):** Claves reales L2-normalizadas con $d_{key} = 2 d_k$ y $d_{val} = d_k$. Estado $M \in \mathbb{R}^{d_k \times 2d_k}$ ($2 d_k^2$ floats/head exactos).
4. **`CausalAttentionMHA`:** Atención Causal Softmax $O(N^2)$ como techo teórico de capacidad.

---

## 2. Resultados Provisionales ($d_k=32$ y $d_k=64$)

### Tabla 1: Best Accuracy Final (%) en MQAR según Carga de Pares ($d_k=32$)

| Modelo | 32 Pares ($L=256$) | 64 Pares ($L=512$) | 128 Pares ($L=1024$) | 256 Pares ($L=2048$) |
| :--- | :---: | :---: | :---: | :---: |
| **CausalAttentionMHA** (Techo $O(N^2)$) | 99.97% | 100.00% | 100.00% | 100.00% |
| **ChunkwiseComplexDeltaPhase** | **99.66%** | **99.32%** | **95.61%** 🌟 | **72.29%** 🌟 |
| **RealDeltaNet Square** | 94.82% | 86.63% | 71.56% | 0.86% |
| **RealDeltaNet Rectangular** (Iso-Floats) | 3.93% ⚠️ | 88.93% | 62.62% | 3.20% |

### 3.3 Eficiencia de Muestreo: Velocidad de Convergencia vs Truncamiento de Épocas [SEÑAL]
Una inspección detallada de la curva de loss en $d_k=64$ (256 pares, $L=2048$) revela una distinción crítica:
- **`ChunkwiseComplexDeltaPhase`:** Logra convergencia rápida hacia Loss $< 0.20$ en la **Época 10** (Loss $0.1705 \to 0.0017$ en Época 13), alcanzando **99.94%**.
- **`ChunkwiseRealDeltaNetSquare`:** Muestra una pendiente de descenso tardía pero acelerada en las últimas épocas:  
  `Época 15: Loss 61.97` $\to$ `Época 17: Loss 19.31` $\to$ `Época 20: Loss 5.7074` (Acc: 83.87%).

**Conclusión:** El valor de $83.87\%$ del control real a 20 épocas **no refleja una incapacidad estructural**, sino un **cierre prematuro por truncamiento de épocas**. La verdadera ventaja de `ComplexDeltaPhase` en esta prueba es de **Eficiencia de Muestreo y Velocidad de Convergencia** (aprende la asociación en la mitad de épocas).



## 3. Hallazgos Principales

### 3.1 Ventaja Compleja Amplificada bajo Saturación [SEÑAL]
En la escala $d_k=32$, `ComplexDeltaPhase` mantiene un **72.29% de precisión** con 256 pares ($L=2048$), mientras que ambos baselines reales colapsan por completo al nivel del azar ($<4.0\%$).

A 128 pares, `ComplexDeltaPhase` alcanza **95.61%**, superando por **+24.05 pp** a `RealSquare` (71.56%) y por **+32.99 pp** al control iso-floats `RealRectangular` (62.62%).

### 3.2 Anomalía de Colapso Puntual en Rectangular [RUIDO-SOSPECHA]
`RealDeltaNetRectangular` obtiene un **3.93%** en la configuración más fácil (32 pares), pero funciona a **88.93%** en la siguiente (64 pares). Esta curva no monótona se clasifica provisionalmente como un fallo de optimización/grid de LR estrecho [2e-3, 4e-3, 8e-3].

---

## 4. Amenazas a la Validez

1. **Semilla Única ($n=1$):** Todos los puntos provienen de una única corrida (`seed=42`). No hay cálculo de error estándar ($SE$).
2. **Grid de LR Acotado (3 valores):** El hiperparámetro de LR no se barrió de forma continua; el colapso puntual a 32 pares en Rectangular puede ser un artefacto del grid.
3. **Log Incompleto:** La escala $d_k=64$ quedó truncada en los puntos de alta carga (128 y 256 pares) debido a límites de tiempo en la sesión original.

---

## 5. Sugerencia de Siguientes Experimentos

1. **Promoción a Nivel 2 del Acantilado 128-pairs:** Ejecutar 5 semillas fijas ($n=5$) en la celda $d_k=32$, $Pairs=128$ agregando $1e-3$ al grid de LR para verificar el margen con SE.
2. **Completar $d_k=64$ bajo Carga:** Barrer los puntos 128 y 256 pares en $d_k=64$ para verificar si la brecha de fase se mantiene al aumentar la dimensión.
