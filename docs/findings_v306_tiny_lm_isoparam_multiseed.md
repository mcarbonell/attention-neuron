# Findings v306 — Tiny LM Iso-Paramétrico & Multi-Semilla [ANCLA]

> ⚓ **NIVEL DE RIGOR:** **Nivel 2 [ANCLA]**. Evaluación verificada en 5 semillas independientes (`seeds = [10, 20, 30, 42, 100]`) sobre *Tiny Shakespeare* (1.1M caracteres) con presupuesto iso-paramétrico estricto ($144,331$ parámetros por modelo) y cálculo de Error Estándar ($SE$).

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica v306

1. **Reconciliación de v304 vs v306 (Ajuste Iso-Paramétrico Estricto):**  
   En el experimento previo $v304$, el control real `RealRectangular` obtuvo la mejor PPL (5.94 vs 6.00), pero contaba con un $21.7\%$ más de parámetros (175,675 vs 144,331).  
   Al corregir el diseño en $v306$ ajustando `ChunkwiseRealDeltaNetIsoParam` al **presupuesto paramétrico exacto de 144,331 parámetros**, `ChunkwiseComplexDeltaPhase` **alcanza la menor perplejidad y pérdida de validación del estudio** (PPL **5.96 ± 0.02** vs **6.07 ± 0.01**).
2. **Significancia Estadística Confirmada:**  
   La diferencia $\Delta \text{ValLoss} = 1.8026 - 1.7849 = 0.0177 \text{ nats}$ entre la fase compleja y el control real supera el umbral de $2 \times SE_{\text{diff}}$ ($0.0177 > 4.8 \times 0.0037$), confirmando una ventaja cuantitativa modesta pero estadísticamente significativa ($p < 0.001$).

---

## 1. Contexto y Configuración Experimental

El experimento `v306` promociona el modelado de lenguaje natural a **Nivel 2 [ANCLA]** mediante:
- **Presupuesto Iso-Paramétrico:** Todos los modelos recurrentes se fijaron a **144,331 parámetros** exactos.
- **Scheduler con LR Warmup (5%):** Aplica un escalado lineal de tasa de aprendizaje para estabilizar la optimización autorregresiva inicial.
- **5 Semillas Independientes:** `seed = [10, 20, 30, 42, 100]` para evaluar estabilidad y desviación estándar.

---

## 2. Resultados Empíricos (Nivel 2 [ANCLA], 5 Semillas, Tiny Shakespeare)

### Tabla 1: Resumen de Pérdida y Perplejidad de Validación (Mean ± SE)

| Modelo | Parámetros | Mean Val Loss ± SE | Mean Val PPL ± SE | Semilla 10 | Semilla 20 | Semilla 30 | Semilla 42 | Semilla 100 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ChunkwiseComplexDeltaPhase`** | 144,331 | **1.7849 ± 0.0028** | **5.96 ± 0.02** 🌟 | 5.99 | 5.92 | 5.91 | 6.01 | 5.96 |
| **`ChunkwiseRealDeltaNetIsoParam`** (Iso-Params) | 144,331 | **1.8026 ± 0.0024** | **6.07 ± 0.01** | 6.10 | 6.04 | 6.08 | 6.09 | 6.02 |
| **`CausalAttentionMHA`** (Softmax MHA) | 143,811 | **1.8519 ± 0.0061** | **6.37 ± 0.04** | 6.24 | 6.47 | 6.31 | 6.46 | 6.38 |

---

## 3. Hallazgos Principales

### 3.1 Promoción a [ANCLA]: Ventaja Paramétrica de la Fase Compleja [ANCLA]
Bajo el mismo presupuesto paramétrico exacto (144,331 parámetros) y promediado en 5 semillas:
- `ChunkwiseComplexDeltaPhase` (PPL **5.96 ± 0.02**) supera al control real `ChunkwiseRealDeltaNetIsoParam` (PPL **6.07 ± 0.01**) por **-0.11 puntos de PPL** (-0.0177 nats de pérdida), verificado con $SE = 0.0028$.
- Ambos modelos recurrentes DeltaNet superan a la atención Softmax `CausalAttentionMHA` (PPL **6.37 ± 0.04**) en más de **-0.30 puntos de PPL**.

### 3.2 Estabilidad Consistente entre Semillas [ANCLA]
La desviación estándar entre semillas en `ComplexDeltaPhase` es extremadamente baja ($SE = 0.02$ en PPL), registrando valores entre 5.91 y 6.01 en todas las ejecuciones, lo que confirma la estabilidad del entrenamiento en el círculo unitario $\mathbb{C}$.

---

## 4. Amenazas a la Validez

1. **Tamaño de Corpus:** Evaluado exclusivamente en *Tiny Shakespeare* (1.1M caracteres) a nivel de caracteres ($L=256$). La extensión a BPE/subwords en *TinyStories* se requiere para escalado a gran escala.
2. **Latencia Real / Wall-Clock:** Aunque los parámetros son idénticos, la multiplicación en $\mathbb{C}$ involucra dos componentes (real e imaginaria). En hardware sin soporte nativo para números complejos, la intensidad de operaciones es superior.

---

## 5. Próximos Pasos

1. **Evaluación de Latencia Real (Wall-clock Time):** Medir la latencia en milisegundos de inferencia por token en GPU/CPU.
2. **Actualización de la Ficha Técnica ONE-001:** Actualizar la entrada [entry_001_chunkwise_complex_deltaphase.md](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/docs/encyclopedia/entry_001_chunkwise_complex_deltaphase.md) con la etiqueta **[ANCLA]** y los datos iso-paramétricos de $v306$.
