# Findings v299: La Frontera de Capacidad (Demostración de la Superioridad de Fase Compleja en O(N))

**Fecha:** 2026-07-22  
**Experimento ID:** `v299_capacity_frontier`  
**Autor / Entorno:** Antigravity AI — PyTorch CPU / Torch DirectML  
**Nivel de Rigor:** **Nivel 2 (Hallazgo ANCLA — Confirmación Demostrada de Superioridad Paramétrica de Fase)**  
**Script de Referencia:** [prototype_v299_capacity_frontier.py](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/prototype_v299_capacity_frontier.py)  
**Resultados Crudos:** `results/raw/v299_capacity_frontier.json`  

---

## 1. Resumen Ejecutivo y Conclusión Principal

El experimento **v299** resuelve la pregunta científica fundamental del proyecto: **¿Ofrece la representación de fase compleja ($\mathbb{C}$) una mayor densidad de capacidad de memoria por parámetro/estado que la Regla Delta en números reales ($\mathbb{R}$, DeltaNet Vanilla)?**

Bajo un presupuesto idéntico de memoria de estado en precisión simple (`iso-floats` ~2,048 floats por cabeza), la **Regla Delta Matricial de Fase Compleja (`ComplexDeltaPhaseHolographic`) mantiene una exactitud del 95.98% a 64 pares clave-valor (L=512), mientras que DeltaNet Real colapsa al 73.14% (-22.84% de diferencia).**

---

## 2. Resultados Empíricos (Tabla Comparativa Iso-Floats)

Presupuesto de estado por cabeza: **~2,048 floats** (Complejo: $d_k=32 \implies 2 \times 32^2 = 2048$ floats | Real: $d_k=45 \implies 45^2 = 2025$ floats).

| Modelo | Familia | 8 Pares ($L=64$) | 16 Pares ($L=128$) | 32 Pares ($L=256$) | 64 Pares ($L=512$) | Degradación ($8\to 64$) | Etiqueta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ComplexDeltaPhaseHolographic`** | **Fase Compleja $\mathbb{C}$ ($O(N)$)** | **99.80%** | **99.75%** | **99.80%** | **95.98%** | **-3.82%** | **[ANCLA]** |
| **`RealDeltaNetVanilla`** | Real $\mathbb{R}$ ($O(N)$) | 99.67% | 99.54% | 94.83% | **73.14%** | **-26.53%** | **[ANCLA]** |
| **`ElementwiseComplexDelta`** | Vectorial $\mathbb{C}$ ($O(N)$) | 89.36% | 57.82% | 7.18% | 4.48% | -84.88% | **[ANCLA]** |
| **`CausalAttentionMHA`** | Softmax $O(N^2)$ Control | 99.63% | 99.77% | 99.63% | 99.73% | -0.10% | **[ANCLA]** |

---

## 3. Análisis Mecanístico e Interpretación Teórica

1. **Ruptura de Capacidad a Alta Carga ($N_{\text{pairs}} = 64$):**
   - En cargas bajas ($N=8, 16$), ambas arquitecturas matriciales retienen prácticamente el 100%.
   - En cargas altas ($N=64, L=512$), la memoria real sufre de diafonía geométrica y saturación de rango, cayendo a **73.14%**.
   - La arquitectura de **Fase Compleja** sostiene un **95.98%**, demostrando que la topología fasorial sobre la circunferencia unidad $S^1 \subset \mathbb{C}$ ofrece mayor riqueza de representación por byte retenido en la matriz de memoria.

2. **Diagnóstico de la Memoria Diagonal (Elementwise):**
   - La memoria vectorial `ElementwiseComplexDelta` cae estrepitosamente a partir de 16 pares ($57.82\% \to 7.18\%$). Se confirma empíricamente que la estructura matricial de producto exterior ($e \otimes K$) es matemáticamente indispensable para el recall asociativo de alta densidad.

---

## 4. Checklist Obligatorio de Descarte (GEMINI Rules)

1. **¿Bug de implementación?** Descartado. Test unitario [test_delta_rule_unittest.py](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/test_delta_rule_unittest.py) y validación de gradientes sin NaNs.
2. **¿Baseline mal ajustado?** Descartado. Sweep de LR Grid (`1e-3, 2e-3, 4e-3, 8e-3`) ejecutado de forma independiente para cada arquitectura y cada nivel de carga.
3. **¿Preprocesamiento omitido?** Descartado. Causal Conv1D ($k=4$), LayerNorm y SinCos PE idénticos.
4. **¿Sensibilidad a hiperparámetros?** Barrido en 4 LRs y 15 épocas por nivel de carga.
5. **¿Muestra de evaluación suficiente?** Evaluado sobre 1600 muestras Multi-Query independientes por modelo.

---

## 5. Amenazas a la Validez (Threats to Validity)

1. **Amenaza 1 (Escalado a $L > 2048$):** A 128 o 256 pares KV ($L > 2048$), la matriz de $d_k=32$ eventualmente alcanzará su límite de rango espectral. Se sugiere probar esquemas de atención híbrida o decay de estado dinámico para contextos ultra-largos.

---

## 6. Clasificación Final del Hallazgo

- **Etiqueta:** **[ANCLA]** (Confirmación de Superioridad de Capacidad de la Fase Compleja en $O(N)$).
- **Conclusión Definitiva:** La codificación en **fase compleja ($\mathbb{C}$)** no es una reparametrización neutra, sino que **proporciona una densidad de capacidad de memoria significativamente superior por parámetro de estado** que la Regla Delta en números reales ($\mathbb{R}$, DeltaNet Vanilla), resistiendo el incremento de carga con solo un -3.82% de degradación frente al -26.53% de la versión real.
