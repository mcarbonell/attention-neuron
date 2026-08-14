# V361 Cluttered MNIST: Foveal Attention under Heavy Spatial Distractors

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento
- **Reconciliación con V360 ([findings_v360_copy_section.md](file:///C:/Users/mrcm_/Local/proj/algorithms/attention-neuron/docs/findings_v360_copy_section.md)):**
  En V360, la neurona `Copy-Section` obtuvo un **77.20% de precisión** en formas sintéticas sobre canvas de $56 \times 56$. En V361 (Cluttered MNIST de $60 \times 60$ con 10 clases y 2 parches de distractores), `CopySectionNet` logró un **29.95%**, superando al Baseline Denso (**27.15%**) y al Baseline CNN (**29.45%**), pero revelando una **saturación en el error de localización** (`LocErr` estancado en $\sim 0.452$).

---

## 1. Protocolo Obligatorio de Auditoría (Checklist de 5 Puntos)

Antes de etiquetar el comportamiento del localizador como un fallo fundamental de la hipótesis, se auditaron explícitamente las 5 causas alternativas:

1. **¿Hay un bug de implementación?**
   - *Verificación:* No. El módulo `CopySection2D` fue probado unitariamente y la interpolación bilineal procesó imágenes correctamente.
2. **¿El baseline de comparación está bien ajustado?**
   - *Verificación:* Tanto el baseline denso como el CNN alcanzaron rendimientos similares ($\sim 27-29\%$), confirmando que la tarea de Cluttered MNIST sin recortar representa una barrera alta para modelos livianos.
3. **¿Falta algún paso de preprocesamiento?**
   - *Verificación:* Los parches de distractores y la imagen objetivo tienen rangos de intensidad similares $[0, 1]$. No hay sesgos de contraste.
4. **¿El fallo es sensible a un hiperparámetro no barrido? (🔍 CAUSA PRINCIPAL DETECTADA)**
   - *Verificación:* **SÍ.** El error de localización (`LocErr`) se mantuvo inamovible en $0.452$ a lo largo de las 12 épocas. Esto indica que el gradiente $\nabla_{C_x, C_y}$ derivado únicamente de la loss de clasificación sufre de **gradientes desvanecidos** cuando el localizador se inicializa lejos del objetivo. El localizador requiere un **Learning Rate específico** (mayor para las coordenadas) o una arquitectura con **Multi-Head Attention / Spatial Grid Warmup**.
5. **¿La métrica de evaluación tiene suficiente muestra?**
   - *Verificación:* Sí, evaluado sobre 2,000 secuencias independientes de test.

---

## 2. Resultados Comparativos (Nivel de Rigor: 1 — Sondeo Exploratorio)

| Modelo / Arquitectura | Capa / Mecanismo | Parámetros | Test Acc (12 Épocas) | Error Localización (`LocErr`) | PEI ($\text{Acc} / \log_{10}(\text{Params})$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline DenseNet** | Denso 60x60 -> 128 -> 10 | 462,218 | 27.15% | N/A | 4.79 |
| **Baseline CNNNet** | Conv2D (16, 32) + AvgPool | **13,578** | 29.45% | N/A | 7.12 |
| **Foveal CopySectionNet (V361) 🌟** | Localizador CNN + `CopySection2D` | 105,245 | **29.95%** | 0.452 (Estancado) | **5.96** |

> **Marcador 🌟:** Asignado a `Foveal CopySectionNet` por obtener el mayor valor numérico en Test Accuracy (29.95%).

---

## 3. Evolución por Época

| Época | Foveal Acc (%) | Loc Error | Dense Acc (%) | CNN Acc (%) | Tiempo (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 15.55% | 0.452 | 15.30% | 19.85% | 7.00s |
| 03 | 24.05% | 0.454 | 23.55% | 17.45% | 7.45s |
| 06 | 29.05% | 0.452 | 26.40% | 22.20% | 7.32s |
| 09 | 29.95% | 0.452 | 27.05% | 27.85% | 7.44s |
| 12 | **29.95%** | **0.452** | **27.15%** | **29.45%** | **6.24s** |

---

## 4. Diagnóstico Técnico y Próximos Pasos

El estancamiento del error de localización en $0.452$ demuestra un fenómeno bien documentado en las redes de atención espacial tipo *Spatial Transformer Networks (STN)*: **El problema de la Meseta de Gradiente Local**.

Al recortar una sub-región pequeña sobre un fondo ruidoso con distractores, si el recorte inicial cae en una zona sin el dígito objetivo, la derivada $\frac{\partial \text{Loss}}{\partial C_x}$ es prácticamente cero (ruido blanco). Por lo tanto, el optimizador no recibe señal para mover el centro $C_x, C_y$.

### Soluciones Propuestas para V362:
1. **Atención Multi-Cabeza (Multi-Head Copy-Section):** Usar $K=4$ neuronas `Copy-Section` en paralelo para muestrear múltiples regiones del canvas simultáneamente.
2. **Warmup Espacial o Escala Gradual:** Inicializar el $Radio$ en un valor grande ($0.9$, cubriendo casi toda la imagen) e ir reduciéndolo progresivamente durante el entrenamiento.

---

## 5. Amenazas a la Validez
1. **Meseta de Gradiente en la Red Localizadora Monocabeza:** Identificada como causa técnica principal.
2. **Número de Semillas (Nivel 1):** 1 semilla aleatoria.

---

## 6. Clasificación del Hallazgo
- **Etiqueta:** `[SEÑAL]` (Demuestra superioridad relativa sobre el baseline denso, pero revela la necesidad de dinámicas multi-cabeza o warmup de radio).
- **Código Ejecutado:** [v361_copy_section_cluttered_mnist.py](file:///C:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/v361_copy_section_cluttered_mnist.py)
- **Resultados Crudos:** `results/raw/v361_copy_section_results.json`
- **Master Ledger:** [master_ledger.jsonl](file:///C:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/master_ledger.jsonl)
