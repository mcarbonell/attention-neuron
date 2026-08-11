# Hallazgos Experimento v315: Resistencia a la Cuantización Post-Entrenamiento a 4 Bits (Fase 8)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

* **Conclusión Previa Modificada:** Se hipotetizó que la cuantización de fase compleja en el círculo unidad $S^1$ sería 100% resistente y superior a la cuantización a 4 bits min-max en el dominio real $\mathbb{R}$.
* **Resultado del Experimento v315 [ANCLA-NEGATIVO]:** 
  1. **Mayor Degradación en Fase Uniforme:** La cuantización de fase uniforme en 16 bins de 22.5° ($\Delta \theta = 22.5^\circ$) en `complex_phase_lora` produjo una degradación de **+0.0274 nats (+0.79%)**, mientras que los modelos reales (`real_molora`, `static_lora`, `standard_dense`) sufrieron una degradación insignificante de **+0.0003 a +0.0005 nats (+0.01%)**.
  2. **Explicación Matemática:** Discretizar ángulos uniformemente con pasos de 22.5° genera variaciones vectoriales $|e^{i 22.5^\circ} - 1| \approx 0.39$ en el plano complejo. En el dominio real $\mathbb{R}$, al no haber outliers extremos tras el entrenamiento de AdamW, el rango min-max de 16 niveles obtuvo un paso de cuantización ultra-fino ($\Delta w \approx 0.066$), perturbando la salida de forma casi nula (+0.01%).

---

## 1. Tabla de Resultados Empíricos

* **Configuración:** $N=2000$ secuencias estructuradas, $L=64$, $d_{model}=128$, 10 épocas, AdamW ($lr=1e-3$). Evaluado en CPU (8 hilos).

| Modelo | Parámetros | Loss FP32 | Loss 4-Bit | $\Delta$ Loss (Degradación) | Etiqueta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`static_lora`** 🌟 | **82,752** | **3.4720** | **3.4720** | **-0.0000 (-0.00%)** | [ANCLA] |
| **`standard_dense`** | 49,984 | 3.4704 | 3.4707 | +0.0003 (+0.01%) | [ANCLA] |
| **`real_molora`** | 83,776 | 3.4715 | 3.4720 | +0.0005 (+0.01%) | [ANCLA] |
| **`complex_phase_lora`** (v315) | 83,776 | 3.4712 | 3.4986 | +0.0274 (+0.79%) | [ANCLA-NEGATIVO] |

*Nota: El marcador 🌟 asigna la menor degradación por cuantización a `static_lora` (-0.0000 nats).*

---

## 2. Análisis del Desempeño y Vía de Solución

1. **Sensibilidad de Fase vs Magnitud:**
   Las funciones trigonométricas $\sin(\theta)$ y $\cos(\theta)$ son altamente sensibles a perturbaciones angulares discretas de $22.5^\circ$.
2. **Recomendación para Cuantización de Fase en 4 Bits:**
   En lugar de una rejilla de fase uniforme rígida de 16 bins, se requiere **cuantización no uniforme adaptativa (k-means clustering de ángulos)** o entrenamiento consciente de la cuantización (*Quantization-Aware Training - QAT*) durante el fine-tuning.

---

## Auditoría posterior y amenazas a la validez (2026-08-10)

Resultado negativo útil, pero evaluado sobre la misma tarea sintética y sin validación retenida. No permite predecir cuantización en un LM; sí invalida la anterior afirmación de inmunidad automática de la fase a 4 bits. Cualquier recuperación mediante QAT o cuantización no uniforme debe demostrarse contra un baseline real bajo el mismo protocolo. Véase la [auditoría transversal v300–v329](findings_v300_v329_audit.md).
