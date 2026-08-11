# Hallazgos Experimento v312: MoLoRA en MQAR Benchmark (Fase 5)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

* **Conclusión Previa Modificada:** En los experimentos `v310` y `v311`, MoLoRA demostró ser superior a la capa Densa y a LoRA estático en tareas de secuencia estructurada. Se hipotetizó que MoLoRA también podría resolver memoria asociativa de largo contexto en MQAR.
* **Resultado del Experimento v312:** 
  1. **Brecha Abrumadora de Atención [ANCLA-NEGATIVO]:** Multi-Head Self-Attention (`mha`) alcanzó un **35.16% Target Accuracy** (y continua subiendo), mientras que todas las variantes lineales sin atención (`fast_molora`, `static_lora`, `dense`) colapsaron cerca del azar (~0% a 1.95% Acc).
  2. **Diagnóstico Técnico:** El router de MoLoRA $g(x_t) = \text{Softmax}(W_{router} x_t)$ opera de forma estrictamente local token-por-token. Sin un mecanismo de producto interno Causal de historia ($Q_t K_s^T$) o de estado dinámico recurrente (como DeltaNet), MoLoRA carece de capacidad de lectura asociativa del pasado.
* **Reconciliación de Dominio:** MoLoRA es una arquitectura para reemplazar capas de **Feed-Forward / Proyección (FFN)**, no para sustituir la mezcla temporal de secuencia (Atención/Fase).

---

## 1. Tabla de Resultados Empíricos

* **Configuración:** MQAR ($L=64$, $N_{pairs}=8$, Vocab=120), 400 pasos de entrenamiento, AdamW ($lr=2e-3$). Evaluado en CPU (8 hilos).

| Modelo | Parámetros | Target Acc (%) | Loss Final | Wall Clock (s) | PEI | Etiqueta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`mha`** (Causal Attention) 🌟 | 163,448 | **35.16%** | **2.7003** | 9.08 | **0.0674** | [ANCLA] |
| **`static_lora`** (r=64) | 96,888 | 1.95% | 4.6434 | **4.73** | 0.0039 | [ANCLA-NEGATIVO] |
| **`fast_molora_K8_r8`** | 98,936 | 1.17% | 4.6424 | 13.16 | 0.0023 | [ANCLA-NEGATIVO] |
| **`dense`** | 64,120 | 1.17% | 4.6433 | 3.17 | 0.0024 | [ANCLA-NEGATIVO] |
| **`fast_molora_K16_r4`** | 100,984 | 0.00% | 4.6813 | 40.24 | 0.0000 | [ANCLA-NEGATIVO] |

*Nota: El marcador 🌟 asigna la mejor precisión a `mha` (35.16%).*

---

## 2. Análisis del Desempeño y Frontera Causal

1. **Incapacidad Causal de los Adaptadores Locales:**
   Las redes compuestas únicamente por capas `Linear` o `MoLoRA` sin mecanismo de contexto no pueden resolver la asociación de pares dispersos a $L=64$.
2. **Definición de Arquitectura Híbrida (Vía para Fase 6 - v313):**
   MoLoRA debe integrarse en los bloques FFN de un Transformer/DeltaNet, donde la atención/fase se encarga de la agregación espacial $L \times L$ y MoLoRA se encarga de la transformación dinámica $d \times d$.

---

## Auditoría posterior y amenazas a la validez (2026-08-10)

La separación conceptual entre mezcla temporal y FFN es una hipótesis razonable, pero las cifras son la accuracy/loss del último batch de entrenamiento, con una semilla y 400 pasos, no evaluación independiente. La conclusión debe limitarse a este protocolo corto de MQAR; requiere evaluación *on-the-fly* retenida y multisemilla. Véase la [auditoría transversal v300–v329](findings_v300_v329_audit.md).
