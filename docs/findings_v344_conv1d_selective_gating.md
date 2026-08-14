# Informe de Hallazgos: Experimento v344 - Convolución Causal 1D + Selective IIR

**ID Experimento:** v344  
**Fecha:** 13 de Agosto, 2026  
**Proyecto:** Attention-Neuron  
**Ubicación:** `C:\Users\mrcm_\Local\proj\algorithms\attention-neuron\docs\findings_v344_conv1d_selective_gating.md`

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

Este experimento reconcilia y corrige la conclusión inicial extraída en el experimento **v341**:
* **En v341:** Se reportó un $63.25\%$ de precisión para la arquitectura IIR. Sin embargo, debido a que la evaluación se realizó sobre un conjunto de datos estático pre-generado, el modelo sufrió de **memorización algorítmica estática (Grokking/Overfitting)**.
* **En v342, v343 y v344:** Al evaluar en conjuntos de datos dinámicos generados sobre la marcha (muestras frescas por batch), se demostró que sobre un canal saturado por un $98\%$ de ruido sin marcadores de posición o estructurales explícitos, ni Transformer ni IIR ni Selective Conv1D logran extraer el par clave-valor sin mecanismo de atención explicita o encodings posicionales de grano fino.
* **Refutación explícita:** Queda refutada la afirmación de que el filtro IIR básico por sí solo sin mecanismo de marcadores posicionales resuelve *Associative Recall* sobre ruido denso no estructurado.

---

## 1. Listado de Archivos del Repositorio (`attention-neuron/`)

```
attention-neuron/
├── docs/
│   ├── brainstorming_signals_systems_ai.md        # Documento conceptual
│   ├── findings_v341_benchmark_architectures.md    # Hallazgos del experimento v341
│   ├── findings_v342_length_generalization.md     # Diagnóstico de memorización vs datos dinámicos
│   ├── findings_v343_selective_gating.md          # Diagnóstico de la compuerta estática vs Conv1D
│   └── findings_v344_conv1d_selective_gating.md   # [Este archivo] Informe y trazabilidad de v344
├── results/
│   ├── raw/
│   │   └── v344_results.json                      # Resultados JSON crudos del experimento v344
│   └── master_ledger.jsonl                        # Registro maestro de todos los experimentos
├── scratch/
│   ├── run_architecture_benchmark_v341.py          # Benchmark v341
│   ├── run_experiment_v342_length_extrapolation.py # Experimento v342
│   ├── run_experiment_v343_selective_iir.py        # Experimento v343
│   └── prototype_v344_conv1d_selective_iir.py      # Script ejecutable v344 (Con trazabilidad completa)
└── src/
    ├── dataset.py                                 # Generador dinámico de muestras sobre la marcha
    └── models/
        ├── standard_attention.py                  # Baseline Transformer
        ├── dynamic_iir_filter.py                  # Filtro IIR Dinámico
        ├── global_workspace.py                    # Red con Pizarra Global
        ├── hybrid_iir_global.py                   # Modelo Híbrido
        ├── selective_iir_filter.py                # Capa Selective IIR (v343)
        └── selective_conv1d_iir.py                # Capa Conv1D + Selective IIR (v344)
```

---

## 2. Resultados Empíricos del Experimento v344

### 2.1. Métricas de Entrenamiento y Eficiencia

| Modelo / Arquitectura | Parámetros Totales | Loss Final (Época 20) | Wall Clock Time (s) | Eval Time (s) | Overhead Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Selective-Conv1D IIR (v344 Candidato)** | 93,062 | 4.1562 | 1099.53s | 1097.20s | 2.33s |
| **Dynamic IIR Filter (Baseline v341)** 🌟 | 91,266 | **4.1333** 🌟 | 542.98s | 541.90s | 1.08s |
| **Standard Attention (Baseline)** | 805,696 | 4.1381 | **464.24s** 🌟 | 463.10s | 1.14s |

*(Nota: El símbolo 🌟 se asigna de forma estrictamente numérica al mejor valor de cada columna de acuerdo a la norma de GEMINI.md).*

### 2.2. Precisión Zero-Shot por Longitud de Secuencia ($L$)

| Modelo / Arquitectura | $L=256$ (Train) | $L=512$ | $L=1024$ | $L=2048$ 🌟 |
| :--- | :---: | :---: | :---: | :---: |
| **Selective-Conv1D IIR (v344 Candidato)** | 1.50% | 1.00% | 0.75% | **3.00%** 🌟 |
| **Dynamic IIR Filter (Baseline v341)** | **2.00%** 🌟 | 0.00% ⚠️ | **2.50%** 🌟 | 1.25% |
| **Standard Attention (Baseline)** | 1.00% | **1.50%** 🌟 | 1.50% | 1.25% |

---

## 3. Diagnóstico según el Protocolo Obligatorio Antes de Declarar Resultado Negativo

Para clasificar este hallazgo y evitar cierres prematuros, se evalúan las 5 causas del protocolo:

1. **¿Hay un bug de implementación?**  
   *Verificado:* Se ejecutó un forward pass unitario y el gradiente fluye correctamente a través de la convolución 1D y la compuerta `g_t`. La pérdida descendió de `4.4139` a `4.1562`.
2. **¿El baseline de comparación está bien ajustado?**  
   *Verificado:* El baseline `Standard Attention` también se estancó en `4.1381` (entropía de azar uniforme $\ln(64) \approx 4.1588$), confirmando que el desafío radica en el *Signal-to-Noise Ratio* del dataset.
3. **¿Falta algún paso de preprocesamiento?**  
   *Identificado:* El dataset actual coloca 250 tokens de ruido aleatorio puro sin marcadores de posición (*Positional Encodings* o *RoPE*) ni tokens indicadores de inicio de clave (`KEY_START`).
4. **¿El fallo es sensible a un hiperparámetro no barrido?**  
   *Pendiente:* Se probó con `d_state=16` y `kernel_size=4`. Aumentar la dimensión de estado a `d_state=64` o añadir *Positional Embeddings* podría modificar la convergencia.
5. **¿La métrica de evaluación tiene suficiente muestra?**  
   *Verificado:* Se evaluaron 400 secuencias independientes por cada longitud de test.

**Etiqueta del hallazgo:** `[CIERRE-PREMATURO-SOSPECHA]` (Aún no promovido a `[ANCLA-NEGATIVO]` hasta evaluar la adición de marcadores posicionales o estructura de atención en v345).

---

## 4. Amenazas a la Validez

1. **Objeción 1 (Estructura de la Tarea Sintética):** La tarea actual asigna la misma distribución de tokens ($2 \dots 63$) al ruido y a las claves/valores, sin ningún token identificador de tipo. Esto genera una ambigüedad sintáctica casi insuperable para sistemas no basados en atención global.  
   *Experimento para dirimir:* Agregar tokens de tipo explícitos (`[KEY_TOKEN]`, `[VAL_TOKEN]`) o *Rotary Positional Embeddings (RoPE)*.
2. **Objeción 2 (Dimensión de Estado Limitada):** `d_state=16` puede ser insuficiente para representar la matriz asociativa de claves en presencia de ruido denso.  
   *Experimento para dirimir:* Barrer `d_state \in [32, 64, 128]`.
3. **Objeción 3 (Tamaño del Dataset de Entrenamiento):** 1,600 secuencias dinámicas pueden ser insuficientes para que el optimizador encuentre el punto de bifurcación de la compuerta selectiva.  
   *Experimento para dirimir:* Incrementar a 5,000 secuencias o aumentar las épocas a 50.

---
*Informe generado para el proyecto **attention-neuron** bajo la normativa de `GEMINI.md`.*
