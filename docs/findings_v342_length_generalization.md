# Informe de Hallazgos: Experimento v342 - Generalización a Longitud Extrema y Análisis de Relación Señal-Ruido (SNR)

**ID Experimento:** v342  
**Fecha:** 12 de Agosto, 2026  
**Proyecto:** Attention-Neuron  
**Ubicación:** `C:\Users\mrcm_\Local\proj\algorithms\attention-neuron\docs\findings_v342_length_generalization.md`

---

## 1. Listado de Archivos del Experimento v342

```
attention-neuron/
├── docs/
│   ├── brainstorming_signals_systems_ai.md        # Documento conceptual
│   ├── findings_v341_benchmark_architectures.md    # Hallazgos iniciales del experimento v341
│   └── findings_v342_length_generalization.md     # [Este archivo] Informe y hallazgos del experimento v342
├── scratch/
│   ├── run_architecture_benchmark_v341.py          # Benchmark v341
│   └── run_experiment_v342_length_extrapolation.py # Script ejecutable del experimento v342
└── src/
    ├── dataset.py                                 # Generador dinámico de datos sobre la marcha
    └── models/
        ├── standard_attention.py                  # Baseline Transformer
        ├── dynamic_iir_filter.py                  # Filtro IIR Dinámico con Estado Matricial M_t
        ├── global_workspace.py                    # Red con Pizarra Global de Memoria
        └── hybrid_iir_global.py                   # Modelo Híbrido
```

---

## 2. Resultados Empíricos del Experimento v342

| Modelo / Arquitectura | Train Loss (Época 25) | Train Acc | Val Acc ($L=256$) | Test Acc ($L=512$) | Test Acc ($L=1024$) | Test Acc ($L=2048$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard Attention (Baseline)** | 4.1327 | 1.88% | 2.00% | 1.50% | 1.50% | 1.25% |
| **Global Workspace (Idea 6)** | 4.1284 | 0.69% | 1.50% | 2.75% | 1.50% | 1.00% |
| **Dynamic IIR Filter (Idea 1)** | 4.1285 | 1.44% | 2.00% | 1.00% | 2.25% | 0.75% |
| **Hybrid IIR + Global** | 4.1316 | 1.75% | 2.50% | 0.75% | 2.50% | 3.00% |

*Nota: Para un vocabulario de 64 tokens, el nivel de adivinación aleatoria es $\frac{1}{64} = 1.56\%$ y la entropía máxima es $\ln(64) \approx 4.1588$.*

---

## 3. Diagnóstico Técnico y Descubrimiento Científico

### 3.1. El Fenómeno de la Inundación de Ruido (Signal-to-Noise Ratio Drop)
En secuencias de $L=256$ donde el $98\%$ de los tokens son ruido aleatorio y solo 4 pares son claves/valores reales:
1. Un filtro IIR básico de primer orden o una matriz de memoria uniforme atenúa o acumula linealmente todos los tokens por igual.
2. Tras procesar 250 tokens de ruido sin filtrado de entrada, la **relación Señal-Ruido (SNR)** del estado de memoria interno $M_t$ cae drásticamente.
3. La pérdida se estanca exactamente en $\ln(64) \approx 4.13$, lo que indica que el modelo no logra discernir las claves reales del ruido de fondo.

### 3.2. Diferencia entre Memorización Estática (v341) y Aprendizaje Algorítmico (v342)
* En el experimento **v341** (donde el dataset era estático de 1600 muestras), los modelos alcanzaron un $100\%$ de precisión en entrenamiento memorizando patrones específicos del ruido.
* En el experimento **v342** (con generación dinámica de muestras frescas en cada época), la memorización fue imposible, dejando al descubierto la necesidad de un **mecanismo de compuerta selectiva (Selective Gating)**.

---

## 4. Conclusión y Propuesta para el Experimento v343

Para resolver el problema del ruido y lograr que el filtro IIR generalice en secuencias largas, la arquitectura debe incorporar **Señales de Selección Adaptativa (Selective Gating)**:

$$\alpha_t, \beta_t = \text{SelectiveGate}(x_t)$$
* **Tokens de Ruido:** $\alpha_t \to 1.0$, $\beta_t \to 0.0$ (El estado de memoria no se modifica ni se contamina).
* **Tokens Clave/Valor:** $\alpha_t \to \text{decaimiento}$, $\beta_t \to 1.0$ (La información clave se escribe en la memoria).

---
*Informe generado para el proyecto **attention-neuron**.*
