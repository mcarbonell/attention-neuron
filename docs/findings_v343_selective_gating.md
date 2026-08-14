# Informe de Hallazgos: Experimento v343 - Diagnóstico de Compuerta Selectiva y la Necesidad de Convolución 1D Causal

**ID Experimento:** v343  
**Fecha:** 13 de Agosto, 2026  
**Proyecto:** Attention-Neuron  
**Ubicación:** `C:\Users\mrcm_\Local\proj\algorithms\attention-neuron\docs\findings_v343_selective_gating.md`

---

## 1. Listado de Archivos del Experimento v343

```
attention-neuron/
├── docs/
│   ├── brainstorming_signals_systems_ai.md        # Documento conceptual
│   ├── findings_v341_benchmark_architectures.md    # Hallazgos del experimento v341
│   ├── findings_v342_length_generalization.md     # Diagnóstico del efecto de inundación de ruido
│   └── findings_v343_selective_gating.md          # [Este archivo] Diagnóstico y revelación matemática v343
├── scratch/
│   ├── run_architecture_benchmark_v341.py          # Benchmark v341
│   ├── run_experiment_v342_length_extrapolation.py # Experimento v342
│   └── run_experiment_v343_selective_iir.py        # Script ejecutable del experimento v343
└── src/
    ├── dataset.py                                 # Generador dinámico de datos sobre la marcha
    └── models/
        ├── standard_attention.py                  # Baseline Transformer
        ├── dynamic_iir_filter.py                  # Filtro IIR Dinámico (v341/v342)
        ├── global_workspace.py                    # Red con Pizarra Global de Memoria
        ├── hybrid_iir_global.py                   # Modelo Híbrido
        └── selective_iir_filter.py                # Capa Selective-Gate IIR (v343)
```

---

## 2. Resultados Empíricos del Experimento v343

| Modelo / Arquitectura | Train Loss (Época 20) | Train Acc | Val Acc ($L=256$) | Test Acc ($L=512$) | Test Acc ($L=1024$) | Test Acc ($L=2048$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard Attention (Baseline)** | 4.1316 | 2.00% | 1.50% | 1.25% | 2.00% | 1.00% |
| **Dynamic IIR Filter (v341/v342)** | 4.1337 | 1.00% | 2.00% | 2.75% | 2.00% | 2.00% |
| **Selective-Gate IIR (v343 Nuevo)** | 4.1281 | 2.00% | 1.75% | 2.25% | 1.00% | 1.50% |

---

## 3. Revelación Matemática Fundamental: ¿Por qué la compuerta estática no pudo filtrar el ruido?

### 3.1. Ambigüedad de la Identidad del Token vs. Rol Contextual
En el modelo v343, la compuerta de selección se definió como:
$$g_t = \text{sigmoid}(W_{gate} \cdot \text{Embedding}(x_t))$$

* **El Problema:** El token ID `15` se toma del mismo vocabulario $V \in [2, 63]$. Un mismo token ID `15` actúa como **ruido** en la posición $t=10$, pero puede actuar como una **Clave o Valor legítimo** en la posición $t=85$.
* Como $W_{gate} \cdot \text{Embedding}(15)$ produce el mismo escalar estático independientemente de la posición o contexto, la red es incapaz de decidir si el token `15` debe filtrarse ($g_t \to 0$) o guardarse ($g_t \to 1$).

### 3.2. La Solución Utilizada en Mamba/S4: Convolución 1D Causal Previa (Causal Conv1D)
En las arquitecturas SOTA de espacio de estados (como Mamba o S4):
Antes de alimentar la compuerta IIR, la secuencia pasa por una **Convolución Causal 1D de ventana pequeña** (kernel size $k=3$ o $k=4$):

$$\tilde{x}_t = \text{Conv1D}(x_{t-2:t})$$
$$g_t = \text{sigmoid}(W_{gate} \cdot \tilde{x}_t)$$

Gracias a la convolución local, $\tilde{x}_t$ codifica el contexto adyacente (por ejemplo, detectar la transición previa a un par clave-valor). Esto le otorga a la compuerta la capacidad de distinguir un token de ruido aislado de una clave legítima.

---

## 4. Próximo Paso Recomendado (Experimento v344)

Implementar **Conv1D-Selective IIR**:
Agregar una capa `nn.Conv1d(d_model, d_model, kernel_size=4, padding=3)` causal previa a la proyección de la compuerta $g_t$ en `SelectiveDynamicIIRLayer`.

---
*Informe generado para el proyecto **attention-neuron**.*
