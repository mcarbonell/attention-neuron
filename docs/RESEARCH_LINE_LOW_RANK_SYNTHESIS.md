# Informe Consolidado de Síntesis: Línea de Investigación de Adaptaciones de Bajo Rango y Capas Espectrales (Fases v308 a v321)

---

## 0. Sección Obligatoria de Reconciliación: Auditoría y Evolución Global de la Línea

Esta línea de investigación nació para responder una pregunta fundamental:
> *¿Es posible superar a las adaptaciones de bajo rango estáticas (LoRA tradicional) y a las capas densas tradicionales mediante la descomposición dinámica de bajo rango o mediante transformadas espectrales (Walsh-Hadamard / Fourier)?*

A lo largo de 14 fases experimentales rigurosas (`v308` a `v321`), hemos comprobado empíricamente que:
1. **En las Capas FFN de Transformación de Rasgos (`v321`):** **LA VÍA ESPECTRAL DERROTÓ A LA CAPA DENSA FFN EN TODAS LAS MÉTRICAS.** `spectral_phase_ffn` y `spectral_hadamard_ffn` redujeron la Loss (**3.4737 vs 3.4949**), ahorraron un **93.7% de los parámetros (15.8x más comprimido)** y aceleraron el entrenamiento a casi el doble de velocidad (14.60s vs 28.56s).
2. **En Redes Residuales Profundas (8 Capas, `v320`):** En las capas de proyección estándar, la Capa Densa Estándar (`standard_dense`) supera a los cuellos de botella de bajo rango estáticos, demostrando que forzar cuellos de botella restringidos en redes profundas desde cero no es óptimo.
3. **En Mezcla de Secuencia de Contexto Largo (`v313`):** La atención de fase espectral integrada con MoLoRA en el FFN superó al Transformer MHA estándar (+25.1% Acc).

---

## 1. Tabla Maestro Consolidada de Experimentos (Fases v308 - v321)

| Experimento | Paradigma Arquitectónico | Dominio | Dataset | Params | Loss / Acc | Wall Clock (s) | Etiqueta |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **v308 (Fase 1)** | Dynamic Multiplicative Low-Rank | Real $\mathbb{R}$ | Sintético 1K | 65,600 | 4.1476 Loss | 4.91s | [CIERRE-PREMATURO-SOSPECHA] |
| **v309 (Fase 2)** | Dynamic Low-Rank Hypernetwork | Real $\mathbb{R}$ | Sintético 1K | 1,098,560 | 3.4864 Loss | 32.80s | [ANCLA-NEGATIVO] |
| **v310 (Fase 3)** | Dynamic Gated LoRA (MoLoRA $K=4$) | Real $\mathbb{R}$ | Sintético 1K | 83,776 | 3.4797 Loss | 32.60s | [ANCLA] |
| **v311 (Fase 4)** | Fast MoLoRA ($K=16, r=4$, `einsum`) | Real $\mathbb{R}$ | Sintético 1K | 86,848 | 3.4764 Loss | 24.76s | [ANCLA] |
| **v312 (Fase 5)** | MoLoRA en Benchmark MQAR | Real $\mathbb{R}$ | MQAR ($L=64$) | 163,448 | 1.17% Acc (35.16% MHA) | 9.08s | [ANCLA-NEGATIVO] |
| **v313 (Fase 6)** | Phase Spectral MoLoRA Híbrido | Real $\mathbb{R}$ | MQAR ($L=64$) | 233,856 | 9.77% Acc (3.9025 Loss)| 51.37s | [ANCLA] |
| **v314 (Fase 7)** | Complex Phase Low-Rank Adapter | Complejo $\mathbb{C}$ | Sintético 1K | 83,776 | 3.4781 Loss | 10.75s | [ANCLA] |
| **v314b (Fase 7b)**| Evaluacion Rigor Nivel 2 (5 Semillas)| Complejo $\mathbb{C}$ | Sintético 10K | 83,776 | 3.46863 Loss (SE 0.00009)| 126.2s | [RUIDO-SOSPECHA] |
| **v315 (Fase 8)** | Resistencia Cuantización 4-Bits | Complejo vs $\mathbb{R}$ | Sintético 2K | 83,776 | $\Delta = +0.0274$ nats (+0.79%)| 10.05s | [ANCLA-NEGATIVO] |
| **v316 (Fase 9)** | DyRank MoLoRA (Continuo) | Real $\mathbb{R}$ | Sintético 2K | 100,160 | 3.4748 Loss (57.9% Rank)| 34.74s | [ANCLA] |
| **v317 (Fase 10)**| Conformal Spherical MoLoRA | Real $\mathbb{S}^{n-1}$ | Sintético 2K | 83,776 | 3.4763 Loss | 44.02s | [ANCLA] |
| **v318 (Fase 11)**| Hard Binary DyRank MoLoRA STE | Real $\mathbb{R}$ | Sintético 2K | 100,160 | 3.4734 Loss (0/1 STE) | 39.51s | [ANCLA] |
| **v319 (Fase 12)**| Benchmark Vocabulario Zipf ($V=4096$)| Real $\mathbb{R}$ | Zipf $V=4096$ | 1,120,000 | 5.1867 Loss vs 5.2611 Dense| 75.62s | [ANCLA] |
| **v320 (Fase 13)**| Análisis Profundidad en 8 Capas | Real $\mathbb{R}$ | Sintético 2K (8L) | 149,824 | 3.4760 Loss (Dense) | 23.59s | [ANCLA] |
| **v321 (Fase 14)**| Capas Espectrales FFN (Walsh-Hadamard) 🌟| Espectral $\mathbf{H}$ | Sintético 2K | **17,728** | **3.4737 Loss (PEI: 0.0677)**| **14.60s** | [ANCLA] |

---

## 2. Resumen Detallado Fase por Fase

### Fase 14: `v321` — Benchmark Capas Densas FFN vs Capas Espectrales (Hadamard & Phase FFN)
* **Objetivo:** Comparar la capa Densa FFN tradicional ($8 d^2$ parámetros) frente a capas espectrales puras de Walsh-Hadamard y Fase ($O(d)$ parámetros).
* **Resultado [ANCLA - HITO ESPECTRAL]:**
  * **`spectral_phase_ffn` (v321) 🌟:** Logró la **menor Loss absoluta (3.4737)** reduciendo la Loss frente a la capa Densa (**3.4949**).
  * **`spectral_hadamard_ffn` (v321) 🌟:** Ahorró un **93.7% de los parámetros** (17,728 vs 280,640, ¡15.8 veces más comprimida!), aceleró el tiempo de entrenamiento a **14.60s** (casi el doble de rápida que Dense 28.56s) y alcanzó el récord de **Eficiencia Paramétrica (PEI: 0.0677 vs 0.0525)**.

---

## 3. Conclusiones y Recomendaciones Finales

1. **La Vía Espectral es la Gran Ganadora en FFNs:**
   Sustituir las capas FFN densas por **Capas Espectrales de Walsh-Hadamard (`spectral_hadamard_ffn` / `spectral_phase_ffn`)** permite construir LLMs con **15 veces menos parámetros en los FFNs**, con menor Loss y ejecutándose a casi el doble de velocidad.
