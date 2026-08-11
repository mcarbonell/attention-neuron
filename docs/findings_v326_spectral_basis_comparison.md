# Hallazgos Experimento v326: Benchmark Comprensivo de Bases Espectrales (Fase 5)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

* **Conclusión Previa Modificada:** En experimentos anteriores solo se había evaluado la base espectral discreta de Walsh-Hadamard (FWHT).
* **Resultado Certificado del Experimento v326 [ANCLA]:** **VICTORIA ABSOLUTA DE LA DISCRETE COSINE TRANSFORM (DCT-II).**
  1. **DCT-II es la Base Espectral Superior (97.55% Acc, PEI 1.4558):** La **DCT-II (Discrete Cosine Transform)** alcanzó la **menor loss (0.1177)** y la mayor precisión (**97.55% Accuracy**), superando ligeramente a Walsh-Hadamard (**97.29% Accuracy**).
  2. **Concentración Armónica de Cosenos Reales:** La proyección ortogonal de cosenos reales $\cos\left( \frac{\pi k (2n+1)}{2d} \right)$ concentra la energía espectral en los armónicos continuos de forma más suave que los saltos binarios $\pm 1$ de Walsh-Hadamard.
  3. **Complementariedad FWHT vs DCT-II:** **FWHT** sigue siendo el rey de la eficiencia en hardware digital y ejecución acelerada (222s vs 242s), mientras que **DCT-II** ofrece la mayor capacidad expresiva por coeficiente de frecuencia.

---

## 1. Tabla de Resultados Empíricos

* **Configuración:** $N=2000$ secuencias estructuradas, $L=64, d=128$, 5 capas residuales espectrales, 10 épocas, AdamW ($lr=1e-3, \text{weight\_decay}=0.0$). Evaluado en CPU (8 hilos).

| Base Espectral Ortonormal | Dominio de Proyección | Parámetros | Loss Final | Accuracy % | Wall Clock (s) | PEI | Etiqueta |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`DCT-II (Discrete Cosine)`** 🌟 | Cosenos Reales Armónicos | 685,184 | **0.1177** | **97.55%** | 242.59 | **1.4558** | [ANCLA] |
| **`FWHT (Walsh-Hadamard)`** 🌟 | Binario Discreto $\pm 1$ | 685,184 | 0.1437 | 97.29% | **222.70** | 1.1922 | [ANCLA] |
| **`DWT Haar (Wavelet)`** | Ondículas Multi-Resolución | 685,184 | 0.7814 | 77.65% | 194.02 | 0.2193 | [ANCLA] |
| **`FFT (Real Fast Fourier)`** | Fase Compleja | **348,564** | 2.5715 | 27.17% | **102.60** | 0.0702 | [ANCLA] |

*Nota: El marcador 🌟 asigna la menor Loss y mayor PEI a `DCT-II` (0.1177 Loss, 97.55% Acc).*

---

## 2. Recomendación para la Arquitectura Final de LLM (`tiny-thinker`)

Este benchmark demuestra que el All-Spectral Transformer puede operar opcionalmente con un **FFN Espectral Híbrido DCT-II / FWHT**, combinando la máxima fidelidad de representación de DCT-II con la velocidad de ejecución de Walsh-Hadamard en la CPU.

---

## Auditoría posterior y amenazas a la validez (2026-08-10)

La preferencia DCT-II observada a 10 épocas se invierte en v326b a 25 épocas, lo que muestra sensibilidad al presupuesto de optimización. Sin validación, semillas ni una base ortogonal aleatoria congelada, no puede atribuirse el efecto a la geometría particular de DCT/FWHT. Es un screening de bases bajo este schedule, no una recomendación para LM. Véase la [auditoría transversal v300–v329](findings_v300_v329_audit.md).
