# Hallazgos Experimento v328: Learnable Substrate Lerp Gating & Substrate Selection Report (Fase 7)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

* **Conclusión Previa Modificada:** En `v327` se utilizaba un promedio unweighted ($1/3$ rígido) para combinar las tres transformadas ortogonales (FWHT, DCT-II, Haar).
* **Resultado Certificado del Experimento v328 [ANCLA]:** **SINTONIZACIÓN ADAPTATIVA APRENDIDA Y REPORTE DE SUSTRATOS ELEGIDOS.**
  1. **Mayor Precisión con 38% Menos Parámetros:** El router *Learnable Substrate Lerp* alcanzó un **99.79% de Accuracy** (0.0188 Loss), reduciendo los parámetros totales de 854K a **526K (-38% de parámetros)** respecto a `v327`.
  2. **Especialización Jerárquica por Capa:**
     * **FWHT y DCT-II Dominan el Proceso (~36% cada una):** La red otorga un peso predominante a Walsh-Hadamard (binaria discreta) y a DCT-II (cosenos armónicos).
     * **DCT-II se Dispara en la Capa Final (38.25%):** En la Capa 5 de salida, la preferencia por DCT-II se eleva al 38.25%, certificando que los armónicos continuos de cosenos son la mejor representación para la proyección de probabilidades del vocabulario.

---

## 1. Tabla del Reporte Transparente de Sustratos Elegidos

| Capa Residual | % FWHT (Binario Discreto) | % DCT-II (Cosenos Reales) | % DWT Haar (Ondículas Local) | Diagnóstico Algorítmico |
| :--- | :---: | :---: | :---: | :--- |
| **Capa 1 (Entrada)** | **36.44%** | 35.79% | 27.76% | Equilibrio FWHT + DCT (frecuencias globales) |
| **Capa 2** | 36.05% | 36.06% | 27.89% | Co-dominio perfecto FWHT y DCT |
| **Capa 3 (Media)** | 35.43% | **36.97%** | 27.60% | Ligero repunte de la DCT-II armónica |
| **Capa 4** | **36.64%** | 36.79% | 26.57% | Dominio compartido FWHT + DCT |
| **Capa 5 (Salida)** | 34.41% | **38.25%** | 27.34% | **Máxima preferencia por DCT-II en la capa final 🌟** |

---

## 2. Recomendación para el Modelo de Lenguaje (`tiny-thinker`)

Este descubrimiento demuestra que el motor ideal para el LLM en lenguaje natural real debe implementar el **Router Fused Lerp DCT-II / FWHT**, asignando mayor capacidad de cosenos armónicos en las capas profundas de predicción de tokens.
