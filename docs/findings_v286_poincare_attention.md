# Hallazgos del Experimento: Poincaré Hyperbolic Attention con Proyección Soft-Tanh (v286)

Este documento resume los resultados obtenidos tras refinar el experimento **v286** introduciendo una proyección suave de tangente hiperbólica (Soft-Tanh) en lugar del clipping rígido, comparándola con la atención euclidiana estándar en la búsqueda de ancestros.

## 1. Configuración Experimental
- **Estructura del Árbol:** Grado de ramificación $K=5$, Profundidad $D=3$ (156 nodos en total).
- **Relaciones Evaluadas:** Padre (1-hop), Abuelo (2-hop) y Bisabuelo (3-hop).
- **Tamaño del Dataset:** 430 muestras (344 para entrenamiento, 86 para prueba).
- **Mapeo al Disco (Soft-Tanh):** 
  $$\text{proj}(x) = (1 - \epsilon) \cdot \tanh(\|x\|) \cdot \frac{x}{\|x\|}$$
- **Protocolo de Entrenamiento:** 120 épocas, optimizador Adam con LR=$5.00\times 10^{-3}$, Weight Decay=$1.00\times 10^{-5}$, tamaño de lote 32, promediado sobre **5 semillas independientes** ([42, 43, 44, 45, 46]).
- **Hardware:** AMD Ryzen 7 8845hs, ejecutado en CPU.

---

## 2. Resumen Estadístico de Resultados (Soft-Tanh)

A continuación se muestra la comparación de rendimiento para cada dimensión de embedding $d \in \{2, 4, 8, 16, 32, 64\}$:

| Dimensión ($d$) | Atención | Precisión Test (Promedio $\pm$ Desv. Est.) | Loss Test (Promedio) | PEI (Parametric Efficiency) | Parámetros Totales |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **d = 2** | Poincaré (Soft-Tanh) | **36.51%** $\pm$ 1.74% | 3.3030 | **0.1256** | 805 |
| | Euclidiana | 34.88% $\pm$ 4.29% | **2.9819** | 0.1200 | 804 |
| **d = 4** | Poincaré (Soft-Tanh) | **38.37%** $\pm$ 2.08% | 4.9217 | **0.1211** | 1,477 |
| | Euclidiana | 35.35% $\pm$ 3.34% | **4.6935** | 0.1115 | 1,476 |
| **d = 8** | Poincaré (Soft-Tanh) | **37.44%** $\pm$ 2.48% | **5.3916** | **0.1082** | 2,893 |
| | Euclidiana | 36.74% $\pm$ 5.72% | 5.5158 | 0.0817 | 2,892 |
| **d = 16** | Poincaré (Soft-Tanh) | **38.60%** $\pm$ 1.14% | 5.7991 | **0.1022** | 6,013 |
| | Euclidiana | 37.91% $\pm$ 2.16% | **4.8893** | 0.1003 | 6,012 |
| **d = 32** | Poincaré (Soft-Tanh) | **41.63%** $\pm$ 2.98% | 3.7153 | **0.1009** | 13,405 |
| | Euclidiana | 37.44% $\pm$ 3.15% | **2.4579** | 0.0907 | 13,404 |
| **d = 64** | Poincaré (Soft-Tanh) | **43.49%** $\pm$ 2.28% | 3.1315 | **0.0963** | 32,797 |
| | Euclidiana | 31.86% $\pm$ 0.57% | **2.1137** | 0.0706 | 32,796 |

---

## 3. Análisis de Hallazgos Clave

### A. Ventaja Hiperbólica Preservada
El refinamiento mediante proyección suave (Soft-Tanh) mantiene la superioridad sistemática de la atención geodésica sobre la euclidiana tradicional en **todas** las dimensiones evaluadas. 
- En $d=64$, donde el modelo Euclidiano convencional se ve gravemente afectado por el sobreajuste y cae a **31.86%** de precisión, Poincaré conserva un rendimiento estable de **43.49%** (una brecha absoluta del **+11.63%**).

### B. Distribución Radial Correcta y Estabilidad
La introducción de Soft-Tanh no solo estabilizó la optimización (disminuyendo la desviación estándar en la mayoría de las configuraciones, como en $d=2$ donde baja del 2.71% al **1.74%** y en $d=16$ del 3.24% al **1.14%**), sino que ha resuelto por completo la acumulación en la frontera observada inicialmente. Los gradientes suaves permiten a los embeddings jerárquicos distribuirse armónicamente.

---

## 4. Visualización del Disco de Poincaré Refinado ($d=2$)

La proyección auto-organizada de las claves demuestra cómo la formulación Soft-Tanh distribuye los nodos de forma geodésicamente correcta a lo largo y ancho del disco unitario, resolviendo el colapso perimetral:

![Visualización del Disco de Poincaré Refinado](../results/figures/v286_poincare_disk.png)
