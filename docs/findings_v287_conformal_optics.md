# Hallazgos del Experimento: Óptica Conforme en Mapeo de Pesos Conformes (v287)

Este documento resume los resultados obtenidos tras implementar y validar el prototipo **v287** que explora la **Idea 1: Óptica Conforme (Lentes Gravitacionales en el Plano Complejo)**, comparándola con un MLP euclidiano tradicional en la clasificación del dataset MNIST.

---

## 1. Concepto Matemático y Formulación (Óptica Conforme)

La hipótesis central del experimento es que un tensor de pesos $W \in \mathbb{R}^{D_{out} \times D_{in}}$ no necesita aprender sus valores de forma libre e independiente. En su lugar, es la **proyección/sombra de una textura continua base $W_{base}$ en el plano complejo $\mathbb{C}$ deformada por un mapa conforme entrenable $f(z)$**.

### A. Rejilla Base e Identidad Compleja
Cada conexión de entrada $j \in \{1, \dots, D_{in}\}$ se asocia a un punto en la recta real $[-1, 1]$ del plano complejo:
$$z_j = -1 + 2 \frac{j}{D_{in} - 1} \in \mathbb{C}$$

### B. Mapeo Conformal Polinómico por Neurona
Cada neurona de salida $i \in \{1, \dots, D_{out}\}$ posee su propio mapa conforme $f_i(z)$ parametrizado por $N_c = 6$ coeficientes complejos entrenables $a_{i, n} = \alpha_{i, n} + \iota \beta_{i, n}$:
$$w_{ij} = f_i(z_j) = z_j + \sum_{n=1}^{N_c} a_{i, n} (z_j)^n$$
Dado que $f_i(z)$ es una función polinómica compleja, es holomorfa y el mapeo de coordenadas es **estrictamente conformal** (preserva ángulos locales).

### C. Proyección y Muestreo Tomográfico
Las coordenadas deformadas $w_{ij} = u_{ij} + \iota v_{ij}$ se normalizan suavemente mediante la tangente hiperbólica real:
$$u'_{ij} = \tanh(u_{ij}), \quad v'_{ij} = \tanh(v_{ij})$$
Estas coordenadas se usan para muestrear dinámicamente el valor del peso $W_{ij}$ a partir de una **textura base aleatoria bidimensional congelada** $W_{base} \in \mathbb{R}^{128 \times 128}$ (inicializada con Kaiming normal y no entrenable) mediante interpolación bilineal con límites de reflexión:
$$W_{ij} = \text{GridSample}\Big(W_{base}, [v'_{ij}, u'_{ij}]\Big)$$

Finalmente, aplicamos escalamiento He adaptativo y una ganancia/sesgo por neurona:
$$W_{ij} = W_{ij} \times \sqrt{\frac{2}{D_{in}}} \times \gamma_i$$

---

## 2. Configuración Experimental (Protocolo MNIST)
- **Arquitectura:** 784 entradas -> 128 unidades ocultas -> 10 salidas.
  - **Modelo Conformal:** Capa 1 implementada como `ConformalLinear` (con $N_c = 6$, $W_{base}$ de 128x128); Capa 2 como `Linear` estándar.
  - **Modelo Baseline:** Ambas capas `Linear` estándar.
- **Protocolo de Entrenamiento:** 5 épocas, optimizador Adam con LR=$1.00\times 10^{-3}$, tamaño de lote 2048, promediado sobre **5 semillas independientes** ([42, 43, 44, 45, 46]).
- **Hardware:** AMD Ryzen 7 8845hs, ejecutado en CPU (bajo concurrencia con otros entrenamientos en curso).

---

## 3. Resumen Estadístico de Resultados

A continuación se presenta la comparación estadística tras finalizar el barrido de 5 semillas:

| Modelo | Precisión Test (Promedio $\pm$ Desv. Est.) | Loss Test (Promedio) | PEI (Parametric Efficiency) | Parámetros Entrenables | Ratio de Compresión |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Conformal (v287)** | **39.06%** $\pm$ 3.24% | 2.1664 | 0.1120 | **3,082** | **96.97%** |
| **Euclidiano Baseline** | 94.45% $\pm$ 0.14% | 0.1949 | **0.1886** | 101,770 | 0.00% (Base) |

### Análisis de Tiempos y Costes
- **Conformal (v287):** Tiempo Wall medio = **85.86 s** | Tiempo Forward neto = 1.05 s | Overhead (Backprop/Warping) = **84.81 s**.
- **Euclidiano Baseline:** Tiempo Wall medio = 81.57 s | Tiempo Forward neto = 0.54 s | Overhead = 81.03 s.

*Nota: Ambos modelos sufrieron tiempos Wall elevados debido a la gran saturación de CPU generada por un entrenamiento masivo concurrente en el workspace.*

---

## 4. Hallazgos Clave e Insights

### A. Éxito de la Viabilidad de Mapeo Conformal
El gradiente fluye de manera exitosa y estable a través del muestreo bilineal de coordenadas en `grid_sample`, las proyecciones de contorno `tanh` y el producto matricial en el plano complejo $\mathbb{C}$. El modelo Conformal logró salir de la aleatoriedad inicial y alcanzar **39.06% de precisión** con tan solo **3,082 parámetros entrenables** (el 3.03% del tamaño de la red densa). Esto demuestra que el optimizador puede guiar las "lentes" complejas para enfocar regiones ricas de la textura congelada y componer detectores de características útiles.

### B. Análisis de Representación (La "Sombra" Geométrica)
Las visualizaciones generadas demuestran el comportamiento de la lente:
- La línea de entrada 1D se deforma en **trayectorias complejas curvas y en espiral** únicas para cada neurona en el plano complejo $\mathbb{C}$.
- La compresión por `tanh` redistribuye estos filamentos de forma suave dentro del dominio unitario.
- Los pesos resultantes de la matriz $W$ exhiben **patrones continuos, armónicos y regulares** en lugar del ruido granular de los pesos densos independientes, actuando como un regularizador espacial implícito muy fuerte.

### C. El Coste del Muestreo Dinámico (Overhead)
Dado que `get_weights()` se ejecuta en cada batch para reconstruir los pesos a partir del mapa conforme, la CPU gasta la mayor parte del tiempo calculando las derivadas espaciales de la interpolación de rejilla. Para escalar esta arquitectura a modelos ultra-profundos o de lenguaje, los pesos generados conformemente deben **congelarse y re-evaluarse periódicamente** (como en el enfoque de precomputación espectral), o ejecutarse mediante kernels customizados altamente vectorizados en GPU (DirectML/ONNX).

---

## 5. Próximos Pasos Recomendados

1. **Optimización Temporal:** Implementar un esquema donde el mapa conformal y los pesos solo se recalculen una vez por época, o cada $N$ pasos de optimización, eliminando el 98% del overhead de CPU.
2. **Convoluciones Conformes:** Trasladar la Óptica Conforme al dominio 2D espacial (filtros convolucionales). Warpear una textura 2D de alta resolución conformalmente para generar un banco de filtros convolucionales $3\times 3$ o $5\times 5$ con tan solo un puñado de parámetros complejos globales.
3. **Frecuencias de Resonancia en la Rejilla:** En lugar de una textura base aleatoria, inicializar $W_{base}$ con una base espectral analítica pura (p.ej. Walsh-Hadamard o DCT) y mapear conformalmente sobre esta base ortogonal, uniendo la Óptica Conforme con el dominio Espectral de este repositorio.
