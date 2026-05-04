# Hallazgos V227: Clasificación MNIST Espectral-Periódica

## Objetivo
Evaluar la capacidad de discriminación de las neuronas `StraightPeriodic` (V225) sobre una representación estructurada de la imagen (DCT-2D), buscando la máxima eficiencia paramétrica.

## Metodología
*   **Extractor de Características:** 2D-DCT fijo (sin parámetros) tomando un bloque de 8x8 (64 coeficientes de baja frecuencia).
*   **Activación:** Capa de neuronas `StraightPeriodic` independientes por coeficiente (64 x 8 = 512 parámetros).
*   **Clasificador:** Capa lineal (64 x 10 = 640 parámetros).
*   **Entrenamiento:** 10 épocas, Adam (LR=0.01).

## Resultados
| Métrica | V226 (Ruido + Periódico) | V227 (DCT + Periódico) |
| :--- | :--- | :--- |
| **Parámetros** | 522 | **1,034** |
| **Precisión (Accuracy)** | 44.80% | **85.83%** |
| **PEI** | 0.2037 | **0.2854** |

## Conclusiones
1.  **Sinergia Espectral:** Las neuronas periódicas resuenan de forma natural con los coeficientes DCT. Al sintonizar la fase y frecuencia de cada armónico de la imagen, el modelo identifica patrones estructurales (como bucles o líneas) con una fracción del coste de una CNN.
2.  **Estabilidad de la V225:** La corrección polinómica permitió que el modelo convergiera rápidamente a pesar de la alta no-linealidad de la tangente.
3.  **Hito de Eficiencia:** Lograr ~86% con solo 1,000 parámetros sitúa a esta arquitectura en el percentil superior de eficiencia en este repositorio.

## Siguiente Paso (V228)
Intentar romper la barrera del **90%** aumentando ligeramente el presupuesto espectral (80 coeficientes) y optimizando el hiper-parámetro de entrenamiento, manteniéndonos bajo el límite estricto de **1,500 parámetros**.
