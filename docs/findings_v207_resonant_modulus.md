# Hallazgos V207: El Desafío del Módulo con Resonancia de Fase

## Objetivo
Evaluar si la Neurona de Resonancia de Fase (V205), al poseer una base nativa en ondas (cosenos), puede resolver la función módulo ($x \pmod y$) mejor que las redes continuas clásicas (MLP) y las redes polimórficas (V193), superando el "Muro de la Discontinuidad".

## Resultados (x % y)

| Modelo | Parámetros | Train MSE | Far OOD MSE | Ratio (Estabilidad) |
| :--- | :--- | :--- | :--- | :--- |
| **MLP-Huge** | 1,052,673 | **0.0055** | 4.11 | 743 |
| **Poly-Deep-V193** | 28,385 | 0.0678 | **15.53** | **229** |
| **Resonant-Phase-V207** | **17,089** | 0.0612 | 29.62 | 484 |

## Conclusiones Técnicas y Error Conceptual

### 1. Superioridad Paramétrica Local
Con solo 17k parámetros, la red resonante logró igualar y superar ligeramente el error de entrenamiento de la red Poly-Deep (28k parámetros). Demuestra que las ondas son excelentes para aproximar la forma de sierra (Sawtooth) de manera local usando combinaciones de armónicos (Serie de Fourier).

### 2. El Fracaso de la Extrapolación (OOD)
A pesar del buen ajuste local, la red resonante colapsó en la extrapolación (OOD MSE de 29.62). El análisis matemático revela el porqué:
El módulo $x \pmod y$ es una función periódica donde **la frecuencia depende de la variable $y$** (Frecuencia $\propto 1/y$). 
Nuestra capa de resonancia calcula la fase como una combinación lineal: $\cos(w_1 x + w_2 y)$. 
**Matemáticamente, es imposible que $w_1 x + w_2 y$ represente $x/y$ para todos los valores.** La red se ve obligada a hacer una aproximación de Taylor bidimensional (memorización local) que se rompe por completo cuando la variable $y$ sale del rango de entrenamiento.

### 3. El Siguiente Muro
Esto nos confirma el Hallazgo V194 original: el problema del Módulo no es solo la discontinuidad (salto abrupto), sino la **composición multiplicativa / divisiva de variables**.
Las Redes de Resonancia actuales son excelentes si las frecuencias son fijas (como en las imágenes MNIST donde cada píxel tiene una posición fija). Pero para razonamiento abstracto, las neuronas necesitan poder **modular dinámicamente sus propias frecuencias** basándose en las entradas de otras neuronas (Interacción Multiplicativa).
