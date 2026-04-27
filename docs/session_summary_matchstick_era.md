# Findings: The Matchstick Neuron Era (v51 - v58)

## 1. Conclusiones Algorítmicas
Tras recorrer desde la simplificación de trazos hasta la inicialización estructurada, estas son las verdades fundamentales descubiertas:

- **La Línea es el Átomo (v51)**: Superar el 98.3% en MNIST con solo 6 parámetros por neurona demuestra que la "atención geométrica lineal" es el sesgo inductivo más eficiente para el reconocimiento de formas básicas. Menos es más: las curvas de Bézier (v50) y las líneas dobles (v52) solo añadieron ruido y dificultad de optimización.
- **Ceguera Geométrica (v55-v56)**: Las neuronas de atención local son "ciegas" a lo que no está bajo su campana de Gauss. La inicialización central (v55) falla no por redundancia, sino por falta de gradiente. Los intentos de solucionar esto con "visión borrosa" (v56) fracasan por aplanamiento del gradiente (Fog Plateau).
- **Orden vs. Caos (v57-v58)**: 
    - La **Rejilla (Grid)** es el camino más rápido a la estabilidad (95% en la primera época). Es ideal para sistemas que necesitan fiabilidad y cobertura total inmediata.
    - El **Azar (Random)** es el camino al rendimiento pico. La diversidad estocástica inicial permite encontrar nichos geométricos que la rejilla tarda demasiado en explorar.

## 2. Desempeño por Dataset

| Dataset | Versión | Configuración | Best Acc | Eficiencia |
| :--- | :--- | :--- | :--- | :--- |
| **MNIST** | v51 | 256 Neuronas / Random | **98.30%** | ~130x compresión |
| **MNIST** | v57 | 256 Neuronas / Grid | 97.16% | Máxima estabilidad |
| **CIFAR-10** | v54 | 512 Neuronas / Random RGB | **61.18%** | Señal fuerte en 10 epochs |
| **CIFAR-10** | v58 | 512 Neuronas / Grid RGB | 58.65% | Cobertura total |

## 3. Estado del Arte del Proyecto
Hemos validado que la **Atención Geométrica** puede sustituir a las capas densas iniciales con una fracción de los parámetros, manteniendo la interpretabilidad (podemos ver literalmente qué "trazos" busca la red).

## 4. Recomendación para Futuras Iteraciones
Para escalar este éxito, el siguiente paso natural no es añadir más parámetros a la neurona, sino **profundidad**. Una arquitectura de 2 o 3 capas de Matchsticks donde las capas superiores combinen los trazos de las inferiores (formando ángulos, luego formas, luego objetos) podría ser la clave para romper el techo del 70-80% en CIFAR-10 de forma extremadamente parsimoniosa.
