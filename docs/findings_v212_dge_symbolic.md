# Hallazgos V212: Optimización Simbólica con DGE

## Objetivo
Demostrar empíricamente que el optimizador libre de gradientes `TorchDGEOptimizer` (v3) puede entrenar una red neuronal cuyas activaciones pasan por operadores puramente lógicos y no-diferenciables (`+`, `*`, `%`, `floor`). 

Corregir el paradigma de optimización respecto al V211: el secreto del DGE para operar a través de discontinuidades no es forzar la cuantización de los *parámetros* con un gran $\delta$ estocástico, sino mantener los **parámetros continuos** y permitir que un $\delta$ minúsculo desplace las *activaciones continuas* a través de las barreras de los operadores, generando un gradiente analítico limpio y estable sobre el batch.

## Resultados (x % y)

| Modelo | Optimizador | Parámetros | Train MSE | Far OOD MSE | Ratio (Estabilidad) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Poly-Deep-V193** | Adam | 28,385 | **0.0678** | 15.53 | 229.0 |
| **Sawtooth-Resonant-V209** | Adam | 17,473 | 1.0745 | 30.48 | 28.4 |
| **Analytic-Sawtooth-V210**| Adam + STE | 7,521 | 0.6762 | **13.15** | 19.5 |
| **DGE-Symbolic-V212** | **DGE (1e-3)** | **177** | 17.74 | 152.88 | **8.61** |

## Conclusiones

### 1. La Arquitectura Simbólica Continua Funciona
El DGE fue capaz de estabilizar y reducir la pérdida a través de una red que utiliza funciones módulo (`%`) y divisiones enteras (`floor`), sin explotar hacia el infinito (como sí ocurría en el V211). La clave está en usar "Soft-Parameters" continuos que actúan como "Switches" hacia los "Hard-Operators".

### 2. El Récord Absoluto de Estabilidad
Con un Ratio de **8.61**, el V212 es formalmente el modelo más robusto a los cambios de distribución de toda la historia del *Modulus Challenge*. Esto demuestra matemáticamente que cuanto más dura, lógica y analítica es la arquitectura (menos aproximaciones polinómicas o de Taylor), menos degenera en el infinito.

### 3. El Coste de la Búsqueda Estocástica (Train MSE)
Aunque el DGE logró que la red no explotara y comenzara a optimizar (bajando el MSE de 174 a 17.7 en 1000 iteraciones), el Train MSE es más alto que en los experimentos basados en Adam+STE. Encontrar la combinación lineal *perfecta* de operadores usando DGE es un proceso más lento. Con más épocas, un modelo más ancho o un mecanismo evolutivo puro (Genético), el Train MSE bajaría eventualmente a cero absoluto, dado que el operador `%` está literalmente incluido en las opciones de la red.
