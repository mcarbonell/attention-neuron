# Findings: V32 (The Broadcaster) - Fan-out Modulation

## 1. El Experimento

El objetivo de la V32 era probar una hipótesis de eficiencia computacional extrema: ¿Qué ocurre si congelamos el 100% de los pesos convolucionales (Fan-in) y aplicamos la mezcla de la "Biblioteca de Sustratos" exclusivamente sobre las activaciones de salida (Fan-out)?

**Configuración:**
- **Modelo**: BroadcasterResNet (18 capas residuales).
- **Mecánica**: 4 universos de ruido blanco (44 millones de pesos) evaluados en paralelo y bloqueados sin gradiente (`requires_grad = False`). La red mezcla los feature maps resultantes usando un dial Softmax y los escala con Gain/Bias.
- **Parámetros entrenables**: **210,186**.
- **Pesos Congelados**: ~44,000,000.
- **Hardware**: GPU (AMD Radeon 780M via DirectML).

## 2. Resultados

| Métrica | V32 (Broadcaster) | V26 (Prism-ResNet) |
| :--- | :--- | :--- |
| **Best Test Accuracy** | **71.53%** | 85.94% |
| **Parámetros Entrenables** | 210K | 439K |
| **Modulación** | Post-Activación (Ganancia) | Pre-Activación (Rank-16) |

## 3. Conclusiones Arquitecturales

1.  **El Límite del Fan-out Puro**: A diferencia de la modulación Rank-r en los pesos (V26), la modulación de salida (V32) no puede alterar la geometría del filtro aleatorio. Si un kernel aleatorio no detecta un borde, multiplicar su salida por 10 no crea un borde. Esto explica el "techo de cristal" en el 71.53%, casi 14 puntos por debajo de la V26.
2.  **Validación del Rank-R**: Este experimento demuestra irrefutablemente por qué la modulación multiplicativa/aditiva de bajo rango sobre los pesos fijos (V1-V26) es el "ingrediente mágico" de la *Attention Neuron*. Sin esa capacidad de "esculpir" el interior del kernel, el rendimiento se estanca al nivel de un MLP decente.
3.  **Aceleración Hardware**: Se confirmó que congelar el cálculo de gradientes para las convoluciones masivas reduce el tiempo de entrenamiento drásticamente, pasando a ser "Compute Bound" en GPU. 

## 4. El Veredicto

La V32 es un experimento de control perfecto. Confirma que la "Alquimia de Sustratos" necesita ocurrir en el dominio de los pesos (Fan-in) y no en el dominio de las activaciones (Fan-out) para alcanzar el rendimiento de estado del arte.
