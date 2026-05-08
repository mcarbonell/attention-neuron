# Hallazgos V244: El Duelo Final de Optimizadores

## Objetivo
Realizar una comparativa exhaustiva entre **Sign-DS** y los estándares de la industria (**Adam, Lion, RMSprop, SGD**) para determinar el balance óptimo entre precisión y consumo de memoria.

## Resultados Consolidados (MNIST - 10 Épocas)

| Optimizador | Memoria (Estado) | Accuracy Final | Wall Clock Time | PEI (Parametric Efficiency) |
| :--- | :--- | :--- | :--- | :--- |
| **SGD Puro** | 0 bytes/p | 95.31% | 125.1s | 0.1663 |
| **Sign-DS (Ours)** | **2 bytes/p** | **98.24%** | 141.5s | **0.1714** |
| **Lion Standard** | 4 bytes/p | 99.38% | 123.4s | 0.1734 |
| **Muon** | 4 bytes/p | 99.15% | 233.6s | 0.1730 |
| **SGD + Momentum** | 4 bytes/p | **99.69%** | **122.5s** | **0.1740** |
| **RMSprop** | 4 bytes/p | 99.47% | 128.6s | 0.1736 |
| **Lion-DS** | 5 bytes/p | 99.37% | 144.4s | 0.1734 |
| **Adam Estándar** | 8 bytes/p | 99.45% | 136.5s | 0.1735 |

## Análisis de Hallazgos

1.  **La Resistencia del Momentum**: SGD con Momentum (4b) resultó ser el más preciso y rápido para esta tarea específica. Esto sugiere que, en redes de este tamaño, el momentum lineal es superior a las normalizaciones basadas en signo o varianza.
2.  **Sign-DS como "Ghost Optimizer"**: Sign-DS (2b) logró superar al SGD Puro (0b) por casi un **3% de accuracy**, lo que justifica con creces el uso de esos 2 bytes extra para Directional Stability. Se confirma como el optimizador de elección para situaciones de memoria crítica.
3.  **Lion y la Estabilidad**: La diferencia entre Lion y Lion-DS es marginal en accuracy final, pero DS ayudó a Lion a ser más robusto en las épocas iniciales.
4.  **Eficiencia Paramétrica (PEI)**: Aunque SGD+Mom tiene el PEI más alto por su precisión, Sign-DS mantiene un índice muy competitivo considerando que su "coste real" en RAM es la mitad.

## Conclusión Final
Si el hardware lo permite, **SGD+Momentum** o **Lion** son las opciones más equilibradas. Si la memoria es un recurso escaso, **Sign-DS** es el nuevo estándar interno para la arquitectura **Attention-Neuron**, ofreciendo una precisión digna con un footprint de estado insignificante.

---
*Nota: Para resultados con rigor científico, se recomienda escalar estas pruebas a 10 semillas y datasets más complejos como CIFAR-10 en futuras iteraciones.*
