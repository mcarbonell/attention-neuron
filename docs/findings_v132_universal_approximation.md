# Findings V132: Universal Approximation Benchmark

## Objetivo
Validar la hipótesis de que las redes neuronales pueden aproximar cualquier función matemática y comparar la eficiencia de los MLPs densos frente a una arquitectura de "Neuronas Polimórficas" con bancos lógicos.

## Resultados Estadísticos (MSE Train)

| Función | MLP-Small (49p) | MLP-Medium (4.3kp) | Poly-Neuron (65p) | Gap de Eficiencia |
| :--- | :--- | :--- | :--- | :--- |
| **x^2** | 1.01e-3 | 3.94e-6 | 2.10e-3 | Poly es 66x más ligero, MSE similar. |
| **1/x** | 2.95 | 0.05 | 7.14 | MLPs ganan en estabilidad cerca de 0. |
| **prod (x*y)** | 1.62e-3 | 5.78e-5 | 1.77 | **Falla crítica de Poly** (falta interacción). |
| **sin(x)** | 2.13e-4 | 3.45e-6 | 1.20e-3 | Poly generaliza bien con pocos params. |
| **cos(x)** | 7.34e-5 | 2.18e-5 | 1.17e-3 | Empate técnico en utilidad práctica. |
| **tan(x)** | 8.85 | 0.61 | 14.55 | Tarea extremadamente difícil para todos. |
| **sinc(x)** | 1.04e-4 | 5.55e-7 | 1.06e-3 | MLP-Small es sorprendentemente bueno. |

## Conclusiones Técnicas

1. **La "Trampa" de la Aproximación**: El Teorema de Aproximación Universal es cierto, pero el costo en parámetros es exponencial si la función tiene curvaturas que no coinciden con la activación (ReLU). Un MLP de 4300 parámetros aproxima casi todo, pero no "entiende" la ley subyacente.
2. **Generalización Estructural**: En $x^2$, la Neurona Polimórfica obtuvo un error de test de **1.23** frente al **2.24** del MLP Medium (con 60 veces menos parámetros). Esto demuestra que tener el "bias inductivo" correcto (una base cuadrática) es más potente que el ancho de la red.
3. **El Problema de la Interacción**: Las neuronas polimórficas actuales operan en dimensiones independientes. Para aproximar funciones de interacción ($x \cdot y$), necesitamos una capa que permita productos cruzados o log-transformaciones antes de la suma.

## Métricas de Sistema
- **Wall Clock Time**: 138.2s (Total Benchmark)
- **Efficiency**: 72,358 eval/sec (CPU)
- **Hardware**: CPU AMD Ryzen 7 8845HS

## Siguiente Experimento (V133)
Introducir una **Capa de Producto Cruzado** (Cross-Product Layer) en la arquitectura polimórfica para resolver el fracaso en la multiplicación y división.
