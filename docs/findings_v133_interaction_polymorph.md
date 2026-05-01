# Findings V133: Interaction Polymorphic Neurons

## Objetivo
Resolver el fallo de las neuronas polimórficas en tareas de interacción (multiplicación y división) mediante la introducción de un **Canal de Interacción** explícito (Bancos PROD y DIV).

## Comparativa de Rendimiento (V132 vs V133)

| Función | Poly V132 (65p) | Poly V133 (153-225p) | MLP Medium V132 (4.3kp) | Mejora Poly |
| :--- | :--- | :--- | :--- | :--- |
| **prod (x*y)** | 1.77 | **0.000049** | 0.000057 | **1000x mejor** |
| **sin(x)** | 0.0012 | **0.000341** | 0.000003 | 3.5x mejor |
| **sinc(x)** | 0.0010 | **0.000088** | 0.0000005| 11x mejor |
| **div (x/y)** | - | **12.45** | 0.05 (MLP) | Falla estabilidad |

## Conclusiones Técnicas

1.  **Dominio de la Multiplicación**: La introducción del canal `PROD` (proyecciones duales multiplicadas) permitió a la red de 225 parámetros igualar el error de entrenamiento de un MLP de 4,300 parámetros.
2.  **Generalización Superior**: En la tarea `prod`, la neurona polimórfica V133 obtuvo un MSE de test de **0.23** frente al **2.00** del MLP Medium. Esto confirma que el bias inductivo (tener un multiplicador físico) permite a la red entender la operación fuera del rango de entrenamiento.
3.  **El reto de la División**: Aunque añadimos un canal `DIV`, la división sigue siendo extremadamente inestable debido a las asíntotas. El MLP sigue siendo superior en `div` por su capacidad de aproximar la hipérbola con tramos lineales seguros.
4.  **Estabilidad**: El uso de `torch.nan_to_num` y `clamp` fue crítico para evitar que el canal de división arruinara el entrenamiento de las otras bases.

## Métricas de Sistema
- **Efficiency**: 64,500 eval/sec (Ligeramente más lento que V132 por la complejidad de la capa).
- **Hardware**: CPU AMD Ryzen 7 8845HS.

## Siguiente Paso (V134)
Integrar un **Cerebelo Espectral** (Walsh/DCT) como una de las bases de la neurona polimórfica para manejar funciones de alta frecuencia y ruido sin necesidad de capas densas profundas.
