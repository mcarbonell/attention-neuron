# Findings V51: Matchstick Neurons (Line Segments)

## Resumen del Experimento
En esta iteración (v51), simplificamos las neuronas de trazo (Stroke Neurons v50) eliminando la curvatura de Bézier y utilizando únicamente **segmentos de línea recta** definidos por dos puntos $(p_0, p_1)$. El objetivo era determinar si la complejidad geométrica de las curvas era necesaria para alcanzar altas precisiones en MNIST.

## Resultados (Métricas Estables - Optimizado con LR más bajo)
- **Best Accuracy**: **98.30%** (Epoch 9)
- **Final Accuracy**: 98.17%
- **Total Evaluations**: 2350 batches (10 epochs)
- **Wall Clock Time**: ~128s
- **Params per Neuron**: 6 (4 coordenadas + 2 anchos de trazo)

## Análisis Comparativo (Actualizado)
| Versión | Geometría | Params/Neurona | Best Acc (MNIST) | Notas |
| :--- | :--- | :--- | :--- | :--- |
| v50b | Bézier Cuadrática | 8 | 97.88% | |
| v51 (v1) | Línea Recta | 6 | 97.78% | LR 0.005 |
| **v51 (v2)** | **Línea Recta** | **6** | **98.30%** | **LR más bajo - ÓPTIMO** |

### Observaciones Clave:
1.  **Sensibilidad al LR**: Los parámetros de coordenadas (x, y) son extremadamente sensibles. Un LR más bajo ha permitido una sintonización fina de la posición y orientación de las "cerillas", desbloqueando un rendimiento significativamente superior.
2.  **Hito de Precisión**: Superar el 98% con solo 6 parámetros por neurona es un resultado sobresaliente. Indica que la estructura lineal es el "lenguaje" natural de MNIST.
3.  **Eficiencia**: Mantenemos la compresión de ~130x respecto a una capa densa, pero ahora con una precisión competitiva con modelos mucho más pesados.

## Conclusión
Las **Matchstick Neurons** demuestran que la esencia de la visión artificial temprana (tipo Gabor o células simples de V1) puede ser capturada de forma extremadamente eficiente mediante geometría aprendible. La línea recta es el "átomo" de la forma para MNIST.

## Próximos Pasos
- ¿Qué pasa si permitimos **múltiples cerillas** por neurona? (p.ej. una "X" o una "L" definida por 2 líneas).
- Probar esta misma arquitectura en **CIFAR-10** para ver si la simplicidad de las líneas se mantiene ante texturas y formas más complejas.
- Evaluar la robustez ante rotaciones y traslaciones.
