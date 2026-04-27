# Findings V54: RGB Matchstick Neurons (CIFAR-10)

## Resumen del Experimento
Tras el éxito en MNIST, trasladamos las **Matchstick Neurons** a CIFAR-10. Cada neurona detecta un segmento de línea recta pero ahora con **sensibilidad al color (RGB)** mediante un vector de pesos aprendible.

## Resultados (Sondeo Rápido)
- **Dataset**: CIFAR-10 (32x32, RGB)
- **Best Accuracy**: **61.18%** (Epoch 9)
- **Final Accuracy**: 60.79%
- **Epochs**: 10
- **Num Neurons**: 512
- **Params per Neuron**: 6 (Geometría) + 3 (Color) = 9 parámetros.

## Análisis
| Modelo | Dataset | Best Acc | Notas |
| :--- | :--- | :--- | :--- |
| Matchsticks v51 | MNIST | 98.30% | El "átomo" de la forma. |
| **Matchsticks v54** | **CIFAR-10** | **61.18%** | **Señal fuerte en imágenes naturales.** |

### Observaciones Clave:
1.  **Arranque Explosivo**: El paso de azar a >52% en la primera época indica que la red localiza instantáneamente bordes y contrastes de color útiles.
2.  **Representación Visual**: A diferencia de una capa densa que vería píxeles aislados, esta capa ve "trazos de color". Para CIFAR-10, esto actúa como un banco de filtros Gabor aprendibles pero mucho más parsimoniosos.
3.  **Color + Geometría**: La combinación de 2 puntos y 3 pesos RGB es suficiente para categorizar objetos básicos con una precisión muy superior a una MLP de tamaño equivalente.

## Conclusión
Las neuronas geométricas escalan a datasets más complejos. El 61% es una base sólida para explorar arquitecturas más profundas o multiescala.

## Próximos Pasos
- **Arquitectura Profunda**: ¿Qué pasa con 2 o 3 capas de Matchsticks?
- **Global Average Pooling**: En lugar de una cabeza MLP pesada, usar una estructura más convolucional.
- **Visualización**: Ver qué colores y formas están detectando (ej: ¿líneas azules horizontales para el cielo/mar?).
