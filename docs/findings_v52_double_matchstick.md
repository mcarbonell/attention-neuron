# Findings V52: Double Matchstick Neurons

## Resumen del Experimento
En la versión v52, incrementamos la capacidad de representación de cada neurona permitiendo **dos segmentos de línea recta** por unidad. Esto permite que una sola neurona actúe como un detector de estructuras compuestas (esquinas, cruces, trazos paralelos) mediante una operación de distancia mínima (OR espacial).

## Resultados (Métricas Estables)
- **Best Accuracy**: **97.52%** (Epoch 6)
- **Final Accuracy**: 97.28%
- **Params per Neuron**: 10 (8 coordenadas + 2 $\sigma$)
- **Wall Clock Time**: ~158s (un incremento de ~23% respecto a v51)

## Análisis Comparativo
| Versión | Geometría | Params/Neurona | Best Acc (MNIST) | Notas |
| :--- | :--- | :--- | :--- | :--- |
| v51 | 1 Línea | 6 | **97.78%** | Más simple y efectiva. |
| **v52** | **2 Líneas** | **10** | **97.52%** | Mayor capacidad, menor precisión. |

### Hallazgos Clave:
1.  **Ley de Rendimientos Decrecientes**: A pesar de tener más parámetros y capacidad geométrica, la precisión máxima bajó ligeramente (de 97.78% a 97.52%). Esto sugiere que para MNIST, es más eficiente tener **muchos detectores simples** que detectores complejos.
2.  **Dificultad de Optimización**: Con 2 líneas, el paisaje de pérdida es probablemente más complejo. Es posible que las dos líneas de una misma neurona terminen colapsando en la misma posición o compitiendo de forma redundante si no hay una presión para que se separen.
3.  **Overhead Computacional**: El tiempo de ejecución subió de 128s a 158s debido al cálculo de distancias para el doble de puntos muestreados.
4.  **Interpretación**: Al ver la galería (v52_double_matchsticks_gallery.png), es probable que muchas neuronas no estén aprovechando la segunda línea para formar estructuras coherentes, sino que actúan como ruido o redundancia.

## Conclusión
La simplicidad de la **línea única (v51)** sigue siendo la reina para MNIST. La "neurona cerilla" simple es un prior más fuerte y fácil de optimizar que la versión doble.

## Próximos Pasos
- Volver a la línea única pero explorar la **especialización de escala**: neuronas con líneas muy cortas (detalles) vs líneas largas (estructura global).
- Probar la robustez de v51 contra ruido o ataques adversarios simples comparado con una MLP estándar.
- Intentar **CIFAR-10** con la arquitectura v51; quizás allí la estructura sea más necesaria.
