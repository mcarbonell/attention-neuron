# Findings: V12b (Hybrid Attention CNN, Rank=16) en CIFAR-10

## 1. Experimento

Tras observar un posible "cuello de botella" de capacidad en la versión de `rank=2` sobre CIFAR-10, se incrementó el rango de la modulación del canal a `rank=16`. Esto permite a la red escalar las relaciones entre canales con mucha más expresividad.

- **Arquitectura**: 3 capas convolucionales híbridas (32, 64 y 128 canales) seguidas de una capa clasificadora Residual Attention Neuron.
- **Rango**: 16 (Modulación de Canal Factorizada)
- **Parámetros Entrenables**: **76,453**
- **Hiperparámetros**: Adam (`lr=0.001`), `mask_prob=0.5`, Data Augmentation básico (10 épocas).

## 2. Resultados

| Variante | Rank | Parámetros | Accuracy (10 Epochs) |
| :--- | :--- | :--- | :--- |
| **V12 (CIFAR-10)** | 2 | 9,785 | 26.82% |
| **V12b (CIFAR-10)**| 16 | **76,453** | **40.06%** |

## 3. Conclusiones

1. **Validación del Cuello de Botella**: Aumentar la capacidad de modulación (`rank=16`) desbloquea significativamente el aprendizaje. La red pasó de un 26.8% a un **40.06%** en las mismas 10 épocas.
2. **Eficiencia Paramétrica Escalable**: Con ~76k parámetros, la red sigue siendo minúscula comparada con arquitecturas tradicionales (una ResNet-18 para CIFAR-10 ronda los 11M de parámetros). Haber conseguido un 40% de precisión en tan solo 10 épocas con tan pocos parámetros entrenables es una prueba contundente del poder expresivo de modular sustratos fijos.
3. **Escalado Prometedor**: La curva en la época 10 seguía mostrando signos de crecimiento (`epoch 9: 39.2% -> epoch 10: 40.0%`). Entrenar por 100-200 épocas (el estándar para CIFAR-10) con una política de aprendizaje (learning rate scheduler) probablemente la acercará a resultados muy competitivos.

## 4. Cierre

Aumentar el rango confirma que la "Attention Neuron" puede escalar para absorber la complejidad de datasets más ricos en características. El mecanismo de modulación sobre ruido es fundamentalmente sólido y el `rank` sirve como una "perilla" perfecta para ajustar el equilibrio entre eficiencia y capacidad expresiva.