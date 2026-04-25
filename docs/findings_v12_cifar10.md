# Findings: V12 (Hybrid Attention CNN) en CIFAR-10

## 1. Experimento

Tras validar el éxito de la arquitectura híbrida convolucional (V12) en MNIST, se procedió a escalar el experimento al dataset **CIFAR-10**, el cual es significativamente más complejo (imágenes a color 32x32, características más complejas, mayor variabilidad).

- **Arquitectura**: 3 capas convolucionales híbridas (32, 64 y 128 canales) seguidas de una capa clasificadora Residual Attention Neuron.
- **Parámetros Entrenables**: **9,785**
- **Hiperparámetros**: Adam (`lr=0.001`), `rank=2`, `mask_prob=0.5`, Data Augmentation básico (RandomCrop, RandomHorizontalFlip).

## 2. Resultados

| Época | Accuracy en Test |
| :--- | :--- |
| 1 | 9.59% |
| 5 | 19.67% |
| 10 | **26.82%** |

## 3. Conclusiones

1. **Prueba de Aprendizaje Positiva**: El modelo está aprendiendo de forma constante y estable. Pasar del azar (~10%) a casi un 27% en tan solo 10 épocas con apenas ~9,700 parámetros entrenables confirma que el gating estructural en convoluciones no se rompe con imágenes a color.
2. **Capacidad vs Complejidad**: CIFAR-10 requiere extraer jerarquías de características mucho más profundas que MNIST. La lentitud en la convergencia sugiere que una arquitectura de tan solo ~9k parámetros (restringida a un `rank=2` en el canal y `1x1` espacial sobre el kernel) podría estar sufriendo un "cuello de botella" de capacidad expresiva ("underfitting").
3. **Próximos Ajustes Sugeridos**: Para llevar esta variante al nivel de arquitecturas de visión tradicionales en CIFAR-10, sería recomendable explorar:
   - Incrementar el **rango (rank)** de la modulación de canal (ej. de 2 a 8 o 16).
   - Modificar la velocidad de aprendizaje (`lr=0.01` inicial con scheduler).
   - Entrenar por **más épocas** (las CNNs en CIFAR-10 suelen requerir >100 épocas).

## 4. Cierre

Este experimento asienta una base sólida. La red consigue extraer representaciones útiles en un problema complejo modificando exclusivamente los pesos de modulación sobre sustratos convolucionales aleatorios, abriendo el camino para investigar el trade-off exacto entre "rango de modulación", "número de parámetros" y "precisión final" en visión artificial.