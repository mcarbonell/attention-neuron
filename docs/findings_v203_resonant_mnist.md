# Hallazgos V203: Escalado de Resonancia a MNIST

## Objetivo del Experimento
Tras validar la capacidad no lineal de las Neuronas de Resonancia en el problema XOR (V202), el objetivo del V203 era escalar esta arquitectura al dataset MNIST (clasificación de dígitos 28x28) para probar si la codificación de fase y la interferencia de ondas pueden resolver problemas de alta dimensionalidad.

## Innovación Matemática (FastResonantLayer)
Para evitar el colapso de memoria (OOM) inherente al cálculo de un tensor 3D de diferencias de fase `(Batch, Out, In)`, implementamos una optimización crítica basada en la identidad trigonométrica:
`cos(x - w) = cos(x)cos(w) + sin(x)sin(w)`

Esto permitió calcular la interferencia de fase exacta mediante dos operaciones de producto punto densas (`F.linear`), logrando una velocidad de entrenamiento competitiva (~13 segundos por época en CPU).

## Resultados del Experimento

El modelo se entrenó durante 5 épocas usando una arquitectura de una capa oculta (784 -> 128 -> 10):

```json
{
  "dataset": "MNIST",
  "epochs": 5,
  "accuracy": 0.9622,
  "wall_clock_time": 67.0,
  "params": 203520,
  "optimizer": "Adam(lr=0.005)"
}
```

**Progresión de Accuracy (Test):**
- Época 1: 76.61%
- Época 2: 86.46%
- Época 3: 86.72%
- Época 4: 95.90%
- Época 5: 96.22%

## Conclusión
La arquitectura de Fase es perfectamente viable para visión artificial. **Más de un 96% de precisión en solo 5 épocas** demuestra que el mapeo de la intensidad de un píxel a una Fase (ángulo), seguido de una resonancia armónica constructiva, proporciona gradientes estables y capacidad de generalización.

La red no "suma pesos", sino que "sintoniza frecuencias". Cada clase (dígito) se ha convertido en un patrón de resonancia armónica. Este éxito nos da luz verde para explorar la reducción extrema de parámetros, por ejemplo aplicando Transformadas de Fourier o Walsh a la `FastResonantLayer` para crear una red híbrida.
