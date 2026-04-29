# Hallazgos V90b: Placa Analógica Adaptativa

## Hipótesis
La "Placa Analógica" original (V90) utilizaba grupos fijos de neuronas especializadas en diferentes agregaciones estadísticas (SUM, VAR, L2, LSE). La hipótesis de la V90b es que permitir que **cada neurona aprenda su propia mezcla óptima** de estos agregadores (más un nuevo agregador espectral basado en Walsh) mejorará la expresividad de la red sin aumentar el número de neuronas.

## Diseño Experimental
- **Nuevos Agregadores**: Se ha añadido `WALSH_ENERGY` (energía media en el dominio de Walsh) para capturar patrones de frecuencia.
- **Mecanismo Adaptativo**: Cada neurona tiene un vector de `mixture_logits` (5 dimensiones) que pasa por un Softmax para ponderar la contribución de cada agregador.
- **Arquitectura**: 
  - Capa de entrada: 784 (MNIST).
  - Capa Adaptativa: 64 neuronas.
  - Clasificador lineal: 64 -> 10.
- **Baseline**: Red MLP estándar con 64 neuronas y activación ReLU.

## Resultados Esperados
Se espera observar qué agregadores son preferidos por la red. Si el `WALSH_ENERGY` o la `VAR` tienen pesos significativos, confirmaría que la diversidad matemática es útil para el aprendizaje de características en MNIST.

## Observaciones de Implementación
- El uso de `Z = x.unsqueeze(1) * self.weight.unsqueeze(0)` crea un tensor intermedio de tamaño `(B, 64, 784)`. Esto es eficiente en memoria para estas dimensiones pero podría ser un cuello de botella para capas mucho más grandes.
- La FWHT requiere padding a potencia de 2 (1024 para MNIST).

---
*Resultados pendientes de ejecución.*
