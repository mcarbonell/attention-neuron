# Findings: V2 (Log-Gated Attention Neuron)

## 1. Experimento

Se ha evaluado la variante **V2 (Log-Gated Attention Neuron)** utilizando una parametrización exponencial para garantizar que el factor multiplicativo sea siempre estrictamente positivo.

- **Fórmula**: `W_eff = W_init * exp(S) + A + sin(bias)`
- **Inicialización**: El pre-gating $S$ (de bajo rango) se inicializa con valores muy pequeños, de modo que `exp(S)` comience muy cerca de 1, emulando la estabilidad inicial.

Entrenamiento en MNIST (10 épocas, Adam, `rank=2`, `mask_prob=0.5`).

## 2. Resultados

| Variante | Accuracy (10 Epochs) | Tiempo / Época | Comentarios |
| :--- | :--- | :--- | :--- |
| **V1 (Residual)** | 87.61% | ~11.1s | Formulación lineal residual. |
| **V2 (Log-Gated - Exp)** | **87.07%** | ~13.0s | Gating estrictamente positivo. |

## 3. Conclusiones

1. **Rendimiento Equivalente**: Forzar una geometría estrictamente positiva mediante la función exponencial no aporta una mejora tangible en la capacidad de aprendizaje ni en la estabilidad a corto plazo (10 épocas) respecto a aprender la matriz de modulación $M$ directamente.
2. **Penalización Computacional**: La evaluación repetida de la función `exp()` dentro del flujo crítico de cálculo introdujo un "overhead" medible, incrementando el tiempo de época de 11 a 13 segundos. 
3. **Implicaciones Hardware**: Teniendo en cuenta que el objetivo final de la arquitectura es ser "hardware-friendly" (fotónica, analógica), la adición de operaciones no lineales complejas como la exponencial en la ruta de modulación de los pesos resta atractivo a esta variante a menos que ofrezca una mejora sustancial (que no ha sido el caso).

## 4. Siguientes Pasos

La parametrización exponencial pura no es el camino óptimo. Queda pendiente decidir si se explora la versión acotada `1 + alpha * tanh(S)` o si se priorizan variantes estructurales (Sparse, Dual-Speed) manteniendo la modulación lineal simple.