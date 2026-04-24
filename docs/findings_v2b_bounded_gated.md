# Findings: V2b (Bounded Gated Attention Neuron)

## 1. Experimento

Se ha evaluado una variante acotada de la modulación (V2b) para comprobar si restringir el factor de gating multiplicativo mejoraba la regularización y la estabilidad del modelo.

- **Fórmula**: `W_eff = W_init * (1 + alpha * tanh(S)) + A + sin(bias)`
- **Parámetros**: Se fijó `alpha = 1.0`, lo que significa que el factor multiplicativo resultante está estrictamente acotado en el rango continuo `(0, 2)`.

Entrenamiento en MNIST (10 épocas, Adam, `rank=2`, `mask_prob=0.5`).

## 2. Resultados

| Variante | Accuracy (10 Epochs) | Restricción Gating ($M$) |
| :--- | :--- | :--- |
| **V1 (Residual)** | 87.61% | No acotado |
| **V2 (Log-Gated - Exp)**| 87.07% | Estrictamente positivo |
| **V2b (Bounded - Tanh)**| **83.48%** | Acotado a (0, 2) |

## 3. Conclusiones

1. **Pérdida de Capacidad Expresiva**: Acotar fuertemente la magnitud de la modulación (rango 0 a 2) penaliza claramente el rendimiento. La arquitectura parece necesitar la libertad de aplicar factores multiplicativos grandes (o negativos) para reorganizar y adaptar la topología aleatoria inicial del sustrato.
2. **Descarte de la Variante**: Como ocurrió con la versión exponencial (V2), la adición de una función de activación no lineal en la vía multiplicativa no solo reduce la afinidad con implementaciones analógicas (hardware), sino que empíricamente debilita la capacidad de aprendizaje frente a un planteamiento de gating residual puramente lineal.

## 4. Próximos Pasos

Con los experimentos Log-Gated (V2 y V2b) descartados frente a la superioridad de la versión Residual pura (V1), se concluye que el gating debe permanecer libre y lineal. El siguiente paso natural es explorar variantes estructurales como la **V3 (Sparse Attention Neuron)** o dinámicas de entrenamiento como la **V6 (Dual-Speed Attention Neuron)**.