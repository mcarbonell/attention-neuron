# Findings: V1 (Residual Attention Neuron)

## 1. Experimento

Tras el estudio de ablación (V4 vs V5), se ha probado la variante **V1 (Residual Attention Neuron)**, que modifica la formulación base para tratar la modulación como una corrección residual sobre la conectividad aleatoria base.

- **Fórmula**: `W_eff = W_init + W_init * M + A + sin(bias)`
- **Inicialización**: La parte multiplicativa ($M$) se inicializa cercana a 0 (en lugar de 1), haciendo que al principio del entrenamiento `W_eff` sea casi idéntico al sustrato aleatorio `W_init`.

El modelo fue entrenado en MNIST por 10 épocas usando Adam (`rank=2`, `mask_prob=0.5`).

## 2. Resultados

| Variante | Accuracy (10 Epochs) | Comentarios |
| :--- | :--- | :--- |
| **V1 (Residual)** | **91.53%** | Salto de ~5% vs Pure Multiplicative. Curva de aprendizaje muy suave. |
| **V10e (SOTA actual)**| ~88.80% | Modelo original con inicialización unity. |
| **V4 (Multiplicative)**| 86.64% | Sin corrección aditiva. |

## 3. Conclusiones

1. **Estabilidad y Expresividad**: Formular la modulación como un término residual (`W_init + W_init * M`) en lugar de un reemplazo directo (`W_init * M`) permite una optimización mucho más estable. La red parte del "ruido base" puro y suavemente enciende o apaga las conexiones necesarias, evitando shocks de inicialización.
2. **Validación del Camino**: Este enfoque es mucho más limpio conceptualmente para comparar con baselines low-rank convencionales (como LoRA) y demuestra que la suma de "Identidad Base + Gating Estructural Residual + Corrección Aditiva" es una formulación muy robusta.

## 4. Próximos Pasos

El siguiente paso (V2: Log-Gated Attention Neuron) explorará si parametrizar esta modulación mediante funciones acotadas o estrictamente positivas (ej. `M = exp(S)`) puede mejorar el condicionamiento del gradiente y empujar el rendimiento por encima del 88.8%.