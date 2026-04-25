# Findings: V16 (Over-Parametrized Attention Neuron)

## 1. Experimento

El objetivo de la V16 fue alcanzar el 99% de accuracy en MNIST utilizando exclusivamente Attention Neurons sobre un sustrato de pesos aleatorios fijos.

**Configuración:**
- **Arquitectura**: 3 capas (784 -> 1024 -> 1024 -> 10).
- **Rango**: `rank=32`.
- **Mecánica**: Modulación dual (multiplicativa + aditiva).
- **Parámetros entrenables**: 319,134 (~17.1% de un MLP equivalente).
- **Optimizador**: AdamW + OneCycleLR (30 épocas).
- **Estabilización**: LayerNorm + Dropout(0.1).

## 2. Resultados

| Métrica | Valor |
| :--- | :--- |
| **Best Test Accuracy** | **98.45%** |
| **Final Training Loss** | 0.0023 |
| **Wall Clock Time** | 413.3s (CPU) |
| **Parámetros** | 319,134 |

## 3. Conclusiones

1.  **Salto de Calidad**: Se ha superado el anterior récord del proyecto (~94.4%) por un margen del 4%. Esto demuestra que el **rango (rank)** y la **profundidad** son los factores determinantes para la capacidad de estas redes.
2.  **Validación del Sustrato**: Lograr un 98.45% con pesos aleatorios fijos (solo el 17% de params son entrenables) demuestra que la modulación de bajo rango es suficiente para "esculpir" funciones de decisión complejas.
3.  **Límite de Generalización**: El bajísimo training loss (0.0023) frente al test accuracy indica que la red tiene capacidad de sobra para memorizar el dataset, pero necesita mejores mecanismos de generalización (o aumento de datos) para alcanzar el 99%.

## 4. Próximos Pasos (V17)

- Incrementar el rango a **64**.
- Arquitectura más ancha en las capas iniciales (**2048**).
- Implementar **Data Augmentation** (rotaciones/traslaciones) para forzar generalización.
- Cambiar LayerNorm por **BatchNorm1d**.
