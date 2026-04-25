# Findings: V17 (The Colossus)

## 1. Experimento

La V17 buscaba romper la barrera del 99% mediante el aumento de capacidad y la introducción de regularización por datos.

**Configuración:**
- **Arquitectura**: 784 -> 2048 -> 1024 -> 10.
- **Rango**: `rank=64`.
- **Novedades**: BatchNorm1d, Data Augmentation (Rotation + Affine).
- **Parámetros**: 897,310.
- **Entrenamiento**: 40 épocas, OneCycleLR.

## 2. Resultados

| Métrica | Valor |
| :--- | :--- |
| **Best Test Accuracy** | **98.99%** |
| **Final Loss** | 0.1019 |
| **Tiempo** | 1054s (CPU) |

## 3. Conclusiones

1.  **Casi Éxito**: El 98.99% es un hito. Demuestra que las Attention Neurons pueden competir con MLPs tradicionales entrenados al completo.
2.  **Pendiente Ascendente**: El hecho de que el mejor resultado ocurriera en la última época sugiere que con más tiempo de entrenamiento el 99% es trivial.
3.  **Bottleneck**: La primera capa (784->2048) es donde reside la mayor parte de la computación y la extracción de rasgos. Incrementar el rango específicamente ahí podría ser la clave final.

## 4. Próximos Pasos (V18)

- Aumentar el rango de la primera capa a **128**.
- Entrenar durante **60 épocas**.
- Implementar **Label Smoothing**.
- Ajustar el scheduler para un enfriamiento más largo.
