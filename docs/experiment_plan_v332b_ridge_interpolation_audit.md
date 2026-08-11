# Plan v332b — Auditoría de ridge e interpolación en v332

## Reconciliación

v332 mostró la eficiencia esperada de la base coincidente, pero `dense_linear` presenta un pico no monótono cerca de `n=64=d_model`. Puede ser double descent de interpolación, sensibilidad al ridge `1e-4` o condicionamiento numérico. v332b no cambia el teacher ni reclama un nuevo ganador: audita esa anomalía.

## Diseño

- Mismos teachers, alumnos, semillas y test de v332.
- Tamaños locales: `n={48,64,80,128}`.
- Ridge: `{1e-5, 1e-4, 1e-3, 1e-2}`.
- Solución cerrada en float64 para el ajuste y predicción; se registra también el número de condición de `XᵀX + λI`.
- Comparación primaria: curva de `dense_linear` frente a n/ridge para cada teacher. Las diagonales coincidentes son controles de estabilidad.

## Predicciones

Un pico que disminuye suavemente al incrementar ridge y se concentra en `n≈64` es consistente con doble descent/condicionamiento estadístico. Un comportamiento errático que persiste lejos de `n=64` y para ridge grandes requiere auditar la implementación antes de interpretar la curva.

## Ejecución

```powershell
python scratch/prototype_v332b_ridge_interpolation_audit.py --mode pilot
python scratch/prototype_v332b_ridge_interpolation_audit.py --mode level2
```
