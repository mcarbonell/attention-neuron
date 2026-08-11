# Findings v332b — Auditoría de ridge e interpolación

> **Estatus:** Nivel 2, cinco semillas, cálculo float64. [SEÑAL] de un pico de interpolación/condicionamiento en la regresión densa; no es una nueva comparación de arquitecturas.

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

v332 mostró una pérdida densa anómalamente alta en `n=64=d_model`. v332b repite localmente con float64, $n\in\{48,64,80,128\}$ y ridge $\{10^{-5},10^{-4},10^{-3},10^{-2}\}$. El pico persiste, de modo que no puede atribuirse sólo a precisión float32; pero su magnitud depende de ridge y no debe tratarse como rendimiento típico de Dense.

## 1. Resultado

Para `dense_full`, la MSE densa con ridge $10^{-4}$ es 0.2583 a n=48, 1.1761 a n=64, 0.0133 a n=80 y 0.00488 a n=128. En n=64, elevar ridge a $10^{-2}$ reduce MSE a 0.1183 y el número de condición medio de 596,363 a 16,412. Con ridge $10^{-5}$, la MSE es 8.1201 y el SE 6.4105.

Las diagonales coincidentes se mantienen en torno a 0.00252–0.00256 en todo el barrido y todos los ridges. El patrón es consistente con un pico de interpolación y sensibilidad al condicionamiento cerca de $n=d$, no con un fallo general del harness.

## 2. Auditoría y amenazas a la validez

1. El ridge no fue ajustado individualmente por teacher/n; v332b es una auditoría de sensibilidad, no una competición con baseline denso óptimamente afinado.
2. El número de condición describe el sistema ridge, pero no separa por completo double descent estadístico de decisiones de prior/regularización.
3. Sigue siendo una tarea lineal gaussiana sintética; no extrapolar a entrenamiento de Transformers.

## 3. Conclusión

La evidencia de eficiencia de la base coincidente de v332 no depende del artefacto de n=64. Para futuras comparaciones densas de baja muestra debe reportarse una curva de ridge o seleccionar regularización en validación, en lugar de citar un único punto de interpolación.

## 4. Artefactos

- Script: `scratch/prototype_v332b_ridge_interpolation_audit.py`
- Plan: `docs/experiment_plan_v332b_ridge_interpolation_audit.md`
- JSON Nivel 2: `results/raw/v332b_ridge_interpolation_audit_20260811T141653Z.json`
