# Plan v332 — Eficiencia muestral y longitud de descripción de modulaciones espectrales

> **Estado:** pre-registrado; no ejecutado. Nivel 1: una semilla para comprobar el harness. Nivel 2: cinco semillas. Este experimento no afirma igualdad de capacidad universal entre clases espectrales y densas.

## Reconciliación con v330/v331

v331 no encontró una ventaja de FWHT/DCT sobre bases aleatorias en Tiny Shakespeare al igualar la arquitectura Lerp. Por tanto, v332 no vuelve a preguntar si una base fija gana universalmente en lenguaje. Prueba una hipótesis más estrecha y compatible con teoría de la información: una parametrización restringida puede requerir menos muestras y menos bits cuando la función objetivo pertenece a su subclase, aunque un modelo denso también pueda representarla.

## Pregunta

Para una transformación lineal objetivo que es representable tanto por un alumno denso como por una modulación diagonal en su base correcta, ¿la modulación correcta alcanza menor MSE con menos ejemplos y con menor longitud de descripción cuantizada? ¿El efecto cambia de base al rotar el teacher?

## Diseño

Entradas $x\sim\mathcal N(0,I_{64})$ y salidas $y=W_*x+\epsilon$, con ruido gaussiano fijo. Se usan tres teachers:

| Teacher | $W_*$ | Alumno espectral coincidente |
| :--- | :--- | :--- |
| `dct_sparse` | $U_{DCT}^T\operatorname{diag}(a)U_{DCT}$, con 8 coeficientes activos | `dct_diagonal` |
| `random_sparse` | $Q^T\operatorname{diag}(a)Q$, con la misma esparsidad | `random_diagonal` |
| `dense_full` | Matriz densa aleatoria de rango completo | Ninguno; control de frontera |

Para cada teacher, tamaño de muestra `n ∈ {4, 8, 16, 32, 64, 128}` y semilla, se ajustan por mínimos cuadrados con ridge fijo:

1. `dense_linear`: matriz libre de 64×64 (4,096 parámetros).
2. `dct_diagonal`: 64 ganancias diagonales en DCT-II fija.
3. `random_diagonal`: 64 ganancias diagonales en la misma matriz aleatoria Q usada por `random_sparse`.

No hay optimizador iterativo: la solución cerrada elimina LR, épocas y dinámica de entrenamiento como confusores. Se registra por condición el ajuste, MSE train/test, coste y cuantización post-hoc a 4/8/16 bits. El test tiene 8,192 ejemplos nuevos y fijos por semilla.

## Predicciones falsables

| Resultado | Lectura permitida |
| :--- | :--- |
| DCT diagonal aventaja a Random diagonal y Dense en `dct_sparse` con n pequeño | Evidencia de eficiencia muestral condicionada a coincidencia teacher–base. |
| Random diagonal aventaja a DCT y Dense en `random_sparse` | Control de simetría: no es una propiedad universal de DCT. |
| Dense mejora al aumentar n y es mejor en `dense_full` | Verifica que la restricción espectral no tiene capacidad universal equivalente. |
| La variante coincidente mantiene MSE tras cuantización con menos bits totales | Evidencia descriptiva/MDL, no una prueba de VC-dimension. |

El análisis primario compara MSE test emparejado por semilla. La evidencia requiere $|\Delta|\ge2\times SE$ en Nivel 2. Se reporta la curva completa MSE frente a n y MSE frente a bits, nunca sólo el mejor punto.

## Límites explícitos

La clase densa es un superconjunto de las clases diagonales espectrales; sólo hay igualdad de representación **para los teachers DCT/Random dispersos concretos** entre Dense y el alumno coincidente. Reducir parámetros o bits no reduce VC-dimension de una clase funcional idéntica: mide complejidad descriptiva bajo este prior y distribución de tareas.

## Ejecución

```powershell
python scratch/prototype_v332_spectral_sample_efficiency.py --mode pilot
python scratch/prototype_v332_spectral_sample_efficiency.py --mode level2
```

El script debe guardar JSON con teachers, semillas, matrices/huellas, soluciones, curvas y comparaciones emparejadas; el ledger sólo se añade tras éxito.
