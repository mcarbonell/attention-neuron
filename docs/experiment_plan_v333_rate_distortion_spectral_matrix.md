# Plan v333 — Curva rate–distortion–compute de matrices espectrales

> **Estado:** diseño pre-registrado. V333 sustituye la comparación binaria «diagonal espectral frente a densa» por un barrido continuo del número de coeficientes espectrales de una matriz.

## 0. Reconciliación

v332 demuestra eficiencia cuando el teacher es diagonal en una base, pero sólo usa 64 grados de libertad. No responde a la objeción correcta: una matriz puede representarse mediante una expansión espectral cada vez más rica, y recuperar la matriz densa cuando el número de coeficientes alcanza el espacio completo. V333 mide esa transición.

## Pregunta

Para una matriz $W\in\mathbb{R}^{64\times64}$ representada como

$$W=U^\top C V,$$

¿cómo cambian error de test, bits de descripción y coste al permitir $K\in\{16,64,256,1024,4096\}$ coeficientes activos de $C$? ¿La curva depende de que la energía del teacher esté concentrada en DCT-2D, en una base aleatoria o sea densa/no compresible?

`K=4096=d^2` equivale a la capacidad de una matriz completa en esa expansión; no se afirmará compresión en ese extremo.

## Teachers y controles

| Teacher | Coeficientes de $C$ | Prior que debe ganar |
| :--- | :--- | :--- |
| `dct2d_decay` | Energía power-law concentrada en bajas frecuencias DCT-2D | DCT-2D top-K |
| `random2d_decay` | Misma energía, índices/base permutados aleatoriamente | Random-2D top-K |
| `dense_unstructured` | Coeficientes i.i.d. en todo el plano | K alto / Dense |

Alumnos por cada $K$: `dct2d_topk`, `random2d_topk` y `dense_linear`. La base aleatoria usa matrices ortogonales fijas, compartidas con su teacher. Los alumnos conocen el orden de soporte predefinido (bajas frecuencias para DCT, orden de energía equivalente para el control random); no se usa test para elegir K.

## Medición

- Entradas gaussianas, salida lineal con ruido; mismos splits por semilla.
- Barrido de muestras `n={16,32,64,128,256}` y cinco semillas Nivel 2.
- Ridge seleccionado de una pequeña rejilla por validación, igual para todos los alumnos de cada condición.
- Métricas: MSE test, $K$, parámetros, bits a 4/8/16 bits, tiempo de ajuste y latencia de forward tras calentamiento.
- Curvas obligatorias: MSE–K, MSE–bits, MSE–n y latencia–K. Reportar área bajo curva rate–distortion, no sólo el mejor K.

## Criterios falsables

1. En `dct2d_decay`, DCT debe alcanzar una MSE objetivo con un K y bits menores que Random/Dense.
2. En `random2d_decay`, el resultado debe invertirse para Random; de no hacerlo, el harness favorece DCT artificialmente.
3. En `dense_unstructured`, las curvas espectrales deben necesitar K cercano a $d^2$ para igualar Dense. Eso no es un fallo: es el límite de compresibilidad.
4. Si el control random reproduce la curva DCT en el teacher DCT, no hay evidencia de ventaja específica de DCT, sólo de truncación/regularización.

## Alcance y amenazas

Este experimento mide representación lineal de matrices sintéticas, no lenguaje ni un kernel rápido. Un forward `U^T C V` materializado puede costar $O(d^2)$ aunque $C$ sea dispersa; el claim de compute requiere una implementación sparse/FHT separada. Su contribución es establecer la curva de capacidad y bits que un kernel eficiente tendría que explotar.

## Ejecución propuesta

Crear `scratch/prototype_v333_rate_distortion_spectral_matrix.py`. Primero Nivel 1 con una semilla; después Nivel 2. El JSON debe contener cada punto de curva, soporte, hashes de bases, ridge validado, metadatos y diferencias emparejadas.
