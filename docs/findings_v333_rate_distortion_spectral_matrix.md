# Findings v333 — Curva rate–distortion de matrices espectrales

> **Estatus:** Nivel 2, cinco semillas, validación separada para ridge y 8,192 ejemplos de test por semilla. [SEÑAL] en matrices sintéticas: una expansión espectral aumenta de capacidad comprimida a capacidad densa completa al incrementar $K$; no prueba todavía ahorro de cómputo en kernels reales ni transferencia a lenguaje.

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

v332 comparaba una diagonal espectral de 64 parámetros con una matriz densa de 4,096 parámetros. Ese diseño mostraba eficiencia cuando el teacher coincidía con la base, pero no respondía a la objeción de que una expansión espectral con más coeficientes puede recuperar cualquier matriz densa.

v333 sustituye esa dicotomía por $W=U^\top C V$, con $K\in\{16,64,256,1024,4096\}$ coeficientes activos de $C$. A $K=d^2=4096$, la expansión recupera la capacidad completa de una matriz de 64×64. Por tanto, la limitación observada en v332 era de presupuesto $K$, no una imposibilidad intrínseca de la representación espectral expandida.

## 1. Protocolo

- **Teachers:** `dct2d_decay` (energía power-law en DCT-2D), `random2d_decay` (misma energía en una base ortogonal aleatoria) y `dense_unstructured` (matriz plena aleatoria).
- **Alumnos:** `dct2d_topk`, `random2d_topk` y `dense_linear`.
- **Datos:** entradas gaussianas de dimensión 64, ruido de salida 0.05, tamaños de entrenamiento `{16,32,64,128,256}`, cinco semillas y test fijo de 8,192 ejemplos por semilla.
- **Ajuste:** ridge en float64, seleccionado mediante validación entre `{1e-5,1e-3,1e-1}` sin consultar test.
- **Medidas:** MSE de test, $K$, parámetros efectivos, bits a 4/8/16 y tiempo de forward materializado.

## 2. Curva de capacidad a n=256

| Teacher / alumno | K=16 | K=64 | K=256 | K=1024 | K=4096 / Dense |
| :--- | ---: | ---: | ---: | ---: | ---: |
| DCT-2D / DCT top-K | 0.01390 | 0.00855 | 0.00549 | 0.00380 | **0.003316** |
| DCT-2D / Random top-K | 0.04968 | 0.04933 | 0.04772 | 0.03951 | 0.003316 |
| Random-2D / Random top-K | 0.01380 | 0.00849 | 0.00544 | 0.00379 | **0.003328** |
| Random-2D / DCT top-K | 0.04926 | 0.04894 | 0.04718 | 0.03914 | 0.003328 |
| Denso no estructurado / DCT top-K | 0.98898 | 0.98018 | 0.94297 | 0.77934 | 0.003321 |
| Denso no estructurado / Dense | — | — | — | — | **0.003321** |

En los dos teachers compresibles, la base coincidente domina a bajo K y el control aleatorio invierte exactamente el efecto. En el teacher denso no estructurado, ninguna base fija comprime de forma útil: necesita $K=4096$ para igualar Dense. A $K=4096$, DCT top-K, Random top-K y Dense coinciden por construcción en la misma MSE, verificando la recuperación de capacidad completa.

## 3. Rate–distortion bajo cuantización de 4 bits

En teacher DCT-2D a n=256:

| K | Bits efectivos (4-bit + escala) | MSE sin cuantizar | MSE cuantizada |
| ---: | ---: | ---: | ---: |
| 16 | 96 | 0.01390 | 0.01395 |
| 64 | 288 | 0.00855 | 0.00890 |
| 256 | 1,056 | 0.00549 | **0.00703** |
| 1,024 | 4,128 | 0.00380 | 0.00951 |
| 4,096 | 16,416 | 0.00332 | 0.01843 |

La cuantización uniforme global introduce una curva no monótona: añadir coeficientes pequeños incrementa la distorsión por rango dinámico compartido. El mejor compromiso observado de esta tabla es K=256, no el extremo de máxima capacidad. Esto es una [SEÑAL] para evaluar cuantización por bandas, escalas por bloque o codificación de coeficientes significativos.

## 4. Conclusión

La afirmación defendible es una curva, no una oposición binaria: una expansión espectral tiene un control gradual de rate–distortion. Con una base alineada, pocos coeficientes alcanzan una fracción alta de la calidad; con un target no compresible, el presupuesto debe subir hacia $d^2$ y desaparece la ventaja de parámetros. El extremo $K=d^2$ verifica que la expansión puede igualar una matriz densa, a costa de perder compresión.

## 5. Auditoría y amenazas a la validez

1. **Teacher favorable y soporte conocido:** el orden top-K está construido para la distribución generadora. La simetría DCT/Random controla favoritismo de base, pero no demuestra que un modelo aprenda o detecte ese soporte en datos reales.
2. **Coste no algorítmico:** el forward materializa $U^\top C V$; los tiempos no representan una implementación sparse, FFT/FHT o kernel compilado. V333 mide capacidad/bits, no speedup real.
3. **Cuantización simple:** un único min–max por matriz y 32 bits de escala no es un códec óptimo. La degradación a K alto puede reducirse con escalas por banda/bloque, cuantización no uniforme o entropy coding.
4. **Regresión lineal sintética:** no extrapolar directamente a FFNs, atención, BPE o LLMs. V330/v331 continúan siendo el control negativo relevante en Tiny Shakespeare.
5. **Selección de ridge:** se valida en una rejilla pequeña; un baseline denso con tuning más amplio podría desplazar partes de la curva de muestra, aunque no la identidad exacta a K=4096.

## 6. Artefactos

- Script: `scratch/prototype_v333_rate_distortion_spectral_matrix.py`
- Plan: `docs/experiment_plan_v333_rate_distortion_spectral_matrix.md`
- JSON Nivel 2: `results/raw/v333_rate_distortion_spectral_matrix_20260811T150432Z.json`
