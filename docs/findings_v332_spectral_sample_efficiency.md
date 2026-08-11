# Findings v332 — Eficiencia muestral y longitud de descripción espectral

> **Estatus:** Nivel 2, cinco semillas y test retenido de 8,192 ejemplos por semilla. [SEÑAL] limitada a teachers sintéticos construidos en una base conocida; no constituye evidencia de ventaja universal en lenguaje.

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

v330/v331 no encontraron ventaja verificable de bases espectrales frente a controles aleatorios en Tiny Shakespeare. v332 no contradice ese resultado: cambia deliberadamente la pregunta. Construye teachers que son diagonales y dispersos en DCT o en una base ortogonal aleatoria, y pregunta por eficiencia muestral/descriptiva cuando el sesgo inductivo del alumno coincide con la estructura generadora.

## 1. Protocolo

Entradas gaussianas de dimensión 64 y $y=W_*x+\epsilon$, con ruido de desviación 0.05. Se comparan `dense_linear` (4,096 parámetros) y diagonales DCT/aleatoria (64 parámetros), ajustadas por ridge cerrado. La clase densa puede representar todos los teachers; la diagonal coincidente representa exactamente los teachers dispersos de su base. Barrido $n\in\{4,8,16,32,64,128\}$, cinco semillas, test fijo de 8,192 ejemplos y cuantización post-hoc a 4/8/16 bits.

## 2. Resultados principales a n=128

| Teacher | Denso | DCT diagonal | Aleatoria diagonal | Lectura |
| :--- | ---: | ---: | ---: | :--- |
| DCT disperso | 0.00510 | **0.00252** | 0.07931 | La base DCT coincidente llega al suelo de ruido. |
| Aleatorio disperso | 0.00508 | 0.07909 | **0.00252** | Control simétrico: la ventaja sigue a la base del teacher. |
| Denso completo | **0.00507** | 1.00450 | 1.00488 | La restricción diagonal no tiene capacidad universal equivalente. |

Para el teacher DCT, DCT−aleatoria a n=128 es `-0.07679` con `SE=0.000277` (`2×SE=0.000554`). La misma relación se invierte para el teacher aleatorio. La curva completa muestra el efecto desde n=4: DCT coincidente logra 0.00361 frente a 0.08536 denso y 0.12202 aleatoria.

## 3. Longitud de descripción

En el teacher DCT a n=128, DCT diagonal cuantizada a 4 bits usa 288 bits incluyendo escala y obtiene MSE 0.00267. Denso a 16 bits usa 65,568 bits y obtiene 0.00510. Esto es evidencia de descripción compacta bajo este prior, no de una menor VC-dimension de una clase funcional idéntica.

## 4. Auditoría y amenazas a la validez

1. **Teacher favorable por construcción:** el teacher DCT está exactamente en la subclase DCT; el control aleatorio y el teacher denso verifican la simetría y frontera, pero no hacen el resultado transferible a lenguaje natural.
2. **Anomalía densa en n=64:** la curva densa es no monótona cerca de $n=d$. v332b la audita con float64 y ridge; no se usa ese punto aislado para afirmar inferioridad densa.
3. **MDL aproximado:** los bits son cuantización uniforme y no código universal/aritmético completo. Miden una longitud de descripción operacional, no el mínimo teórico.
4. **Linealidad:** no evalúa modulación trigonométrica, atención ni representación no lineal. Es un test de principio de base y muestras.

## 5. Conclusión

v332 respalda la afirmación limitada: una parametrización diagonal en la base que genera el target puede requerir muchas menos muestras y bits que una matriz densa que también puede representarlo. No respalda que DCT/FWHT mejoren universalmente una FFN densa; v330/v331 siguen delimitando negativamente esa afirmación en Tiny Shakespeare.

## 6. Artefactos

- Script: `scratch/prototype_v332_spectral_sample_efficiency.py`
- Plan: `docs/experiment_plan_v332_spectral_sample_efficiency.md`
- JSON Nivel 2: `results/raw/v332_spectral_sample_efficiency_20260811T141111Z.json`
