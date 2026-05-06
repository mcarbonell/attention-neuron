# Hallazgos V240: El Triunfo de la Diferenciabilidad Mixta

## Objetivo
Validar si un Mixture of Experts (MoE) puede beneficiarse de tener expertos con diferentes regímenes de optimización: **Analítico (Adam)** para la forma general y **Simbólico (DGE)** para la lógica discontinua.

## Resultados (Modulus Challenge)

| Métrica | Valor |
| :--- | :--- |
| **Train MSE** | **5.98e-02** |
| **Far OOD MSE** | 10.34 |
| **PEI** | 4.57 |
| **DGE Params Ratio** | 0.13% |
| **Estabilidad** | Alta (Sin oscilaciones salvajes) |

## Análisis Técnico

### 1. Superando la "Trampa de Sísifo"
A diferencia de **V213**, donde Adam intentaba "aprender" el módulo mediante gradientes sintéticos y acababa cayendo por el precipicio de la discontinuidad debido a la inercia, el enfoque híbrido de **V240** es extremadamente estable.
- **Adam** se encargó de la red de Gating y del experto analítico, proporcionando una "base" continua.
- **DGE** optimizó los 6 parámetros del experto simbólico (`v1 % v2`). Al ser solo 6 parámetros, DGE fue extremadamente eficiente y preciso.

### 2. Eficiencia Paramétrica (PEI)
Con solo un **0.13%** de los parámetros optimizados mediante DGE, la red logró una precisión que a un MLP puro le costaría miles de épocas alcanzar. Esto confirma que **no necesitamos que toda la red sea derivable**: solo necesitamos que las partes lógicas tengan el optimizador adecuado.

### 3. Generalización (OOD)
El `Far OOD MSE` sigue siendo el punto débil. Aunque la red aprendió la lógica dentro del rango $[0, 5]$, la extrapolación a $[0, 20]$ muestra que los pesos del experto simbólico (las proyecciones de entrada) no son todavía "identidades perfectas". Sin embargo, la estabilidad del entrenamiento sugiere que con más tiempo o una arquitectura de proyección más simple, la generalización total es posible.

## Conclusión
La arquitectura híbrida es el camino a seguir para sistemas que requieren tanto **percepción (continua)** como **razonamiento (discreto)**. La "minoría DGE" actúa como un ancla lógica que permite a la red derivar soluciones exactas en paisajes matemáticos donde el gradiente tradicional es inútil o engañoso.
