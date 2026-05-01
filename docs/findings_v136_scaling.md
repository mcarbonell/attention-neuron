# Findings V136: Escalabilidad y Saturación Espectral

## Objetivo
Validar si la reducción de parámetros de las neuronas espectrales (Smooth Walsh) se traduce en una ventaja real de velocidad y memoria al escalar a dimensiones de LLM (8192+).

## Resultados del Benchmark (GPU DirectML @ Dim 8192)

| Modelo | Parámetros | Opt Step (ms) | Total Step (ms) | Ventaja Velocidad |
| :--- | :--- | :--- | :--- | :--- |
| **Dense Baseline** | 8,396,800 | 44.83 ms | 45.81 ms | 1.0x |
| **SmoothWalsh (Cached)** | **532,480** | **15.49 ms** | **17.30 ms** | **2.6x** |

## Hallazgos Clave

1.  **El Muro de Adam**: El experimento confirma que en modelos grandes, el tiempo de ejecución está dominado por el optimizador (**97% del tiempo** en el modelo denso). Al reducir los parámetros, reducimos proporcionalmente los estados de Adam, eliminando el principal cuello de botella.
2.  **Eficiencia de Síntesis**: La versión `SW_Cached` demuestra que si sintetizamos los pesos una vez por paso, el Forward espectral es despreciable frente al ahorro en la actualización de pesos.
3.  **Hardware Friendly**: El modelo espectral evitó los fallbacks de CPU que sufrió el modelo denso en DirectML, demostrando que es más apto para hardware con ancho de banda de memoria limitado.

## Conclusión
La arquitectura espectral no es solo una curiosidad matemática; es una **necesidad computacional** para escalar modelos masivos sin colapsar el hardware bajo el peso de los momentos del optimizador.

## Siguiente Paso (V137)
Integrar esta eficiencia en una arquitectura recurrente o de atención donde podamos aprovechar la **compresión espectral de las llaves (KV Cache)**.
