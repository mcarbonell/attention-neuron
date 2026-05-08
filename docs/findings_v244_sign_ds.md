# Hallazgos V244: Sign-DS (The Memory Ghost)

## Objetivo
Evaluar el límite inferior de memoria del optimizador eliminando el buffer de momentum y utilizando únicamente el **Signo del Gradiente** con el mecanismo de **Directional Stability (DS-EMA)** para estabilizar la convergencia.

## Resultados (MNIST - 10 Épocas)

| Métrica | Adam Estándar | Lion-DS (V243) | **Sign-DS (Ours)** |
| :--- | :--- | :--- | :--- |
| **Memoria (Estados)** | 8 bytes/p | 5 bytes/p | **2 bytes/p** |
| **Precisión Final** | **99.45%** | 99.30% | 98.18% |
| **Ahorro Memoria vs Lion** | - | Base | **+60%** |
| **Ahorro Memoria vs Adam** | Base | +37.5% | **+75%** |
| **Wall Clock Time** | 131.4s | 152.6s | 140.0s |

## Análisis de Eficiencia (PEI)
*Total Parámetros: ~535k (log10 ≈ 5.73)*

- **PEI Adam**: 0.1735
- **PEI Lion-DS**: 0.1733
- **PEI Sign-DS**: 0.1713

## Hallazgos Clave
1.  **Viabilidad del Sign-SGD Puro**: El modelo converge sin momentum, lo que valida que DS-EMA es capaz de amortiguar el ruido del gradiente por sí solo. Sin embargo, la pérdida de información del momentum se traduce en una caída de ~1.1% en accuracy.
2.  **Memoria Ultra-Eficiente**: Con solo 2 bytes por parámetro (1 byte para estabilidad, 1 byte para signo previo), Sign-DS es el optimizador más ligero probado hasta la fecha.
3.  **Velocidad de Convergencia**: Sign-DS es notablemente más lento para arrancar (87% vs 89% en Ep0) y le cuesta alcanzar la precisión "fine-grained" al final del entrenamiento.
4.  **Confiabilidad**: A pesar de la falta de momentum, el entrenamiento fue estable y no hubo divergencias, lo que demuestra la robustez del modulador de ganancia basado en DS.

## Conclusión
Sign-DS es una opción extrema para sistemas donde la **RAM es el cuello de botella absoluto** (ej: microcontroladores o modelos LLM masivos en hardware limitado). Para uso general, Lion-DS sigue ofreciendo el mejor balance entre ahorro y precisión.

### Próximos Pasos
- Probar un híbrido: Sign-DS con un momentum comprimido en 1 byte (Sign-Momentum).
- Evaluar en redes espectrales donde la redundancia de parámetros podría mitigar la caída de precisión del 1%.
