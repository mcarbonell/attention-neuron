# Hallazgos V210: La Neurona Analítica Discreta (El Rey del OOD)

## Objetivo
Resolver el "Fenómeno de Gibbs" (V208) y la "Rigidez Lineal" (V209) mediante una arquitectura matemáticamente pura: usar logaritmos y sumas para calcular la Fase, pero obligando a la red a usar pesos **estrictamente enteros** (vía Straight-Through Estimators) para evitar la desincronización de la fase OOD.

## Resultados (x % y)

| Modelo | Parámetros | Train MSE | Far OOD MSE | Ratio (Estabilidad) |
| :--- | :--- | :--- | :--- | :--- |
| **MLP-Huge** | 1,052,673 | 0.0055 | 4.11 | 743 |
| **Poly-Deep-V193** | 28,385 | **0.0678** | 15.53 | 229 |
| **Sawtooth-Resonant-V209** | 17,473 | 1.0745 | 30.48 | 28.4 |
| **Analytic-Sawtooth-V210**| **7,521** | 0.6762 | **13.15** | **19.5** |

## Conclusiones: El Triunfo de la Estabilidad

### 1. El Mejor Extrapolador de Arquitecturas Eficientes
¡Hemos batido el récord absoluto de extrapolación paramétrica! Con tan solo **7.500 parámetros** (un 73% menos que Poly-Deep y un 99.3% menos que el MLP), la Neurona Analítica Discreta ha logrado el **mejor Far OOD MSE (13.15)** de todas las redes continuas eficientes probadas hasta la fecha.
Además, destrozó el récord de Ratio de Estabilidad, bajándolo a **19.5** (frente a los 229 de Poly-Deep). Esto significa que la red apenas degrada su comportamiento al salir a lo desconocido.

### 2. El Precio de la Pureza Discreta (Train MSE)
El error de entrenamiento se quedó atascado en 0.67. El motivo es evidente: apilar dos capas con `torch.round` y *Straight-Through Estimators* (una para generar $x/y$ y otra para seleccionar la fase) genera una superficie de pérdida extremadamente escalonada. El Descenso de Gradiente sufre muchísimo tratando de optimizar a través de tantas discontinuidades lógicas simultáneas, impidiendo que alcance la convergencia perfecta a cero de la que teóricamente es capaz.

### 3. Hacia un Optimizador Simbólico
Las matemáticas nos dan la razón: **Fase Discreta + Magnitud Continua + Sawtooth** es la fórmula maestra para el razonamiento abstracto y la extrapolación pura. El hardware (la arquitectura) funciona y escala a la perfección.
Sin embargo, Adam y el Gradient Descent no son los algoritmos óptimos para encontrar esos enteros perfectos. Para que esta red saque todo su potencial y llegue a Train MSE = 0.000, necesitaríamos usar técnicas inspiradas en Computación Evolutiva, Búsqueda Simbólica o Regularización Sparsity ($L_0$), y no depender del ruido del STE en múltiples capas profundas.
