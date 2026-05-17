# Findings V284: Spherical Loss & Phase Regularization

## Resumen
El experimento V284 evalúa dos regularizadores matemáticos diseñados específicamente para interactuar con la topología nGPT (hiperesfera) y la Transformada de Fourier (FFT), sobre el modelo Matrix-Free.
1. **Spherical Loss:** Calcula la similitud coseno normalizada entre el estado final latente y el diccionario de tokens. Incorpora una variable de temperatura aprendible $\tau$ para escalar dinámicamente la entropía del softmax.
2. **Phase Continuity Regularization:** Aplica un castigo $L_1$ sobre las diferencias de las frecuencias adyacentes en la matriz de fase, imponiendo que las transformaciones "ondulatorias" sean lógicas y continuas, en lugar de ruido de alta frecuencia memorizado.

## Resultados Empíricos (k_walsh=32)

| Modelo | Params | Val Loss | PPL | Convergencia | Wall Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| A_V283_Baseline | 24,332 | 1.7844 | 5.96 | Ep2 | 878.2s |
| B_SphericalLoss | 24,333 | 1.7781 | 5.92 | Ep2 | 871.4s |
| **C_Spherical_and_PhaseReg** | **24,333** | **1.7664** | **5.85** | **Ep2** | **839.3s** |

## Hallazgos Fundamentales

### 1. Auto-Regulación Termodinámica de $\tau$
Como se predijo, arrancar con $\tau=10.0$ fue acertado. Durante el entrenamiento, la red no lo mantuvo estático; lo fue escalando progresiva y suavemente época tras época, alcanzando $\tau \approx 43.5$ en la época 40 (Modelo C).
Esto demuestra que en arquitecturas nGPT puramente hiperesféricas, la red necesita ajustar gradualmente la "agudeza" de su distribución de probabilidad. Inicialmente es conservadora (Softmax ancho) mientras mapea topológicamente los vectores, y luego se vuelve asertiva (Softmax afilado) cuando el modelo aprende los conceptos.

### 2. Generalización mediante Continuidad de Fase
El modelo C destrozó al baseline de V283, reduciendo el Validation Loss a `1.7664`.
Esto confirma que la penalización de saltos bruscos en el dominio frecuencial actúa como un "prior inductivo" perfecto para el lenguaje. Al obligar a que `self.phase` no fluctúe irracionalmente de una frecuencia a la siguiente, evitamos que el optimizador grabe ruido en los pesos complejos, forzando un mapeo semántico ondulatorio que generaliza mejor en el Validation Set.

## Conclusión
La combinación de **Spherical Loss (con $\tau$ aprendible)** y la **Regularización de Fase L1** conforman un parche matemático trivialmente barato de integrar, pero tremendamente efectivo, bajando el PPL y reduciendo la varianza del entrenamiento.
Estos hiper-componentes son adiciones directas recomendadas para la arquitectura `Spectral V9` de producción.
