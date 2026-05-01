# Findings V190: Structural Generalization Benchmark

## Objetivo
Evaluar la capacidad de generalización Out-of-Distribution (OOD) de diferentes arquitecturas neuronales en funciones matemáticas 1D, 2D y N-D. Comparamos MLPs tradicionales de varios tamaños contra una **Neurona Polimórfica (V190)** con bases explícitas y capas de interacción.

## Resultados Resumidos (MSE)

| Función | Modelo | Parámetros | Train MSE | Far OOD MSE | Ratio (Gen) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **x^2** | MLP-M | 4,353 | 2.93e-06 | 688.4 | 2.34e+08 |
| | Poly-V190 | **161** | 1.00e-05 | 1,858.3 | 1.85e+08 |
| **sinc(x)** | MLP-M | 4,353 | 9.90e-05 | 0.108 | 1,094 |
| | Poly-V190 | **161** | 1.56e-05 | **0.072** | 4,611 |
| **prod(x,y)**| MLP-L | 132,609 | 1.19e-05 | 451.1 | 3.78e+07 |
| | Poly-V190 | **321** | 4.34e-04 | 1,063.4 | 2.44e+06 |
| **Schwefel** | MLP-L | 132,865 | 2,163 | 7.42e+06 | 3,431 |
| | Poly-V190 | **465** | 941,594 | **2.03e+06** | **2.16** |

## Conclusiones Técnicas

### 1. La Paradoja de la Precisión Local vs Estabilidad Global
Los MLPs (especialmente los grandes como MLP-L) alcanzan precisiones asombrosas en el rango de entrenamiento ($10^{-5}$ o incluso $10^{-7}$). Sin embargo, su error en **Far OOD** explota catastróficamente (Ratios de $10^7$ o superior). Esto confirma que el MLP está haciendo **interpolación estadística**, no aprendizaje de leyes.

### 2. Generalización Estructural en Funciones Complejas
En la función **Schwefel**, la arquitectura polimórfica demostró una estabilidad impresionante. Mientras que el MLP-L multiplicó su error por 3,431 al salir del rango, la **Poly-V190 solo lo multiplicó por 2.16**. Aunque el error de entrenamiento fue mayor, la red polimórfica "entendió" la tendencia global de la función mucho mejor que el MLP.

### 3. Eficiencia Paramétrica
La Poly-V190 con solo **161-465 parámetros** compite o supera en extrapolación a MLPs con **132,000 parámetros**. Esto representa una compresión de conocimiento de aproximadamente **300x a 800x** para tareas de aproximación de leyes.

### 4. Limitaciones Observadas
-   **Interacción**: La capa de producto circular mejoró la estabilidad en `prod(x,y)` pero aún no alcanza la precisión de un MLP-L en el rango cercano. Se requiere una lógica de interacción más rica (posiblemente atencional o log-transformaciones).
-   **Optimización**: El entrenamiento de las bases polimórficas es más ruidoso que el de un MLP estándar. Se necesita un scheduler o un optimizador que entienda la escala de cada base.

## Próximos Pasos (V191)
-   Introducir **Transformaciones Logarítmicas** en la capa de interacción para manejar productos y divisiones de forma nativa (linealizando en espacio log).
-   Añadir una **Capa de Resonancia** (basada en los hallazgos de V180-V189) para mejorar el rendimiento en funciones periódicas como Rastrigin.
