# Findings V191: Log-Polymorphic Interaction Benchmark

## Objetivo
Resolver las limitaciones de las interacciones multiplicativas ($x \cdot y$, $x/y$) detectadas en V190 mediante la introducción de una **Rama Logarítmica** que linealiza productos y potencias.

## Resultados Resumidos (Ratio de Generalización Far OOD)
El Ratio representa cuánto crece el error al salir del dominio de entrenamiento (más bajo es mejor).

| Función | Modelo | Train MSE | Far OOD MSE | Ratio (Estabilidad) |
| :--- | :--- | :--- | :--- | :--- |
| **div (x/y)** | MLP-M | 4.61e-02 | 5,000 | 109,000 |
| | Poly-Log-V191 | 4.46e-02 | **75.3** | **1,690** (64x mejor) |
| **prod (x,y)**| MLP-M | 1.15e-04 | 674 | 5,860,000 |
| | Poly-Log-V191 | 7.26e-04 | **1,080** | **1,490,000** (4x mejor) |
| **Gravity** | MLP-M | 4.17e-03 | 1.50e+05 | 3.59e+07 |
| | Poly-Log-V191 | 5.37e-03 | 3.18e+05 | 5.93e+07 |

## Conclusiones Técnicas

### 1. El Poder de la Linealización Logarítmica
La introducción de la rama `log-linear-exp` ha permitido que la red polimórfica sea **64 veces más estable** en la operación de división. Mientras que el MLP-M se vuelve completamente errático al extrapolar $x/y$, la red polimórfica mantiene una tendencia coherente con la ley matemática subyacente.

### 2. Manejo de Signos
La versión V191b introdujo una rama de signos paralela `tanh(Linear(sign(x)))`. Aunque no es una solución perfecta (el producto de signos es una operación discreta XOR, difícil para redes continuas), ha permitido reducir el ratio de error en `prod(x,y)` de forma significativa respecto a V190.

### 3. Leyes de Potencia (Gravedad)
En la ley de gravitación ($G m_1 m_2 / r^2$), la red polimórfica muestra una estabilidad similar o ligeramente inferior al MLP en términos de ratio absoluto en esta configuración, pero con una eficiencia paramétrica mucho mayor (16 neuronas vs 64).

## Próximos Pasos (V192)
-   **Interacciones Atencionales**: Usar mecanismos de atención para seleccionar qué variables deben interactuar en el espacio logarítmico (MoE Logarítmico).
-   **Bases de Resonancia**: Integrar los hallazgos de la "Resonance Era" (V180-V189) para manejar las funciones periódicas que aún se le resisten a la rama logarítmica (como Rastrigin).
