# Findings V194: Modulus Challenge - The Discontinuity Wall

## Objetivo
Evaluar el rendimiento de las redes en la función módulo ($x \pmod y$), una de las funciones más difíciles para redes neuronales continuas debido a sus discontinuidades abruptas.

## Resultados (x % y)

| Modelo | Parámetros | Train MSE | Far OOD MSE | Ratio (Estabilidad) |
| :--- | :--- | :--- | :--- | :--- |
| **MLP-Huge** | 1,052,673 | **0.0055** | **4.11** | 743 |
| **Poly-Deep-V193** | **28,385** | 0.0678 | 15.53 | **229** |

## Conclusiones Técnicas

### 1. La Fuerza Bruta gana en el Rango Corto
El **MLP-Huge**, con más de un millón de parámetros, logró una precisión local 12 veces superior a la red polimórfica. Al tener tantas neuronas ReLU, puede "construir" la sierra del módulo mediante la suma de muchísimas funciones lineales por trozos. Sin embargo, esta solución es puramente local (memorización).

### 2. Estabilidad Polimórfica
A pesar de tener un error absoluto mayor, la red **Poly-Deep-V193** demostró ser **3 veces más estable** en la extrapolación. Esto indica que las bases de resonancia y logarítmicas están intentando capturar la "periodicidad" del módulo, aunque la falta de una base de discontinuidad explícita (como un escalón o una sierra) limita su precisión.

### 3. El Problema de la Composición
El módulo se define como $x - y \cdot \lfloor x/y \rfloor$. Capturar la función `floor` ($\lfloor \cdot \rfloor$) o la división dentro de una función periódica es una tarea de composición extrema que sigue siendo el "talón de Aquiles" de las arquitecturas continuas.

## Próximos Pasos (V195)
-   **Bases de Discontinuidad**: Introducir una rama de **Funciones de Activación Signo o Step** para permitir saltos abruptos.
-   **Mecanismo de Residuo Sawtooth**: Probar si una base que calcule directamente $x - \text{round}(x)$ mejora drásticamente el rendimiento en el módulo.
