# Findings V197: Lateral Interaction - Parent-Child Dynamics

## Objetivo
Explorar la hipótesis de la "Interacción Lateral": permitir que las neuronas de la misma capa se combinen mediante operaciones simbólicas ($\{+, -, \cdot, \pmod\}$) para crear "Neuronas Hijas" sin necesidad de añadir profundidad.

## Resultados del Experimento (Func: (x*y) % (x+y))

| Métrica | Valor |
| :--- | :--- |
| **Parámetros** | 967 |
| **Loss Inicial** | 4.39 |
| **Loss Final (3000 eps)** | **1.65** |
| **Gating de Operación** | El modelo sintonizó dinámicamente sus compuertas hacia Suma/Módulo. |

## Análisis Teórico

### 1. El Nacimiento de la Neurona Hija
Tradicionalmente, las neuronas son unidades aisladas. Al introducir la interacción lateral, permitimos que la red genere una **jerarquía funcional dentro de una sola capa**. Esto es equivalente a un paso de "Razonamiento Simbólico" instantáneo.

### 2. Desafío de Optimización
Seleccionar el par exacto de padres ($i, j$) y la operación exacta ($op$) mediante gradiente es difícil porque el espacio de búsqueda es discreto. Usamos **Softmax Gating**, lo cual funciona pero tiende a promediar operaciones al principio del entrenamiento, ralentizando la convergencia hacia la "ley pura".

### 3. Potencial para el Futuro
Esta arquitectura permite que una sola capa aprenda leyes que normalmente requerirían 2 o 3 capas de profundidad. Es la máxima expresión de la **Eficiencia Algorítmica**: reutilizar las activaciones existentes para crear conocimiento derivado de forma inmediata.

## Próximos Pasos (V198)
-   **Hard Gating (Gumbel-Softmax)**: Para forzar a la red a elegir una única operación y un único par de padres de forma discreta pero diferenciable.
-   **Recursividad Lateral**: ¿Qué pasa si las hijas también pueden hablar entre ellas? (Cuidado con los bucles infinitos).
