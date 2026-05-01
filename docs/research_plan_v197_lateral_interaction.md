# Research Plan V197: Lateral Interaction & Child Neurons

## Contexto
En V196, logramos una neurona individual masivamente potente (la "Neurona Total"). Sin embargo, las neuronas en una misma capa siguen procesando la información de forma aislada. La comunicación solo ocurre al pasar a la siguiente capa.

La idea del usuario es permitir que las neuronas "se hablen" dentro de la misma capa, creando **Neuronas Hijas** que sean combinaciones operacionales (suma, producto, módulo) de las **Neuronas Padres**.

## Objetivos
1.  **Interacción Horizontal**: Implementar una capa de interacción lateral donde las neuronas de un mismo nivel puedan combinarse mediante operaciones primitivas.
2.  **Riqueza Simbólica**: Evaluar si permitir $\{+, -, \cdot, \pmod\}$ entre neuronas en el mismo nivel reduce la necesidad de profundidad y mejora la precisión en leyes compuestas.
3.  **Benchmarking**: Comparar la "Capa con Interacción Lateral" contra la "Capa Profunda V193".

## Arquitectura Propuesta: Lateral Poly-Layer
1.  **Base Layer**: Genera $N$ activaciones polimórficas (los "Padres").
2.  **Interaction Matrix**:
    -   Calcula combinaciones par-a-par: $P_i + P_j$, $P_i \cdot P_j$, $P_i \pmod P_j$.
    -   Utiliza un mecanismo de **Gating** para seleccionar qué interacciones son relevantes.
3.  **Derived Layer**: El output final de la capa es la unión de los Padres y las Hijas seleccionadas.

## Desafío Técnico: Diferenciabilidad del Módulo Lateral
Al igual que en V195, usaremos **STE (Straight-Through Estimator)** para que el gradiente fluya a través de las interacciones de módulo entre neuronas.

## Protocolo Experimental
-   **Benchmark**: Funciones de composición extrema (ej. $\sin(x \cdot y) + (x \pmod y)$).
-   **Métrica**: Comparar cuántas "Hijas" son necesarias para igualar el rendimiento de 2 capas secuenciales.
