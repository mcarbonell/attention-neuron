# Research Plan V193: Deep Resonant-Log-Polymorphism

## Contexto
En V192, demostramos que una única capa de neuronas polimórficas (Tríada: Estructural, Logarítmica, Resonante) es masivamente superior a los MLPs en estabilidad OOD. Sin embargo, la precisión local (error de entrenamiento) sigue estando limitada por la incapacidad de la red para modelar composiciones de funciones ($f(g(x))$) de forma nativa.

Este experimento introduce la **Profundidad** en el paradigma polimórfico, permitiendo que las leyes descubiertas en una capa sirvan como bases para la siguiente.

## Objetivos
1.  **Composición de Leyes**: Evaluar si el polimorfismo profundo puede modelar funciones compuestas con la misma eficiencia que funciones simples.
2.  **Precisión Local**: Reducir el MSE de entrenamiento en benchmarks complejos (Rastrigin, Schwefel) sin comprometer la estabilidad OOD.
3.  **Análisis de Parámetros**: Mantener el presupuesto de parámetros por debajo del MLP-L, verificando si "profundidad polimórfica" es más potente que "ancho denso".

## Arquitectura Propuesta: Deep-Poly-Resonant
-   **Stage 1**: Capa Polimórfica V192 (Resonant + Log + Structural).
-   **Stage 2**: Una segunda capa que toma la salida de Stage 1 y la vuelve a proyectar a bases polimórficas.
-   **Residual Path**: Conexión identidad entre etapas para facilitar el flujo de gradiente.

## Protocolo Experimental
-   **Comparativa**: MLP-L (Baseline) vs Poly-V192 (1-layer) vs Poly-V193 (2-layers).
-   **Funciones**: Las mismas de V192 para asegurar continuidad de resultados.
-   **Métrica**: MSE Train y Ratio de Estabilidad OOD.
