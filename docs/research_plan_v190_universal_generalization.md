# Research Plan V190: Structural Generalization vs Statistical Interpolation

## Contexto
En el experimento V132 (Universal Approximation), observamos que los MLPs pueden "memorizar" o "interpolar" puntos de una función matemática con alta precisión, pero no necesariamente comprenden la estructura subyacente. Las **Neuronas Polimórficas**, al tener bases matemáticas explícitas, demostraron una capacidad superior de generalización en funciones que coincidían con sus bases (como $x^2$).

Este experimento busca formalizar esta comparación midiendo el rendimiento **Out-of-Distribution (OOD)** en diferentes niveles de "distancia" respecto al rango de entrenamiento.

## Objetivos
1.  **Cuantificar la Generalización**: Comparar el MSE en rangos de entrenamiento vs rangos de extrapolación.
2.  **Identificar el Límite Estructural**: Determinar hasta qué punto una arquitectura con sesgo inductivo correcto puede alejarse de los datos de entrenamiento sin perder precisión.
3.  **Baseline Robusto**: Evaluar MLPs de diferentes profundidades y anchos para ver si el "exceso de parámetros" ayuda o perjudica la extrapolación.

## Funciones de Referencia

### 1D (Estructurales y Periódicas)
-   **Polinómicas**: $x^2$, $x^3$
-   **Singularidad**: $1/x$ (Entrenar lejos de 0, testear cerca y lejos).
-   **Trigonométricas**: $\sin(x)$, $\tan(x)$
-   **Signal processing**: $\text{sinc}(x)$

### 2D (Operaciones de Interacción)
-   **Lineales**: $x + y$, $x - y$
-   **No Lineales**: $x \cdot y$, $x / y$
-   **Discontinuas**: $x \pmod y$

### N-D (Funciones de Optimización Benchmark)
-   **Rastrigin**: Altamente periódica con muchos mínimos locales.
-   **Ackley**: Superficie rugosa con un pozo central.
-   **Schwefel**: Superficie compleja con picos y valles distantes.

## Protocolo Experimental

### Rangos de Evaluación
Para cada función, definiremos tres dominios:
1.  **D_train**: El rango donde se muestrean los datos de entrenamiento (ej. $[-2, 2]$).
2.  **D_near**: Un rango ligeramente expandido (ej. $[-4, 4]$) para medir la extrapolación inmediata.
3.  **D_far**: Un rango masivamente expandido (ej. $[-10, 10]$) para medir la "comprensión de la ley".

### Arquitecturas a Comparar
-   **MLP-S (Small)**: $\sim 50$ parámetros.
-   **MLP-M (Medium)**: $\sim 4,000$ parámetros.
-   **MLP-L (Large)**: $\sim 50,000$ parámetros.
-   **Structural-Poly (Experimental)**: Arquitectura polimórfica que incluye capas de interacción (Cross-Products) para resolver las fallas de V132.

## Métricas
-   `mse_train`: MSE en $D_{train}$.
-   `mse_near`: MSE en $D_{near}$.
-   `mse_far`: MSE en $D_{far}$.
-   `generalization_ratio`: $\frac{mse_{far}}{mse_{train}}$ (Cuanto más bajo, mejor).

## Próximos Pasos
1.  Implementar `prototype_v190_generalization_benchmark.py`.
2.  Ejecutar el benchmark completo.
3.  Documentar hallazgos en `docs/findings_v190_universal_generalization.md`.
