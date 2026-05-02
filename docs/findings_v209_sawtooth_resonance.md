# Hallazgos V209: Neuronas Sawtooth y el Límite Lineal

## Objetivo
Resolver el "Fenómeno de Gibbs" (identificado en V208) sustituyendo el oscilador armónico (`cos`) por un oscilador discontinuo nativo (`Sawtooth`), eliminando la necesidad de infinitos armónicos para ajustar la función módulo ($x \pmod y$).

## Resultados (x % y)

| Modelo | Parámetros | Train MSE | Far OOD MSE | Ratio (Estabilidad) |
| :--- | :--- | :--- | :--- | :--- |
| **Poly-Deep-V193** | 28,385 | 0.0678 | 15.53 | 229 |
| **Resonant-Phase-V207 (Cos)**| 17,089 | **0.0612** | 29.62 | 484 |
| **Multi-Resonant-V208 (Cos+Log)**| 753 | 0.0551 | 551.76 | 10,000 |
| **Sawtooth-Resonant-V209** | 17,473 | 1.0745 | **30.48** | **28.4** |

## Conclusiones y Análisis

### 1. El Colapso del Entrenamiento Local (Train MSE: 1.07)
A diferencia de la red V207 con cosenos (que logró un MSE local de 0.06), la red Sawtooth fue incapaz de ajustar siquiera los datos de entrenamiento. El motivo matemático es revelador:
La función `Sawtooth(p) = p - round(p)` es una función **estrictamente lineal a trozos**. Dentro de cada intervalo, la neurona actúa como una regresión lineal pura. Mientras que la red V207 usó la curvatura natural de los cosenos (expansión de Taylor) para aproximar sutilmente la no-linealidad de la división $x/y$ en el rango local, la red Sawtooth no tiene curvatura. Al estar forzada a usar fases fijas ($w_1 x + w_2 y$) y al carecer de curvatura, es matemáticamente rígida y fracasa rotundamente.

### 2. Estabilidad Extrema (Ratio: 28.4)
Lo fascinante es que, a pesar de no poder ajustar la curva, la red Sawtooth tiene el **mejor Ratio de Estabilidad de todas las redes de resonancia**. Su error OOD de 30.48 es casi idéntico al de la red V207 (29.62), pero partiendo de un error base muchísimo peor. Al ser lineal a trozos, no sobreajustó. Simplemente trazó líneas rectas discontinuas que se comportan igual de (mal) dentro y fuera de la distribución.

### 3. El Veredicto Final del Módulo
La función Módulo exige dos requisitos simultáneos que nuestras neuronas actuales no tienen juntos de forma estable:
1. **Discontinuidad limpia:** (Para evitar el Fenómeno de Gibbs - Logrado por V209).
2. **Capacidad de División Estricta ($x/y$):** (Para modular la frecuencia de la discontinuidad - Intentado en V208 pero con explosión exponencial).

Para derrotar a este problema, necesitaríamos una "Neurona Sawtooth Analítica" a la que se le inyecte de forma segura el término $x/y$ (por ejemplo, mediante una capa de división real o Attention) sin depender de logaritmos inestables. Por ahora, las Redes Resonantes reinan absolutamente en Visión (donde las coordenadas son fijas), pero requieren un nuevo paradigma aritmético para dominar las matemáticas abstractas puras.
