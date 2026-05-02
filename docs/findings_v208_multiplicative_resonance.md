# Hallazgos V208: Explosión Multiplicativa y el Fenómeno de Gibbs

## Objetivo
Resolver la función módulo ($x \pmod y$) modificando la Neurona de Resonancia para permitir la modulación dinámica de frecuencia mediante características multiplicativas polinómicas/racionales: $\exp(W \cdot \log(x))$. La teoría indicaba que la red podría descubrir el término exacto $x/y$ para generar armónicos perfectos.

## Resultados (x % y)

| Modelo | Parámetros | Train MSE | Far OOD MSE | Ratio (Estabilidad) |
| :--- | :--- | :--- | :--- | :--- |
| **Poly-Deep-V193** | 28,385 | 0.0678 | 15.53 | 229 |
| **Resonant-Phase-V207**| 17,089 | 0.0612 | 29.62 | 484 |
| **Multi-Resonant-V208**| **753** | **0.0551** | **551.76** | **10,000** |

## Análisis del Colapso (Error Conceptual)

A pesar de lograr el mejor Train MSE con solo 753 parámetros, la red V208 sufrió un **colapso catastrófico** en extrapolación (Far OOD MSE de 551). Hemos identificado dos culpables teóricos fundamentales:

### 1. Inestabilidad Exponencial Continua (El problema de x/y)
La red podía teóricamente aprender el término $x/y$ asignando los pesos exactos $[1.0, -1.0]$ en su matriz $W_{log}$. Sin embargo, el descenso de gradiente (Backpropagation) busca mínimos locales en espacios continuos. En lugar de pesos exactos enteros, la red aprende aproximaciones sucias como $x^{0.95} y^{-1.02}$.
Dentro del dominio de entrenamiento ($x,y \in [0, 5]$), la diferencia es minúscula. Pero al extrapolar a OOD ($x, y \in [0, 20]$), el término $x^{0.95}$ diverge agresivamente del comportamiento lineal real, provocando una explosión exponencial de los valores de fase.

### 2. El Fenómeno de Gibbs (Cosenos vs Discontinuidades)
El módulo es una función de sierra (Sawtooth) que contiene una discontinuidad abrupta (un salto vertical). Por el Teorema de Fourier, ajustar un salto abrupto usando funciones suaves y continuas (cosenos) genera oscilaciones violentas en los bordes llamadas el "Fenómeno de Gibbs". Para suprimir estas oscilaciones se necesitan infinitos armónicos meticulosamente sincronizados. Al tener solo 32 osciladores, la red sobreajustó los pesos logarítmicos para forzar el ajuste local, sacrificando toda su capacidad de extrapolación.

## Conclusión
Combinar logaritmos/exponenciales con ondas continuas para aproximar funciones discontinuas es una "receta para el desastre" en extrapolación (OOD). Las arquitecturas puramente continuas no pueden aproximar saltos discretos sin sobreajustar de forma letal.

Esto valida tu conclusión original en `V194`: Necesitamos **Bases de Discontinuidad**. No podemos usar Cosenos o ReLUs para aprender una Sierra. Necesitamos integrar funciones como el redondeo (`round`, `floor`) directamente en el núcleo de la neurona de resonancia.
