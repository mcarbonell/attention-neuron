# Research Plan V191: Logarithmic Interactions in Polymorphic Neurons

## Contexto
En V190, confirmamos que las redes polimórficas generalizan mejor en extrapolación extrema (Ratio de 2.16 en Schwefel vs 3,431 en MLP). Sin embargo, en funciones puramente multiplicativas como `prod(x,y)`, el error seguía siendo significativo debido a que la interacción se basaba en productos circulares ad-hoc.

La mayoría de las leyes físicas y matemáticas fundamentales son de naturaleza multiplicativa o de potencias ($y = a \cdot x^b \cdot z^c$). Estas relaciones se vuelven lineales en el espacio logarítmico: $\log(y) = \log(a) + b\log(x) + c\log(z)$.

## Objetivos
1.  **Linealizar el Producto**: Implementar una capa que transforme las entradas al dominio logarítmico para resolver interacciones complejas como sumas lineales.
2.  **Manejo de Signos**: Diseñar un mecanismo para preservar el signo de las entradas originales, ya que $\log(x)$ solo está definido para $x > 0$.
3.  **Benchmarking Comparativo**: Validar si esta arquitectura logra error casi cero en `prod` y `div` en rangos lejanos, superando tanto a MLPs como a la arquitectura V190.

## Arquitectura Propuesta: Log-Interaction Layer
1.  **Preprocessing**: $x_{abs} = |x| + \epsilon$.
2.  **Log-Transform**: $x_{log} = \log(x_{abs})$.
3.  **Linear Combination**: $z_{log} = W \cdot x_{log} + b$.
4.  **Exp-Transform**: $z = \text{sign}(x_{combined}) \cdot \exp(z_{log})$.
    - *Nota*: El signo se puede inferir de la paridad de los pesos o mediante una rama paralela que procese `sign(x)`.

## Funciones Críticas para V191
-   `prod(x,y)`
-   `div(x,y)`
-   `power(x,y)` ($x^y$)
-   `law_gravitation` ($G \frac{m_1 m_2}{r^2}$)

## Protocolo de Evaluación
Mismo protocolo OOD que V190 (Train, Near, Far) para asegurar comparabilidad directa.
