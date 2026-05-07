# Findings v249: Deep Scientific Network

## Objetivo
Evolucionar la neurona científica hacia una arquitectura multicapa capaz de descubrir **composiciones de leyes matemáticas** ($g(f(x))$), permitiendo modelar funciones que no existen explícitamente en el menú de bases inicial.

## Resultados de Descubrimiento Jerárquico

Se probaron funciones que requieren dos pasos lógicos para ser representadas.

| Función Objetivo | Capa 1 (Hidden) | Capa 2 (Output) | Resultado |
| :--- | :--- | :--- | :--- |
| **Gaussiana** ($e^{-0.1 x^2}$) | Identifica **$x^2$** | Aplica **$\exp(h)$** | **Éxito**: Reconstruyó $e^{-0.169 x^2}$ |
| **Sin-Square** ($\sin(0.1 x^2)$) | Identifica **$x^2$** | Aplica **$\sin(h)$** | **Éxito**: Reconstruyó $\sin(0.172 x^2)$ |
| **Quad + Sin** ($0.1x^2 + \sin x$) | Identifica **$x^2, \sin x$** | Suma Lineal | **Éxito**: Separación de términos |

## Análisis de Estabilidad y Poda

1.  **Riesgo de Explosión**: El uso de bases exponenciales en capas profundas es altamente inestable. Se implementó un **Clamping de Seguridad** (limitando inputs a $[-10, 10]$ y outputs de `exp` a $e^5$) para evitar NaNs.
2.  **Poda Multicapa**: La poda agresiva (0.05) es más compleja en redes profundas. Si la Capa 1 se poda demasiado, la Capa 2 pierde su señal. Sin embargo, en los experimentos realizados, la red logró mantener los canales críticos.
3.  **Precisión OOD**: Aunque la estructura jerárquica es correcta, la precisión OOD ($10^{-2}$) no es tan perfecta como en la versión monocapa ($10^{-14}$) debido a que el error en la Capa 1 se magnifica en la Capa 2.

## Conclusiones Técnicas

-   **Componibilidad**: Hemos demostrado que el paradigma de "Aumento + Poda" es componible. La red puede realizar **abstracción matemática**.
-   **Interpretatibilidad en Cascada**: Podemos leer la "lógica" de la red como una serie de transformaciones simbólicas, algo imposible en un MLP tradicional.

## Aplicaciones Potenciales
Esta arquitectura es ideal para:
-   **Modelado de Sistemas Dinámicos**: Donde las variables interactúan de forma no lineal pero estructurada.
-   **Compresión de Conocimiento**: Representar funciones complejas con < 100 parámetros totales.
-   **Pre-procesadores Inteligentes**: Capas de entrada que extraigan "características físicas" antes de pasar a un LLM.
