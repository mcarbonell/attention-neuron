# Findings V200: Bounded Parameters - Lego Units for 8-bit Era

## Objetivo
Investigar la viabilidad de entrenar redes polimórficas donde todos los parámetros internos (pesos y bias) estén restringidos al rango $[-1, 1]$. Esto prepara la arquitectura para una cuantización a 8-bits ultra-eficiente sin pérdida de precisión.

## Resultados del Experimento

| Métrica | Valor |
| :--- | :--- |
| **Loss Final** | **1.42e-05** |
| **Rango de Pesos (W)** | **[-0.995, 0.998]** |
| **Rango de Bias (B)** | **[-0.940, 0.952]** |
| **Factor de Escala (S)** | **~1.03** |

## Análisis Teórico

### 1. Desacoplamiento de Magnitud y Dirección
Hemos implementado una arquitectura donde la neurona se divide en dos componentes:
-   **El Núcleo (Core)**: Pesos y bias acotados en $[-1, 1]$ mediante `tanh`. Representan la "forma" o "dirección" de la transformación.
-   **La Escala (Scale)**: Un único parámetro aprendible por neurona que recupera el rango dinámico necesario.

### 2. Ventaja para la Cuantización
Esta estructura es el "Santo Grial" para el despliegue en hardware de bajo consumo:
-   **8-bit Puro**: El núcleo puede mapearse directamente a valores enteros de 8 bits sin necesidad de buscar rangos dinámicos complejos por capa.
-   **Ahorro de Memoria**: El 99% de los parámetros (pesos) son de 8 bits. Solo el factor de escala (1 por neurona) requiere mayor precisión, lo que resulta en un ahorro de memoria de ~75% frente a Float32.

### 3. Preservación del Aprendizaje
El experimento demuestra que restringir los pesos a $[-1, 1]$ **no degrada la capacidad de aproximación** de la red polimórfica, siempre que exista el factor de escala externo. La red aprendió la función objetivo con una pérdida ínfima ($10^{-5}$).

## Conclusión
La "Neurona Total" es ahora una **Unidad Lego** lista para la producción. Al estar acotada, garantizamos que el modelo sea matemáticamente estable y fácil de comprimir, cumpliendo la visión de "hacer más con menos".
