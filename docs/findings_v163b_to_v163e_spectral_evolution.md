# Findings V163b - V163e: La Evolución hacia el LLM Espectral

## 1. Escalado Masivo (V163b)
*   **Hito:** Escalamos el ancho del FFN a 4,096 expertos.
*   **Resultado:** Precisión 95.68% (un salto desde el 92%).
*   **Lección:** La inteligencia en el dominio espectral es una función directa del "ancho" del manifold. Más expertos = mayor resolución de la realidad.

## 2. El Límite del Multiplexado (V163c)
*   **Hito:** Intentamos sumar tokens en un solo vector (Suma simple).
*   **Resultado:** Colapso a partir del 2º token (SNR < 1.5).
*   **Lección:** La interferencia constructiva/destructiva requiere una codificación que preserve la identidad. No se puede "apilar" información sin una clave.

## 3. MoE Extremo (V163d - "El Monstruo")
*   **Hito:** Inferencia con 131,072 expertos espectrales.
*   **Resultado:** 308 tokens/s en CPU con 517 MB de RAM.
*   **Lección:** Hemos roto la barrera de los parámetros. Podemos tener modelos de escala masiva que corren en hardware de consumo gracias a la activación ultra-dispersa (0.01%).

## 4. El Sentido del Tiempo (V163e)
*   **Hito:** Multiplexado Espaciotemporal usando desplazamientos circulares (Roll).
*   **Resultado:** Recuperación perfecta de 4 tokens en el mismo espacio (vs 2 en v163c).
*   **Lección:** El orden es una dimensión espectral. Al desplazar las firmas, creamos ortogonalidad temporal, permitiendo una memoria de contexto comprimida.
