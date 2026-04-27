# Hallazgos Experimentos V41-V49: La Frontera de las Proyecciones Congeladas

## Resumen Ejecutivo
Se ha realizado una exploración exhaustiva de la **Capa de Entrada Congelada** en MNIST. Hemos descubierto que la **localidad esparcida** y el **contraste diferencial** son los dos pilares que permiten a una red clasificar con alta precisión (>98%) sin necesidad de entrenar su frontend.

La serie **V48** demostró que tratar a la red como un detector de comparaciones locales (+1/-1) es superior a cualquier base matemática predefinida (Gabor/Fourier).

---

## 1. Clasificación de Estrategias y Resultados

| Familia | Experimento | Descripción | Acc Pico | Conclusión |
| :--- | :--- | :--- | :--- | :--- |
| **Global** | V41 | Ruido Blanco Global | 97.09% | Referencia ELM clásica. |
| **Global** | V45 | Base de Fourier | 96.36% | Buena cobertura, poca especificidad. |
| **Local Sum** | V44 | Parches Aleatorios (6x6-14x14) | 97.90% | **Mejor Acc/Param (20k params)**. |
| **Local Sum** | V48d | Dual Parches 2x2 | 98.25% | La ultra-localidad funciona. |
| **Contraste** | V48f | Diferencia 3x3 (+1/-1) | 98.35% | El contraste es más informativo que la suma. |
| **Contraste** | **V48h** | **Quad Contrast (2+ vs 2-)** | **98.44%** | **Mejor configuración estructural**. |
| **Estructura** | V48j | On-Center / Off-Surround | 98.08% | Biológicamente inspirado, muy robusto. |
| **Profundo** | V46 | Local Perlin + MLP 512 | **98.58%** | **Record Absoluto de la sesión**. |

---

## 2. Lecciones Aprendidas

1. **El Poder del Contraste (+/-)**: Las neuronas que restan información de un parche a otro actúan como detectores lógicos ("Hay trazo aquí pero NO aquí"). Esto genera una separación del espacio de características mucho más limpia para el clasificador.
2. **Localidad Esparcida**: MNIST no necesita ver toda la imagen a la vez. El éxito de los parches 2x2 y 3x3 sugiere que la red puede reconstruir el número simplemente sabiendo "qué micro-texturas hay y dónde están".
3. **Punto Dulce de Parches**: Añadir demasiados parches por neurona (V48i) no mejora el pico, solo estabiliza el promedio. El cerebro visual parece preferir detectores simples y específicos.
4. **Eficiencia Extrema (V44)**: Alcanzar un **97.9% con solo 20,490 parámetros** entrenables es el resultado más impresionante en términos de compresión de inteligencia.

---

## 3. Estado del Arte (SOTA) y Contexto

Respecto al **SOTA de MNIST (99.8% - 99.9%)**, nuestras redes congeladas están a un ~1.5% de distancia. 

### ¿Por qué no llegamos al 99%?
Las redes SOTA actuales (como las de *Wan et al.* o ensambles de CNNs) utilizan:
- **Aumentación Elástica**: Deforman los números como si fueran de goma.
- **Convoluciones Entrenables**: Sus filtros se adaptan exactamente a la curvatura de los números.
- **Comités de Expertos**: Usan 30-50 redes votando juntas.

Nuestras redes, al tener la **primera capa congelada**, son "inflexibles" ante pequeñas rotaciones o traslaciones. Cada píxel de desplazamiento cambia totalmente la activación de nuestros parches fijos.

---

## 4. Conclusión Final de la Sesión
Hemos demostrado que es posible construir un clasificador de dígitos casi perfecto (**98.5%**) delegando la visión a una **proyección aleatoria local y fija**, centrando todo el aprendizaje en un readout MLP sencillo. 

La **V44** queda como el estándar de eficiencia y la **V48h** como el estándar de arquitectura de contraste para futuros trabajos en proyecciones congeladas.
