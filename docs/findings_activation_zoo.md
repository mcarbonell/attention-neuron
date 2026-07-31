# Hallazgos: Zoológico de Activaciones Periódicas-Infinitas

## Introducción
Tras el éxito de la rampa sigmoidal periódica, exploramos otras combinaciones de funciones periódicas divergentes y funciones de confinamiento (squashing). Cada combinación ofrece una topología única para el aprendizaje de ritmos y patrones espectrales.

## Comparativa Visual

![Zoológico de Activaciones](../results/figures/activation_zoo.png)

![Zoológico de Activaciones v2](../results/figures/activation_zoo_v2.png)

## Análisis de Especies

### 1. La Rampa Lineal: $\arctan(\tan(x))$
*   **Comportamiento:** Un "Sawtooth" perfecto.
*   **Ventaja:** Gradiente constante ($=1$). No sufre de saturación. Es la forma más pura de inyectar aritmética de módulos en una red.
*   **Uso ideal:** Regresión matemática y cálculo de índices de memoria circular.

### 2. El Detector de Pulsos: $\sigma(\sec(x))$
*   **Comportamiento:** Picos periódicos extremadamente agudos.
*   **Ventaja:** Funciona como un **filtro de coincidencia**. La neurona solo "despierta" cuando la señal de entrada está en una fase muy específica.
*   **Uso ideal:** Detección de anomalías periódicas o sincronización de relojes internos en arquitecturas recurrentes.

### 3. La Escalera Diferenciable: $x - \text{Mod}(x)$
*   **Comportamiento:** Una serie de escalones planos con subidas rápidas.
*   **Ventaja:** Permite simular la función `round()` o `floor()` de forma que el gradiente pueda fluir. La red puede aprender a "cuantizar" sus propios valores.
*   **Uso ideal:** Direccionamiento de memoria (Indexing), aprendizaje de constantes enteras y compresión de pesos (Quantization).

El problema de round(x) es que su gradiente es cero en todas partes y NaN en los saltos. Si una red intenta aprender a "redondear", se queda ciega.

Pero, usando lo que acabamos de descubrir con la tangente, podemos crear una "Escalera Diferenciable".

El "Soft Round" mediante Tangente
Sabemos que $x - (x \pmod 1)$ nos da la parte entera. Si sustituimos el módulo por nuestra versión periódica $\arctan(\tan(\dots))$, obtenemos una función que:

Tiene mesetas horizontales (como un redondeo).
Tiene gradiente en las subidas, lo que permite que la red "empuje" un valor de un escalón a otro durante el entrenamiento.

### 4. La Rampa Bipolar: $\tanh(\tan(x))$
*   **Comportamiento:** Una rampa sigmoidal que va de $-1$ a $1$.
*   **Ventaja:** Centrada en cero. Facilita la estabilidad numérica en redes con muchas capas.
*   **Uso ideal:** Sustituto directo de capas densas en arquitecturas espectrales profundas.

## Conclusión
No existe una "mejor" función. La elección depende de la naturaleza del problema:
*   Si buscas **precisión aritmética**, usa el Arcotangente.
*   Si buscas **especialización de fase**, usa la Secante.
*   Si buscas **estabilidad en clasificación**, usa el Sigmoide o Tanh.

Estas funciones abren una nueva dimensión de **Diseño Topológico de Activaciones** que apenas estamos empezando a explorar.






-----

## El zoo de activaciones: hay un error matemático con arreglo directo

**Primero, lo bueno:** el diagnóstico de que `round()` tiene gradiente cero y ciega a la red es correcto, y buscar una relajación diferenciable es el instinto acertado.

**El bug numérico.** $\arctan(\tan(x)) = x - \pi\cdot\text{round}(x/\pi)$. Es la diente de sierra, sí — pero la estás calculando con dos transcendentes que se cancelan, atravesando una singularidad. En float32, `tan(x)` cerca de $\pi/2$ pierde precisión catastróficamente. Equivalente exacto, sin singularidad y más rápido:

```python
sawtooth = torch.remainder(x + math.pi/2, math.pi) - math.pi/2
```

**El error de fondo, que sí importa.** Dices que $x - \text{Mod}(x)$ es una "escalera diferenciable" que permite empujar valores entre escalones. No:

$$\frac{d}{dx}\big[x - \text{sawtooth}(x)\big] = 1 - 1 = 0$$

**Gradiente exactamente cero en casi todo punto.** Los "subidas" con gradiente son las discontinuidades, que tienen medida nula. Es el mismo `round()` con el que empezaste, con más FLOPs. En autodiff no vas a ver ningún error — simplemente esa rama no aprende nunca, y es invisible en el log.

**El arreglo, y es de una línea:**

$$r_\alpha(x) = x - \alpha\,\frac{\sin(2\pi x)}{2\pi}, \qquad r_\alpha'(x) = 1 - \alpha\cos(2\pi x)$$

Con $\alpha=1$ el gradiente es 0 en los enteros y 2 entre ellos: empuja los valores *hacia* los enteros y deja gradiente en todo lo demás. Recocido de $\alpha: 0 \to 1$ durante el entrenamiento te da soft-to-hard cuantización de verdad. Eso es exactamente lo que querías construir.

Literatura que te sitúa: **Straight-Through Estimator** (Bengio et al. 2013) — el estándar y el rival a batir; **soft-to-hard vector quantization** (Agustsson et al., NeurIPS 2017) — el recocido; **ruido uniforme aditivo** (Ballé et al. 2017) — la alternativa en compresión. Y para tu zoo periódico completo: **Ziyin, Hartwig & Ueda 2020**, *Neural Networks Fail to Learn Periodic Functions and How to Fix It* (activación *snake*), y **SIREN** (Sitzmann et al. 2020), donde la inicialización con activaciones sinusoidales lo es todo.

Y el que más te va a interesar: **Nanda et al. 2023**, *Progress Measures for Grokking*. Encontraron mecanísticamente que un transformer aprendiendo suma modular **construye representaciones trigonométricas por su cuenta**. Tu hipótesis de "aritmética de módulos vía activaciones periódicas" tiene evidencia interpretability a favor.
