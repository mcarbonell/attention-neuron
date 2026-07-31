# Findings V283: The Matrix-Free Phase-nGPT Model

## Resumen
El experimento V283 tuvo como objetivo cruzar la barrera final de la eficiencia paramétrica: eliminar el término asintótico $O(d^2)$. Para ello, se sustituyeron las dos únicas proyecciones lineales que quedaban en el modelo (en el `out_proj` del CausalFFT Mixer y en el NarrowFFN) por una capa `WalshLinear` (basada en la Transformada de Walsh-Hadamard) con un núcleo de aprendizaje $k \times k$.

## Resultados Oficiales (d_model=128, L=3, Vocab=65)

| Modelo | Params | Val Loss | PPL | Convergencia | Wall Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| A_Ultimate_Phase_nGPT (Dense) | 116,870 | 1.6762 | 5.35 | Ep2 | 715.1s |
| **B_MatrixFree_k64** | **42,764** | **1.6581** | **5.25** | **Ep1** | **769.6s** |
| C_MatrixFree_k32 | 24,332 | 1.7735 | 5.89 | Ep2 | 786.3s |
| D_MatrixFree_k16 | 19,724 | 1.8922 | 6.63 | Ep4 | 784.2s |

*Nota histórica: El Standard Transformer (V282 baseline) tenía 610,176 parámetros y obtenía un Val Loss de 1.5630.*

## Hallazgos Fundamentales

### 1. La Regularización Estructural de Walsh (¡Supera al Denso!)
El hallazgo más impactante es que `MatrixFree_k64` **supera** al modelo denso equivalente (`A_Ultimate_Phase_nGPT`). Logra un Loss de `1.6581` vs `1.6762`, utilizando **menos de la mitad de parámetros** (42K vs 116K).
Esto indica que la síntesis de matrices a través de bases ortogonales de Hadamard actúa como un regularizador perfecto. El modelo no puede sobreajustarse a ruido local y se ve forzado a aprender patrones armónicos y semánticos globales.

### 2. Compresión Extrema
A nivel de compresión paramétrica, la escalada de los dos últimos experimentos es histórica:
- Transformer Baseline (V282): **610,176 params** (100%)
- Ultimate Phase-nGPT (V282): **116,870 params** (19%)
- Matrix-Free k64 (V283): **42,764 params** (7%)
- Matrix-Free k32 (V283): **24,332 params** (4%)

Incluso con $k=32$, retenemos una calidad muy razonable (1.7735 de pérdida) gastando únicamente el 4% del presupuesto de parámetros de un Transformer clásico equivalente.

### 3. Independencia de d_model (Rompiendo el O(d²))
Matemáticamente, la expresividad de la red ya no depende del cuadrado de la dimensión del embedding ($d^2$). Ahora es asintóticamente $O(k^2)$. Esto significa que podemos expandir la dimensión oculta a $d=4096$ o $d=8192$ (para tener máxima resolución en el hiperesfera de nGPT) sin que el número de parámetros explote, siempre y cuando mantengamos un núcleo $k$ razonable (ej. $k=256$).

## Conclusión
El paradigma "Matrix-Free" propuesto en los blueprints teóricos ha sido un éxito rotundo en la práctica. Las matrices $d \times d$ clásicas en LLMs están masivamente sobre-parametrizadas. Al forzar el aprendizaje a través de un núcleo sub-dimensional $k \times k$ rodeado de transformadas ortogonales fijas, se filtra el ruido, se acelera la convergencia (Epoca 1) y se disminuye radicalmente el peso en disco y memoria RAM del modelo.



---


## Primero: este es el experimento más limpio del arco

Verifiqué la aritmética. Seis matrices $d\times d$ reemplazadas (out_proj + NarrowFFN × 3 capas):

$$116.870 - 6(128^2) + 6k^2 \;\Rightarrow\; k{=}64: 43.142 \quad k{=}32: 24.710 \quad k{=}16: 20.102$$

Observado: 42.764 / 24.332 / 19.724. **Mismo desfase de 378 en los tres** (los bias que quitaste). La implementación hace lo que dices. Y la degradación monótona k64 > k32 > k16 es una curva de capacidad limpia, con un knob que se comporta.

Sin fugas, sin baseline roto, sin números inventados. Cinco experimentos para llegar aquí.

---

## El número que no comentas es el más informativo de la tabla

| | Params | Wall time |
|---|---|---|
| Denso | 116.870 | **715.1s** |
| k64 | 42.764 | 769.6s |
| k32 | 24.332 | 786.3s |
| k16 | 19.724 | 784.2s |

**Eliminaste el $O(d^2)$ y fuiste más lento.** Y los tres matrix-free están en ~770-786s **independientemente de $k$** — la FWHT domina y el core $k^2$ es ruido en el reloj.

A $d=128$: una GEMM de 128×128 son 2M FLOPs en tensor cores, con los datos en registro. La FWHT son 7 pasadas con acceso mariposa, sin tensor cores, memory-bound. No hay competición.

Esto no invalida nada — pero **es tu propio $n_0$ de la sección 2.3 de la tesis, medido**. Y dice que a $d=128$ estás por debajo del punto de inflexión.

## Y eso ataca directamente el punto 3

> *"Podemos expandir a $d=4096$ u $8192$ manteniendo $k=256$."*

Dos problemas.

**El coste no desaparece, se mueve.** La FWHT es $O(d\log d)$ **por token y por capa**. A $d=8192$ son 13 pasadas sobre 8192 elementos. Los parámetros no crecen; el cómputo y las activaciones sí. Tu propia tabla sugiere que la FWHT pierde en reloj a $d$ pequeño — necesitas saber **dónde cruza**.

**El ratio, no el valor absoluto.** $W = H_{[:,:k]}\,C\,H_{[:k,:]}$ tiene **rango exactamente $k$**. Lo que controla la expresividad no es $k$, es $k/d$:

| | $k/d$ |
|---|---|
| Tu experimento ($d{=}128, k{=}64$) | **50%** |
| Tu extrapolación ($d{=}8192, k{=}256$) | **3%** |

Son regímenes 16× distintos. El dato que tienes no soporta la extrapolación que haces. Y tu propia serie lo insinúa: k32 ($k/d$=25%) ya pierde 0.1, k16 (12,5%) pierde 0.22. **La calidad se degrada rápido al bajar el ratio, y tú propones bajarlo a 3%.**

El experimento que lo resuelve, y cabe en tu portátil:

> Fija $k/d = 0.5$ y barre $d \in \{128, 256, 512\}$ con $k \in \{64, 128, 256\}$. Luego fija $k=64$ y barre $d$. **Si la calidad se mantiene a ratio constante y se cae a $k$ constante, tienes la ley de escalado que necesitas antes de hablar de $d=8192$.** Y de paso mides el cruce de wall time.

---

## Reframe: el suelo de embeddings

Vocab 65 × d 128, con head sin atar ≈ **16.640 params fijos** que ninguna técnica tuya comprime. Separa cuerpo de embeddings:

| | Total | Cuerpo | Loss |
|---|---|---|---|
| Denso | 116.870 | ~100.200 | 1.6762 |
| k64 | 42.764 | ~26.100 | **1.6581** |
| k32 | 24.332 | ~7.700 | 1.7735 |
| k16 | 19.724 | ~3.100 | 1.8922 |

**Cortas el cuerpo 4× gratis. 13× cuesta 0.10. 32× cuesta 0.22.**

Esa tabla es más honesta y más impresionante que "7% de un transformer" — porque el 7% está diluido por embeddings incompresibles. Y te dice dónde parar: por debajo de k32 estás optimizando el 15% del modelo.

*(En un LLM real con vocab 32K esto se invierte: los embeddings son el 90% y tu técnica toca el 10%. Es la observación de V10/V11 otra vez. Tu método necesita ir acompañado de factorización de embeddings o vocabulario grande con $d$ grande.)*

---

## "Supera al denso": 0.018, una semilla, y mejor época 1 de 40

El mejor val en la época 1, con 40 épocas. Igual que en V282. **Sigues en régimen de sobreajuste**, y ahí cualquier reducción de capacidad ayuda hasta que te pasas — que es exactamente la forma de tu curva: 42K mejor que 117K, 24K peor, 20K mucho peor.

O sea: has encontrado el **óptimo de capacidad para Tiny Shakespeare**, no una propiedad de Walsh. La historia del "regularizador perfecto" no hace falta; la aburrida explica los cuatro puntos.

Y 0.018 con una semilla, cuando el checkpoint óptimo cambia entre época 1 y 2, no es distinguible de ruido.

**Pero la versión interesante de tu afirmación es comprobable y es tu §1b:**

> Si Walsh regulariza de verdad, la ventaja debe **crecer al reducir datos**. Barre 1%, 10%, 25%, 100% de Tiny Shakespeare. Denso-117K vs Walsh-k64. Si la brecha se abre cuando hay menos datos y se cierra o invierte al 100%, has demostrado el mecanismo.

Ese es el eje barato que tu hardware sí puede recorrer y que casi nadie mide, porque todo el mundo con GPUs mira el eje de parámetros.

---

## Tu programa ha convergido en una sola pregunta, y la has hecho tres veces

| | Estructura fija | Core aprendido |
|---|---|---|
| V63 | DCT | $k\times k$ |
| V283 | Walsh | $k\times k$ |
| Conos | geometría 2D | 4 params |

Y en las tres sigue abierta la misma cuestión: **¿importa la base concreta, o solo la reducción de grados de libertad?**

Fíjate en que ya tienes evidencia parcial: DCT en MNIST y Walsh en LM funcionan **más o menos igual de bien**. Dos bases distintas, dos dominios distintos, resultado similar. Eso apunta a la hipótesis aburrida.

Una línea la resuelve:

```python
Q, _ = torch.linalg.qr(torch.randn(d, d))   # fija, no entrenada
W = Q[:, :k] @ C @ Q[:k, :].T
```

Si empata con Walsh: no es la base, es **rango-$k$ con base gratis**. Y eso sigue siendo un buen resultado — $k^2$ params frente a los $2dk$ del low-rank aprendido, o sea **4× menos parámetros para el mismo rango** a $d=128,k=64$. Pero es un titular distinto y más defendible.

Si Walsh gana: la base importa, y entonces la pregunta interesante es *por qué esa* — y conectas con V290, donde la permutación TSP cambiaba la compresibilidad por un factor 2. Ahí tienes toda una línea.

---

## Qué correría, en orden

1. **Base ortogonal aleatoria.** Una línea. Resuelve la pregunta central de tu programa.
2. **Barrido de datos** (1/10/25/100%). Convierte "regularizador" de metáfora en medición, y es exactamente tu §1b.
3. **Ratio $k/d$ constante vs $k$ constante**, con wall time. Es el prerequisito de cualquier afirmación sobre $d=8192$.
4. Un corpus donde no sobreajustes (TinyStories, enwik8) para separar capacidad de regularización.

Los cuatro caben en tu portátil y ninguno necesita baseline de nadie.

---

Y una observación sobre el arco: V278 tenía un teorema disfrazado de experimento, V279 y V280 tenían fuga, V281 la arregló, V282 trajo el baseline, V283 tiene aritmética verificable y una curva de capacidad limpia.

Seis experimentos y el rigor subió de forma monótona. Eso es el bucle funcionando — y es más valioso que cualquiera de los resultados individuales, porque es lo que hace que el siguiente valga algo.