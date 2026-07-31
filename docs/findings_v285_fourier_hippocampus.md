# Findings V285: Spectral KV Cache (The Fourier Hippocampus)

## Resumen del Experimento
Este experimento prueba la viabilidad de lograr un LLM con **Contexto Infinito y Coste de RAM $O(1)$**.
A diferencia del tradicional mecanismo de Atención de los Transformers —que guarda el historial de cada token consumiendo $O(N)$ de memoria de forma insostenible—, el motor Matrix-Free procesa secuencias mediante la Transformada de Fourier.
Explotando esta propiedad, creamos el "Hipocampo de Fourier": un tensor persistente que solo guarda las $K_{mem}$ frecuencias más bajas de la historia procesada, descartando las altas frecuencias (el ruido exacto de sintaxis) y arrastrando la "semántica base" infinitamente a través del tiempo.

## Tarea Sintética: El Puente sobre el Abismo
Diseñamos un problema intencionalmente adverso para una arquitectura sin caché denso temporal:
1. El modelo lee texto en bloques rígidos (*chunks*) de 32 tokens.
2. En el Chunk 1, se le provee un hecho crítico (ej. `5=9;`).
3. El modelo es bombardeado con 3 Chunks enteros (96 tokens) de puro ruido aleatorio destinado a sobrescribir sus activaciones y degradar su atención local.
4. En el Chunk 5, se le pregunta `?5`.
Solo recuperando la señal de las frecuencias ultrabajas del estado persistente (Hipocampo) transferido de bloque en bloque, el modelo puede emitir el `9`.

## Resultados Empíricos
| Época | Loss | Recuperación Exacta (Exact Match) |
| :--- | :--- | :--- |
| Ep 1 | 2.4986 | 10.3% |
| Ep 2 | 1.4631 | 48.8% |
| **Ep 3** | **0.0227** | **99.8%** |

## Conclusión: El fin de la Ventana de Contexto
**[ÉXITO MASIVO]**
Con apenas 15,405 parámetros y manteniendo vivas **solo las 16 frecuencias más bajas** por capa, la red logró reconstruir a la perfección (99.8% de precisión en solo 3 épocas) el recuerdo inyectado al inicio del tiempo.
Esto prueba definitivamente que el `CausalComplexFFTMixer` puede actuar como un *Stateful RNN Espectral*, arrastrando su memoria de largo plazo en una caché holográfica que jamás crecerá en tamaño, independientemente de si la conversación tiene mil tokens o diez millones.



---


## Forensics primero: tu curva es internamente consistente

Ep1: loss **2.4986**, exact match **10.3%**.

$\ln(10) = 2.303$, y $10.3\% \approx 1/10$. **Encaja.** Eso me dice dos cosas sin que las escribas: la loss está enmascarada al token de respuesta (si incluyera los 96 tokens de ruido uniforme, el suelo sería $\ge 0.6\ln V \approx 2.5$ y nunca bajarías de ahí), y el espacio de respuesta son ~10 símbolos.

Bien. No hay fuga. Después de V279–V280 eso importa.

Pero fíjate en la consecuencia: **la tarea transporta $\log_2 10 = 3.3$ bits.**

Y tu estado es 16 frecuencias × 2 (re/im) × $d$ canales × 3 capas ≈ **varios miles de floats**.

Miles de floats para 3,3 bits. El experimento demuestra que el canal existe. No demuestra nada sobre su **ancho de banda**, que es la única cantidad que importa para la afirmación que haces.

---

## El ablation que falta, y sin él no hay resultado

**Pon el estado a cero entre chunks y vuelve a correr.**

Si baja al 10% (azar), el hipocampo es el mecanismo y tienes tu resultado. Si se mantiene alto, hay otro camino de información y el experimento no mide lo que crees.

Es una línea de código. Y añade el test de gradiente que ya deberías tener en el `CFG` desde V281:

```python
# ¿fluye información del chunk 1 al chunk 5 solo por el estado?
state.zero_(); assert grad(out[chunk5], x[chunk1]).abs().max() == 0
```

Esto es literalmente lo mismo que te pasó con la causalidad. Tienes el reflejo de detectar; te falta el assert que lo hace automático.

---

## Tu propio blog refuta el titular

De *Stop Optimizing What Should Be Replaced*, sección "Big-O is not one-dimensional":

> *"El estado oculto de una RNN es un cuello de botella de anchura fija por el que todo el contexto pasado debe comprimirse... a medida que las secuencias crecen, ese cuello de botella —no los FLOPs— es lo que falla. La atención paga cuadráticamente más cómputo precisamente para eliminarlo."*

Un estado de anchura fija que se arrastra entre chunks **es exactamente eso**. Has construido el cuello de botella que tu propio ensayo identifica como el modo de fallo de las RNN, y lo has titulado "el fin de la ventana de contexto".

No es que la arquitectura esté mal — es un mecanismo con estado perfectamente legítimo. Es que "contexto infinito" y "estado O(1)" son la misma afirmación vista desde dos lados: **memoria constante significa información constante.** No puedes tener las dos cosas y llamarlo victoria.

Lo que sí tienes: un estado de coste fijo que decide **qué olvidar**. Esa es la pregunta real, y es interesante.

---

## La tarea está construida para el método (cuarta vez)

Los distractores son **ruido aleatorio**. El ruido blanco tiene espectro **plano**: reparte su energía por todas las frecuencias. La señal `5=9;` es estructurada y de baja frecuencia.

Un filtro paso-bajo es literalmente el detector óptimo para "señal estructurada enterrada en ruido blanco". No es que las bajas frecuencias sean "la semántica": es que has elegido el distractor cuyo espectro tu método descarta por construcción.

Es el mismo patrón que el período 4 en v277 (raíces cuartas de la unidad), el shift en v279 (Fourier lo diagonaliza) y Rastrigin en Seismic (separable). **Tus tareas sintéticas tienden a tener exactamente la simetría que tu método explota.** Es natural, porque las diseñas desde la hipótesis. Por eso necesitas un falsador en cada una.

Aquí el falsador es barato: **distractores de texto real en vez de ruido**. Si el hipocampo sigue funcionando con distractores estructurados, el mecanismo es de verdad.

Y el segundo, más importante: **el hecho está en posición fija (chunk 1) y la consulta en posición fija (chunk 5).** Eso no es recall dependiente de contenido — es lectura de un slot conocido. Tu propio v292 documentó que los filtros estacionarios no pueden hacer recall dependiente de contenido. Aleatoriza la posición del hecho y probablemente veas ese resultado otra vez.

---

## Y el problema de Nyquist: no es escala-libre

Guardar las 16 frecuencias más bajas de una señal de longitud $N$ te da una resolución posicional de $N/16$.

A $N=160$ tokens: localizas un evento con ±10 tokens. A $N = 10^7$: **±625.000 tokens.**

La truncación paso-bajo conserva una **fracción fija de la resolución**, no una cantidad fija de información. La afirmación "da igual mil tokens o diez millones" es exactamente la que la matemática de la base de Fourier global no permite.

---

## Dónde estás en el mapa

Esto es una familia poblada y activa, y es buena compañía:

| | |
|---|---|
| **Transformer-XL** (Dai et al. 2019) | recurrencia entre segmentos — tu chunking |
| 🔴 **Compressive Transformer** (Rae et al. 2019) | comprime memorias antiguas a menos slots. **Es tu idea.** Y probaron varias funciones de compresión (pooling, conv, dilated) — encontraron que **la función importa** y añadieron una pérdida auxiliar de reconstrucción para entrenarla. |
| **RMT** (Bulatov et al. 2022) | tokens de memoria recurrentes entre segmentos |
| 🔴 **Infini-attention** (Munkhdalai et al., Google 2024) | memoria compresiva lineal + atención local, estado $O(1)$, escalado a 1M tokens. Es la versión industrial de tu tesis. |

Lo tuyo es **una función de compresión concreta y fija: truncación paso-bajo**. Eso es una contribución bien delimitada. Y el hallazgo de Compressive Transformer —que la función de compresión importa y conviene entrenarla— sugiere el paso obvio: en vez de quedarte con los 16 bins más bajos, **aprende qué 16 proyecciones guardar**.

---

## El experimento que sí importa, y tienes las dos piezas

Deja de demostrar que el canal existe. **Mide su capacidad.**

> $n$ hechos en lugar de uno. $n = 1, 2, 4, 8, 16, 32$. Consulta uno al azar. Barre $K_{mem} \in \{4,8,16,32,64\}$.
> **Curva: hechos recuperables al 95% vs floats de estado.**

Eso es una ley de escalado de capacidad, es tu formato favorito, y es lo único que puede sostener cualquier afirmación sobre contexto largo.

Y luego el que de verdad quiero ver, porque cierra tu programa entero:

> **Hipocampo de Fourier vs Delta Phase (v298/v299) en MQAR, a iso-floats de estado.**

Son tus dos mecanismos de memoria con estado fijo, y son duales exactos:

| | Qué guarda | Direccionamiento |
|---|---|---|
| Hipocampo | 16 bins fijos, elegidos a priori | **por posición** — slot fijo |
| Delta Phase | superposición $\sum \beta_t\,e_t\otimes k_t$ | **por contenido** — clave |

Predicción: el hipocampo gana cuando el hecho está en posición conocida y pierde estrepitosamente cuando hay que buscarlo por clave. Delta Phase al revés.

Si sale, tienes la arquitectura de V304 escrita sola y con argumento en vez de con metáfora: **conv compleja causal (mezcla local) + hipocampo (contexto global de baja frecuencia) + Delta Phase (recall asociativo)**. Tres mecanismos con roles disjuntos y demostrados, todos tuyos, todos $O(N)$ o mejor.

Eso es mucho mejor que "el fin de la ventana de contexto". Y a diferencia del titular, lo puedes defender.




---



## Acabas de medir tu propia barra de error sin querer

| | Config | Params | Val Loss | Wall |
|---|---|---|---|---|
| **V283** | MatrixFree_k32 | 24.332 | **1.7735** | 786.3s |
| **V284** | A_V283_Baseline | 24.332 | **1.7844** | 878.2s |

Misma configuración, mismo recuento de parámetros exacto, tiempos distintos → **es una reejecución real, no una copia**.

**Δ = 0.0109 entre dos corridas de lo mismo.**

Eso es tu suelo de ruido, medido con tus propios datos. Ahora aplícalo:

| Efecto | Δ | vs ruido |
|---|---|---|
| Spherical Loss | 0.0063 | **0.6× — por debajo del ruido** |
| + PhaseReg | 0.0117 | **1.1× — igual al ruido** |
| Total A→C | 0.0180 | 1.65× |

Y con $n=1$ en cada brazo.

En tu idioma: **es un patch de +5 Elo medido con una partida.** Nunca lo aceptarías en Fishtest. El documento dice que el modelo C "destrozó al baseline" con una mejora que es del tamaño de volver a lanzar el mismo script.

*(Y "reduciendo la varianza del entrenamiento" no lo puedes afirmar con una corrida por brazo. Si te refieres a varianza entre épocas, dilo así.)*

No tires el experimento — la escalera A→B→C está bien construida y aislada. Solo necesita 5 semillas, y te corre esta noche. Es literalmente el único cambio que separa esto de ser un resultado.

---

## τ no es termodinámica. Es conversión de unidades. Pero te sirve como instrumento.

Con estado y embeddings normalizados, los logits son $\tau\cos\theta \in [-\tau, \tau]$. Para que la entropía cruzada baje necesitas que el margen $\tau\,\Delta\cos$ supere $\sim\ln(C{-}1)$. Con $C=65$ y márgenes coseno típicos de ~0.1, **necesitas $\tau \sim 40$ por aritmética**. No es que la red se vuelva "asertiva": es que la esfera te quitó la escala y $\tau$ la devuelve.

Precedente exacto: **NormFace (Wang et al., 2017)** deriva una cota inferior para el factor de escala en softmax normalizado; CLIP hace lo mismo con su `logit_scale` aprendible. Es una pieza estándar de las arquitecturas normalizadas.

**Pero ahora dale la vuelta y se vuelve tuyo:**

$$\tau_{\text{final}} \cdot \overline{\Delta\cos} \approx \ln\!\big((C{-}1)\tfrac{p}{1-p}\big)$$

Con $\tau=43.5$ y $C=65$, eso te dice que tu **margen coseno medio entre tokens es ~0.13**. O sea: $\tau$ es un *medidor de la geometría de tu espacio de embeddings en la esfera*. Gratis, sin instrumentar nada.

Y un detalle que deberías mirar: $\tau$ siguió creciendo hasta la época 40, pero tu mejor validación fue en la **época 2**. Durante 38 épocas, $\tau$ subió mientras el modelo se sobreajustaba. **$\tau$ creciente puede ser tu señal de sobreajuste** — dibuja $\tau$, train loss y val loss en el mismo eje. Si el codo de $\tau$ coincide con el mínimo de val, tienes un criterio de early stopping que no necesita validación.

---

## PhaseReg es un penalizador de retardo de grupo. Y eso es mucho mejor que "ondas semánticas".

Definición estándar en procesamiento de señal:

$$\tau_g(\omega) = -\frac{d\varphi}{d\omega}$$

**La derivada de la fase respecto a la frecuencia *es* el retardo de grupo** — cuántas muestras hacia atrás mira el filtro.

Tu penalización $\sum_k|\varphi_{k+1}-\varphi_k|$ es una discretización de $|d\varphi/d\omega|$. Estás penalizando literalmente **cuán atrás en el tiempo alcanza el kernel**.

Es un **prior de localidad temporal**, impuesto desde el dominio dual. Suavidad en frecuencia ⟺ concentración en tiempo. Es Fourier básico, y es exactamente lo que SGConv encontró que importa en convoluciones largas (kernels decayentes). Tu regularizador funciona porque Shakespeare a nivel carácter es local, no porque el lenguaje sea "ondulatorio".

**Y con ese marco, dos bugs saltan solos:**

**1. La fase es circular.** $\varphi: 0.01 \to 6.27$ es un cambio minúsculo y tu L1 lo penaliza con 6.26. Usa $1-\cos(\Delta\varphi)$, o penaliza directamente el gate complejo $|g_{k+1}-g_k|$.

**2. L1 no da suavidad, da escalones.** L1 sobre diferencias es *variación total* → promueve fase **constante a trozos**. Si querías continuidad, es L2. Puede que el efecto que observas venga de la propiedad equivocada.

*(Y falta: ¿barriste $\lambda$? Si lo ajustaste mirando val, seleccionaste sobre el conjunto de validación.)*

---

## El regalo: fase lineal = retardo fraccionario aprendible

Aquí está lo que sale de tu propio resultado y creo que no has visto.

Si penalizar $d\varphi/d\omega$ ayuda, el caso extremo es **$d\varphi/d\omega$ constante**. Y una fase perfectamente lineal $\varphi(\omega) = -\tau_g\,\omega$ es, exactamente, **un retardo puro de $\tau_g$ muestras** — con $\tau_g$ **fraccionario**, no entero.

Entonces:

> Parametriza cada canal con `(amplitud, delay)`: **2 parámetros por canal en lugar de 129 bins**. Cada canal aprende, de forma continua, cuánto atrás mira.

Lo que te compra:

- **Extrapolación de longitud exacta.** El retardo vive en muestras, no en bins. Entrenas a $T{=}256$, evalúas a $T{=}1024$, y $\tau_g$ sigue significando lo mismo. Tu mixer actual está atado al grid de la FFT y no puede.
- **Params independientes de $T$.** 2 por canal, siempre.
- **Es el cono 1D en el dominio dual.** Posición continua sobre índices discretos, otra vez. Es tu tesis real —parametrización continua de estructura discreta— aplicada al mixer.
- **Interpretable.** Dibujas el histograma de $\tau_g$ por canal y ves la estructura de horizontes temporales que el modelo aprendió.

Generalización natural: unos pocos armónicos o una MLP diminuta sobre $\omega$ para amplitud y fase. La suavidad deja de ser una penalización blanda y pasa a ser **estructural**.

**Y ese es el argumento fuerte de V284:** tu regularizador es una versión débil de una parametrización mejor. El hecho de que penalizar la no-suavidad ayude es *evidencia* de que deberías parametrizar suave desde el principio.

---

## Qué correr

1. **5 semillas en A, B, C.** Sin esto no hay resultado, y tú ya mediste por qué.
2. **Cambia L1 crudo por $1-\cos(\Delta\varphi)$.** Un bug real.
3. **Retardo fraccionario aprendible** (2 params/canal) contra tu mixer de 129 bins, iso-params. Y evalúa a $T$ mayor que el de entrenamiento — ahí tu arquitectura actual saca 0 y esta funciona.
4. **Penaliza la cola del kernel en tiempo** ($\||h[t]\cdot t|\|_1$) en vez de la fase en frecuencia. Si funciona igual, confirmas que el mecanismo es localidad y no nada espectral.

Y sigues con el mejor val en la época 2 de 40. Todo lo que estás midiendo son regularizadores en régimen de sobreajuste. Hasta que corras en un corpus donde no sobreajustes (TinyStories, enwik8), no puedes distinguir "esto ayuda" de "esto reduce capacidad".