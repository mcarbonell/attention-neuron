## V87 → V87b → V87c es el mejor artefacto metodológico de todo tu corpus

Y quiero ser preciso sobre por qué, porque creo que no lo has visto.

En V87 hiciste una afirmación máxima ("hemos roto la ley de escalado cuadrático"). En V87b la pusiste a prueba con controles iso-parámetro. En V87c **diseñaste el experimento que podía matarla** —la matriz cruzada de bases— añadiste un baseline de PCA que no te convenía, y cuando FWHT sólo ganó en su base nativa, lo etiquetaste `[ANCLA-NEGATIVO]` y reescribiste la hipótesis.

Eso es el ciclo completo. Conjetura → control → refutación → hipótesis refinada, en cuatro experimentos, documentado con la sección de "Amenazas a la Validez" escrita **antes** de que nadie te la pidiera.

Si tienes que enseñar un solo documento de los 297 a alguien que decida sobre ti, es V87c. No porque el resultado sea espectacular, sino porque demuestra que tu instrumento se autocorrige.

Ahora, el error de V87, que es serio y tiene una versión buena debajo.

---

## La contabilidad correcta de la Mega-Layer

`W = H_out[:,:K] · C · H_in[:K,:]` tiene **rango ≤ K = 64**. Comparar eso con una densa 16384×16384 no es comparar dos implementaciones del mismo objeto: es comparar un operador de rango 64 con uno de rango 16384. El "65.540×" mide la diferencia entre dos clases de funciones, no una compresión.

Pero cuando lo comparas contra los **otros** métodos de rango 64, sale algo mejor de lo que has contado:

| | Params entrenables | Almacenamiento de pesos | Ops / vector | Rango |
|---|---|---|---|---|
| Densa | 268 M | **1,07 GB** | 2,68 × 10⁸ | 16384 |
| Rango-64 aprendido (UV) | 2,1 M | 8,4 MB | 2,1 × 10⁶ | 64 |
| Proyección aleatoria + core | 4.096 | 8,4 MB (hay que guardar P) | 2,1 × 10⁶ | 64 |
| **Espectral (FWHT)** | **4.096** | **16 KB** | **4,6 × 10⁵** | **64** |

Dos cosas que salen de ahí y que sí son defendibles:

**1. Con 4.096 parámetros, una factorización aprendida no puede permitirse ni rango 1.** Rango 1 sobre 16384×16384 cuesta 32.768 params. Tú consigues rango 64 con 4.096. Contra rango-64 aprendido a igualdad de rango: **512× menos parámetros**. Ése es el número honesto, y es el mismo mecanismo de tu V4 — base fija de alto rango × modulación pequeña.

**2. Tu ventaja real es el tráfico de pesos, no los FLOPs.** Y da la casualidad de que es el cuello de botella que importa: la decodificación autoregresiva a batch 1 está limitada por ancho de banda leyendo pesos. 16 KB caben en L2; 1 GB no cabe en ningún sitio. La proyección aleatoria tiene tus mismos 4.096 entrenables **pero tiene que almacenar 8,4 MB de P**. Ahí ganas de verdad, y lo dijiste de pasada en una frase.

**Y el diagnóstico que se te escapó:** teóricamente hay 580× menos operaciones. Mediste 40×. Has perdido un factor **14,5×** en la implementación — la FWHT hace log N pasadas con accesos a memoria en zancada y destroza la caché, mientras que la densa corre en BLAS a alta intensidad aritmética. Ese 14,5× es la distancia entre tu asintótica y tu reloj, y **en GPU se te come entero**, porque la densa pasa a tensor cores y la FWHT sigue limitada por ancho de banda.

Reencuadre correcto de todo el hilo: *"Con presupuesto de 4.096 parámetros, la síntesis espectral es el único método que alcanza rango 64 con almacenamiento de pesos nulo. Frente a la proyección aleatoria a igualdad de rango: 512× menos memoria estática y 4,5× menos operaciones."*

Eso es publicable. Lo de la ASI no.

---

## V88 ya lo resolviste tú. En V298.

Esto es lo que más me ha gustado encontrar.

V88 falla y necesitas multiplicar la aguja por **150×** ("saliencia de amígdala") para recuperarla. Eso no es un truco biomimético: es la confesión de que la escritura no tiene corrección de error. Estás integrando chunks por producto externo acumulativo — **Hebbiano puro**. Y la diafonía de ~100 chunks ahoga cualquier item que no venga pre-marcado.

Tu propio ledger:

| | Mecanismo | Resultado |
|---|---|---|
| v293 | Hebbiano puro | 18,94% |
| **v298** | **Regla Delta** | **99,95%** |

`M ← M(I − βkkᴴ) + βvkᴴ` resta lo que ya está almacenado antes de escribir. **Eso es exactamente lo que hace innecesario el 150×.**

V88 no es una línea nueva que retomar. Es v293 con otro nombre, y la corrección ya está escrita en tu repo. Rehacerlo con la regla delta cuesta cambiar una línea de la actualización, y convierte un "[ÉXITO MASIVO]" dudoso en un resultado limpio de capacidad de memoria — que además alimenta directamente el paper de Delta-Phase.

*(Y la métrica correcta no es coseno 0,4861. Es tasa de recuperación del token correcto y curva de capacidad: ítems almacenables vs D, barriendo el factor de saliencia. El 150× no se elimina, se **mide**.)*

---

## Dos cortes rápidos

**V89.** Falta el único número que decide: **accuracy del córtex solo sobre las 10.000 imágenes.** Comparas 91,95% (cerebelo en su subconjunto fácil) con 88,78% (córtex en el difícil) — subconjuntos distintos, no comparables. Si el córtex solo hace 97% sobre todo, tu cascada cambia 2,2× de velocidad por 5 puntos, y eso es un trade-off, no un éxito. Y el resultado no es un punto: es la **curva accuracy vs FLOPs medios** barriendo el umbral de entropía. *(Además, 92% en MNIST tras 3 épocas es bajo para ambas vías — mira si el córtex está infraentrenado antes de sacar conclusiones.)*

**V90e.** ¿Es iso-parámetro? Si separas los pesos en real e imaginario **duplicando** el conteo, el +1% es lo esperado y no hay resultado. Si los partes manteniendo el total, sí lo hay. En v299 hiciste esta contabilidad perfectamente (2048 floats/cabeza en ambos brazos) — aplica el mismo rasero aquí.

Y si es iso-parámetro: **es la tercera vez que la representación compleja gana en tu corpus** (V90e, v298/v299, y la firma de fase de V88). Tres tareas no relacionadas, mismo patrón. Eso no es una anécdota, es tu tesis real, y es mucho más estrecha y más defendible que "representación compacta".

---

## Tu pregunta de verdad: más ideas que tiempo

No te voy a dar ideas. Te voy a dar una función de poda, y en tu idioma: te falta **ordenación de movimientos**. Estás generando ramas legales sin ordenarlas, así que el alpha-beta nunca corta.

Cuatro criterios, todos binarios:

| | ¿Ground truth? | ¿Falsable en <1 semana con tu HW? | ¿Campo saturado? | ¿Transpone con lo que ya tienes? |
|---|---|---|---|---|
| **v299 Delta-Phase** | Sí (MQAR exacto) | Sí | Parcial | **Sí** (V88, V90e, v293) |
| **Tablebases: curva memoria↔búsqueda** | **Perfecto** (verificable exhaustivamente) | Sí | **No, nadie** | Sí (meta-tesis) |
| Tabla de bases estructuradas | Controlado | Sí | Existe disperso | **Sí, con todo** |
| V79 morphing con modos ASM | Sí | Sí | No | Sí (V50, V71) |
| V87 mega-layer a escala | No | No (necesita GPU) | Sí (Monarch) | Parcial |
| **V65 LLM espectral** | No | **No** | **Sí, mucho** | No |

**V65 es la trampa.** Es la idea más excitante que me has enseñado y el peor uso posible de tus recursos: compites con equipos financiados (Diffusion-LM, Plaid, los LLM de difusión de 2025), necesitas GPU para decir algo, y no tienes ground truth. Si la persigues ahora, quemas seis meses y sales con otro [SEÑAL].

**Pero puedes matarla o validarla en dos días, sin entrenar nada.** Coge tus embeddings congelados, trunca la DCT a lo largo de la secuencia a K coeficientes, y entrena una sonda lineal para predecir el token siguiente. Grafica loss recuperable vs K, con la loss del unigrama como suelo. Si a K=25% la loss apenas sube, tienes evidencia real con la que pedir cómputo. Si se dispara, has ahorrado medio año. Eso no es explorar V65: es **decidir si merece la pena explorarla**.

Y el orden que yo seguiría:

1. **v299 terminado.** Es el único que ya está a un paso del cierre.
2. **La curva de tablebases.** Es lo único que nadie más puede hacer y tienes el código.
3. **La tabla de bases**, que es el paper que tu corpus entero ya sostiene.

Todo lo demás va a una lista de "ramas podadas con razón", con una línea cada una explicando por qué. Esa lista es un activo, no una derrota — y cuando llegue el agente autónomo o la GPU, sabrás exactamente dónde retomar.

Tienes 297 experimentos y cero resultados cerrados. El siguiente experimento no te acerca. El siguiente **cierre**, sí.