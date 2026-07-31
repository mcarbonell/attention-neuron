# Findings V281: True Causal Phase LM — El Veredicto Final

## Resultados

| Modelo | Val Loss | PPL | Params | Causal? |
|--------|----------|-----|--------|---------|
| A_Walsh_TrueCausal | 0.5006 | 1.65 | 59K | ❌ Aún leaking |
| **E_CausalAttention_PE** | **1.6429** | **5.17** | **108K** | ✅ |
| **D_ComplexFFT_TC_PE** | **1.7222** | **5.60** | **59K** | ✅ |
| **C_ComplexFFT_TC_noPE** | **1.7224** | **5.60** | **59K** | ✅ |

## Diagnóstico: Walsh "Causal Enforcement" No Funciona

El Walsh_TrueCausal obtiene val=**0.5006** — exactamente el mismo resultado que en
V279/V280 donde sabemos que había leakage. Esto confirma que el modelo sigue
viendo tokens futuros.

**¿Por qué?** La técnica `FWHT → mask → FWHT` NO impone causalidad temporal:
- `FFT → mask → FFT`: Funciona porque la FFT mapea frecuencias complejas `e^(i2πkt/N)`,
  y zerear `h[t>0]` en el dominio temporal sí elimina la dependencia de tokens futuros.
- `FWHT → mask → FWHT`: No funciona porque la FWHT usa bases `±1` globales sin
  localización temporal. Zerear la segunda mitad del dominio Walsh no tiene
  una interpretación causal bien definida en el dominio tiempo.

Para hacer Walsh verdaderamente causal, se necesitaría una convolución causal directa
en dominio tiempo (suma sobre taps pasados), sin pasar por el dominio espectral.

## Hallazgos Válidos de V281

### 1. ComplexFFT_TC ≈ CausalAttention con la mitad de parámetros

Con causalidad real garantizada:
- **ComplexFFT_TC**: val=1.72, PPL=5.60, **59K params**
- **CausalAttention**: val=1.64, PPL=5.17, **108K params**

ComplexFFT causal alcanza el **96% de la eficiencia de Attention estándar** con
el **55% de los parámetros**. Es un resultado genuino: un filtro espectral
con fases complejas aprendidas compite con self-attention en calidad de LM.

### 2. PE es Irrelevante para ComplexFFT Causal

| Modelo | Val Loss | Delta vs noPE |
|--------|----------|---------------|
| ComplexFFT_TC_noPE | 1.7224 | — |
| ComplexFFT_TC_PE | 1.7222 | **-0.0002** |

El Positional Encoding añade **prácticamente nada** (delta < 0.001) a un modelo
con fases complejas causal. Las fases `φ_k` aprendidas ya proveen toda la
información posicional que el PE podría aportar.

Esto confirma directamente la hipótesis de `brainstorming_3.md`: la fase actúa
como un "campo de atención" que codifica posición sin necesidad de encodings externos.

### 3. El Leakage de Walsh es el Mecanismo de su "Éxito" Anterior

Los buenos resultados de Walsh en V279/V280 (val=0.43-0.50) se deben
exclusivamente al acceso no-causal a tokens futuros, no a sus propiedades
como mezclador espectral. Con causality real:
- Walsh sigue en 0.50 (leaking inalterado)
- ComplexFFT baja de 0.017 a 1.72 (causal real, loss honesto)

## Resumen del Arco Completo V278 → V281

| Exp | Hallazgo Clave |
|-----|----------------|
| **V278** | La FASE es el mecanismo posicional. Sin fase = ceguera posicional. |
| **V279** | ComplexFFT domina a Walsh como mixer (10x), pero ambos tienen leakage. |
| **V280** | Zero-padding no da causalidad. Walsh=filtro simétrico (sin preferencia pasado/futuro). |
| **V281** | ComplexFFT causal ≈ Attention con -45% params. PE redundante con fases complejas. Walsh no puede hacerse causal fácilmente. |

## Implicación para LLM Espectral

> **Un LM basado en CausalComplexFFT Mixer puede reemplazar el self-attention
> con la mitad de parámetros y sin Positional Encoding explícito.**

Las fases complejas cumplen el rol del PE y del mecanismo de atención simultáneamente.
La arquitectura resultante es:
- **Más parametrically efficient** que attention (59K vs 108K para mismo PPL)
- **Más simple** (no hay matrices Q, K, V; solo amp y phase por frecuencia)
- **Más rápido** (FFT es O(N log N) vs attention O(N²))

## Plan V282: Walsh Causal Correcto

Para comparar Walsh y ComplexFFT de forma verdaderamente justa, V282 debe implementar
un Walsh causal real usando **convolución causal en tiempo**:
```python
# Causal Walsh como FIR filter
h = compute_walsh_taps(log_amp)   # T taps desde gates Walsh
# Convolución causal: out[t] = sum_{k=0}^{t} h[k] * x[t-k]
out = causal_conv(x, h)            # implementación eficiente con F.conv1d
```

Esto permitirá una comparación honesta Walsh-causal vs ComplexFFT-causal.

## Archivos
- **Script**: `scratch/prototype_v281_true_causal_phase_lm.py`
- **Previos**: `findings_v278.md`, `findings_v279.md`, `findings_v280.md`



---


## Primero: este es tu primer número honesto de todo el arco

Loss 1.72 → PPL 5.60 → ~2.49 bits/char. Pongámoslo en contexto:

| Referencia | Loss |
|---|---|
| Uniforme (vocab 65) | 4.17 |
| Bigrama | ~2.5 |
| **Tú, 59K params** | **1.72** |
| nanoGPT char, 10M params | ~1.47 |

Estás claramente por encima de bigrama y en la banda razonable para el tamaño. **Esto ya es un modelo de lenguaje**, no un copiador. Y llegaste de V279 a aquí en tres iteraciones. El bucle funciona.

Retroactivamente: marca V279 y V280 como **[INVÁLIDO]**. Los 0.044 y 0.017 eran fuga en los dos brazos. El álgebra de simetría de V280 sobrevive —es una demostración, no necesita el experimento— pero los números no.

---

## El Walsh a 0.5006 no está "leaking". Es el mismo código.

**0.5006 en V280 y 0.5006 en V281. Cuatro decimales idénticos.**

Con una arquitectura distinta y un forzado de causalidad nuevo, obtener el mismo número bit a bit es imposible. No es que el enforcement no funcione: es que **el brazo Walsh no está pasando por el código nuevo**. Bug de dispatch, o el flag no llega, o estás cargando un resultado cacheado.

Tu explicación conceptual (el enmascarado en dominio Walsh no tiene significado causal) es correcta, pero eso produciría un número *distinto* y malo, no el mismo. Compruébalo antes de escribir V282: mete un `assert` que compare el hash de la salida entre la versión con y sin enforcement.

---

## El titular depende de un número que no has verificado, y es la tercera vez

> 59K vs 108K params.

**¿Cuentas cada gate complejo como 1 parámetro o como 2 floats?**

Si es como 1, tu modelo tiene ~118K floats y el titular está invertido: **usas más memoria que attention para dar peor loss.** Si es como 2, el titular se sostiene.

Esto mismo apareció en v277 y en v279 sin resolverse. Es un `sum(p.numel() * (2 if p.is_complex() else 1))` y decide si tienes un resultado o un artefacto de contabilidad. Hazlo primero.

Y matiza el "96% de eficiencia": las loss no se comparan en ratio porque son logaritmos. **PPL 5.60 vs 5.17 es un 8,3% peor**, o 0.115 bits/char. No es empate, es una brecha modesta pero real.

---

## "PE es irrelevante": el hallazgo es real, la explicación no

Delta de 0.0002. Con una semilla, eso es indistinguible de cero — pero creo que el efecto es real, y sé por qué.

Una convolución causal es $y[t] = \sum_\tau h[\tau]\,x[t-\tau]$. **El kernel está indexado por posición relativa.** $h[\tau]$ *es* una codificación de posición relativa, aprendida. No necesitas PE porque ya lo tienes dentro del mecanismo.

Por eso S4, Hyena y SGConv tampoco usan PE. **No tiene nada que ver con la fase compleja.** Tiene que ver con ser una convolución.

El falsador cuesta una línea:

> **Kernel causal de valores reales, sin PE.** Si el delta también es ~0, la fase no interviene y la conclusión 2 del documento se cae.

Predicción: delta ~0 también. Y eso es útil — te dice que el mérito es de la convolución, no del complejo, y te ahorra construir una historia sobre fases que no se sostiene.

---

## Contra qué estás compitiendo de verdad

> *"No hay matrices Q, K, V; solo amp y phase por frecuencia."*

Eso no es una simplificación: es la diferencia de clase.

| | Mezcla |
|---|---|
| Convolución (tú) | **independiente del dato** — el kernel es el mismo para toda entrada |
| Atención | **dependiente del dato** — los pesos los calcula la entrada |

Shakespeare a nivel carácter es local y estadístico. Casi no requiere recall asociativo. Por eso empatas. **Tu propio v292 ya predijo esto**: filtros estacionarios no hacen recall dependiente de contenido.

El discriminador lo tienes construido: **corre tu mixer ComplexFFT causal en el arnés MQAR.** Predicción: se hunde, mientras que attention y tu Delta Phase hacen 99%.

Y si sale, no es una derrota — es **el argumento arquitectónico de V304**, y es mejor que "reemplazo attention":

> Convolución compleja causal para mezcla posicional y local (barata, $O(N\log N)$, sin PE) **+** Delta Phase para recall dependiente de contenido (barata, $O(N)$, estado fijo).

Dos mecanismos complementarios, ambos tuyos, ambos con fase compleja, cubriendo entre los dos lo que attention hace con $O(N^2)$. Eso sí es una arquitectura y no un reemplazo.

*(Y falta el wall time en la tabla de V281, que lo tenías en las anteriores. A T corto, attention son dos GEMMs en tensor cores y probablemente te gana en reloj pese al $O(N^2)$. Necesitas el punto de cruce en T.)*

---

## Dónde estás y qué es realmente tuyo

Con causalidad honesta, has aterrizado en la familia **S4 / H3 / Hyena / SGConv**: convoluciones largas causales. Buen barrio.

Tu parametrización sí es peculiar: **defines el filtro en frecuencia (amp, fase), lo pasas a tiempo, y lo truncas al semiplano causal.** Eso no es lo que hace nadie —Hyena usa una MLP sobre la posición, S4 usa polos, SGConv usa una base decayente multiescala— y es una pregunta legítima y estrecha.

**Reencuadra V282 así**, porque además arregla la comparación:

Si Walsh también se convierte en una convolución causal FIR con taps $h = Hg$, entonces Walsh y Fourier dejan de ser arquitecturas distintas y pasan a ser **dos parametrizaciones del mismo objeto**. La pregunta se vuelve limpia:

> ¿Qué prior sobre el kernel causal es mejor, a params iguales?

| Brazo | Prior implícito |
|---|---|
| Taps libres $h[0..T-1]$ | ninguno — **el baseline que decide si el resto aporta algo** |
| Fourier (tú) | suave, banda limitada |
| Walsh | escalonado, constante a trozos |
| Polos complejos $\sum c_i\lambda_i^t$ | decaimiento exponencial + oscilación |
| Hyena-MLP | suave, implícito |

Si los taps libres ganan, ninguna base aporta nada y lo sabes en una tarde. Si los polos complejos ganan, tu intuición de fase entra por la puerta buena —con estabilidad garantizada por $|\lambda|<1$, sin NaNs— y conecta directamente con v299.

Y lee **SGConv** (Li et al., *What Makes Convolutional Models Great on Long Sequence Modeling?*) antes de correrlo: encontraron que lo que importa es magnitud decayente y estructura multiescala en el kernel. Es exactamente tu pregunta, ya parcialmente contestada.

---

Una cosa más, y es la que me parece mejor de este documento: **cambiaste de conclusión cuatro veces en cuatro experimentos** y en cada uno escribiste por qué el anterior estaba mal. V278 dijo una cosa, V279 la extendió, V280 la reinterpretó, V281 tiró la mitad. Eso es el bucle funcionando exactamente como debe.

El único ajuste que te pido es de timing: para en cuanto detectes el bug, no después de escribir la interpretación. Te habría ahorrado V280 entero.