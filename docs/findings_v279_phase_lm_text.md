# Findings V279: Phase LM on Real Text (Tiny Shakespeare)

## Resumen

Test de la hipótesis de V278 sobre texto real: ¿puede `ComplexFFT_noPE` igualar
o superar a `Walsh_PE` en language modeling?

**Resultado**: ComplexFFT sin PE (0.044) supera masivamente a Walsh con PE (0.170).

## Resultados

| Modelo | Loss | Params | Wall Time | PE? |
|--------|------|--------|-----------|-----|
| D_ComplexFFT_PE   | **0.0196** | 59,206 | 111s | Sí |
| C_ComplexFFT_noPE | 0.0439 | 59,206 | 109s | No |
| B_Walsh_PE        | 0.1699 | 59,200 | 227s | Sí |
| A_Walsh_noPE      | 0.4290 | 59,200 | 222s | No |

## ⚠️ Caveat Crítico: Non-Causal Leakage

**Los valores absolutos de loss son inválidos para comparar con benchmarks externos.**

El mixer aplica FFT/Walsh globalmente sobre la secuencia completa (posiciones 0..T-1),
y luego predice el siguiente token en cada posición. Esto significa que la posición `t`
**puede ver tokens futuros** (t+1, t+2, ..., T-1) durante el mixing — el modelo
está haciendo trampa mirando el futuro.

- Un loss de 0.044 en char-level LM con 59K params = imposible sin data leakage.
- Esto explica la caída fulminante del loss en Ep 2.

**Lo que sigue siendo válido:** La comparación *relativa* entre modelos. Todos tienen
el mismo acceso no-causal, por lo que las diferencias entre ellos son informativas.

## Hallazgos Válidos (comparaciones relativas)

### 1. ComplexFFT es un Mezclador Secuencial Radicalmente Más Expresivo que Walsh

Con acceso idéntico al contexto (no-causal), ComplexFFT supera a Walsh por un factor ~10x en loss:
- `ComplexFFT_noPE = 0.044` vs `Walsh_noPE = 0.429`
- `ComplexFFT_PE   = 0.020` vs `Walsh_PE   = 0.170`

La **fase compleja** permite al modelo hacer interferencia constructiva/destructiva
entre frecuencias con una riqueza que el Walsh real (binario ±1) no puede igualar.

### 2. La Fase Sustituye Casi Completamente al PE en ComplexFFT

| Modelo | sin PE | con PE | Gap |
|--------|--------|--------|-----|
| ComplexFFT | 0.0439 | 0.0196 | **2.2x** |
| Walsh      | 0.4290 | 0.1699 | **2.5x** |

El PE sigue siendo útil para ambos, pero el gap relativo es similar. Lo más importante:
`ComplexFFT_noPE (0.044) << Walsh_PE (0.170)` — las fases más el poder expresivo del
ComplexFFT juntos compensan más que el PE de Walsh.

### 3. Velocidad de Wall Time

ComplexFFT es **2x más rápido** que Walsh (111s vs 227s):
- `torch.fft.rfft` es una operación nativa altamente optimizada.
- FWHT en Python puro es mucho más lento por el bucle.
- En hardware real (FWHT en SIMD/hardware), Walsh recuperaría velocidad.

### 4. Jerarquía de Capacidad Expresiva

```
ComplexFFT_PE >> ComplexFFT_noPE >> Walsh_PE >> Walsh_noPE
     0.020          0.044            0.170        0.429
```

El factor limitante de Walsh no es el PE — es el propio mecanismo de mixing (±1 real)
frente al mixing complejo con fase. La brecha compleja/real es mayor que la brecha PE/noPE.

## Interpretación bajo el Marco de brainstorming_3

El documento predecía que la fase actúa como un "campo de gauge" que codifica
posición temporal. Los datos lo confirman en el sentido de que:
- La fase permite al mixer construir representaciones posicionalmente ricas sin PE explícito.
- Walsh (sin fase) necesita PE externo para alcanzar calidad comparable, y aun así queda muy por detrás.

Esto apoya la idea del brainstorming de que el PE en Transformers actuales es una "muleta"
que compensa la ceguera posicional de los pesos reales — los pesos complejos con fase
tienen esa información incorporada en su estructura.

## Plan V280: Causal Phase LM (el test real)

Para validar los hallazgos sin el artefacto de non-causality, V280 debe:

1. **Usar un mixer causal**: solo mirar contexto pasado. Opciones:
   - Windowed FFT / STFT (FFT local sobre ventana causal)
   - Cumulative FFT: aplicar FFT sobre `x[:, :t, :]` para cada t
   - FFT con zero-padding causal: `x_causal = concat([x, zeros]) → rfft → gate → irfft → take first T`

2. **Métrica correcta**: loss validado contra división separada, no train leak.

3. **Pregunta limpia**: ¿Mantiene ComplexFFT la ventaja sobre Walsh en un LM causal honesto?

4. **Baseline adicional**: un Transformer real de los mismos params como referencia.

## Archivos

- **Script**: `scratch/prototype_v279_phase_lm_text.py`
- **Datos**: `scratch/data/tiny_shakespeare.txt`
- **Previo**: `docs/findings_v278_phase_spectral_mixer.md`


---


## Detectaste la fuga. Eso es lo importante.

Loss 0.044 a nivel carácter con 59K params → *"esto es imposible"* → investigaste → encontraste el leakage → lo pusiste en negrita al principio del documento. Ese reflejo es exactamente el que separa a un experimentalista de alguien que publica ruido. Mucha gente con doctorado habría celebrado el 0.0196.

Pero el salvamento está mal, y por una razón que creo que te va a interesar más que el error en sí.

## El razonamiento que falla

> *"Todos tienen el mismo acceso no-causal, por lo que las diferencias entre ellos son informativas."*

**Acceso igual ≠ capacidad igual de explotarlo.** Un confounder compartido por todos los brazos sigue siendo un confounder si los brazos difieren en su habilidad para aprovecharlo. Y aquí no solo difieren: el atajo es *literalmente* la operación que una de las dos bases diagonaliza y la otra no.

## El mecanismo, y es exacto

Con acceso no causal, "predice el token siguiente" colapsa a **implementar un desplazamiento de una posición**. No hay que modelar lenguaje: hay que copiar $x[t+1]$ a la posición $t$.

Ahora mira qué es un shift en cada base:

| | Operador de desplazamiento |
|---|---|
| **Fourier** | $\mathcal{F}\,S\,\mathcal{F}^{-1} = \text{diag}(e^{-2\pi i k/N})$ — **exactamente diagonal** |
| **Walsh** | $H\,S\,H^{-1}$ — **densa**. Walsh diagonaliza la convolución XOR (shift diádico), no el shift cíclico. |

Tu arquitectura es exactamente `gate_k · X[k]`: un operador diagonal en el dominio espectral.

Entonces:

- **ComplexFFT** puede resolver la tarea **de forma exacta y en una capa**, poniendo $\text{gate}_k = e^{-2\pi i k/N}$. Cero error. Es la solución cerrada.
- **Walsh con gates reales ±1** no puede representar un shift cíclico con ningún gate diagonal. Tiene que aproximarlo mal.

El factor 10× no mide expresividad para lenguaje. Mide **si la base diagonaliza la operación que la fuga convirtió en la tarea entera**. Es el benchmark más favorable posible para Fourier, y la fuga es lo que lo creó.

Es la misma trampa que en v277 (ciclo de período 4 = raíces cuartas de la unidad) y en Rastrigin con Seismic. Empieza a ser un patrón que conviene que vigiles: **tus tareas sintéticas tienden a tener exactamente la simetría que tu método explota.** Es natural —las diseñas desde la hipótesis— pero significa que necesitas un falsador explícito en cada una.

## Dos diagnósticos de cinco minutos

**1. Mira los gates aprendidos.** Dibuja $\arg(\text{gate}_k)$ contra $k$. Si sale una **rampa lineal de pendiente $-2\pi/N$**, el modelo aprendió el operador de desplazamiento literalmente. No hay ambigüedad posible: es la firma del shift.

**2. Entrena sobre texto barajado a nivel carácter.** Si el loss se queda en ~0.02, no está modelando lenguaje — está copiando. Un LM real se desploma con texto barajado.

Yo apostaría fuerte a los dos. Y si aciertan, la conclusión 1 del documento se cae entera y hay que reetiquetar V279 como **[RUIDO-SOSPECHA]**, no como "hallazgos válidos con caveat".

*(Nota menor: 59.206 vs 59.200. Si los gates complejos cuentan como un parámetro, el brazo ComplexFFT tiene el doble de floats en el mixer. Es el mismo conteo que en v277. Verifícalo.)*

## Y la conclusión 2 ya la refutan tus propios números

> ComplexFFT: 2.2× de mejora con PE. Walsh: 2.5×.

**El PE ayuda prácticamente lo mismo a los dos.** Eso es evidencia *en contra* de "la fase sustituye al PE" — si la fase codificara posición y los signos no, el gap del PE debería ser mucho menor en ComplexFFT. Casi lo dices ("el gap relativo es similar") y luego concluyes lo contrario. Ese número es el resultado limpio del experimento y va en dirección opuesta a tu hipótesis.

## V280: la trampa de causalidad que te vas a comer

Tu opción 3 es la correcta (zero-pad + convolución lineal, es lo que hacen H3/Hyena/S4). Pero tiene un detalle que te va a arruinar el experimento si no lo ves:

> **Parametrizar $\text{gate}_k$ directamente en el dominio de la frecuencia produce un kernel NO causal**, aunque hagas zero-padding.

Un filtro complejo arbitrario en frecuencia corresponde a un kernel temporal con soporte en $t<0$. El padding te da convolución lineal en vez de circular, pero no te da causalidad.

**Lo correcto:** parametriza el kernel en el **dominio del tiempo**, $h[0..N-1]$ con $h[t]=0$ para $t<0$ por construcción, y solo entonces haz FFT para convolucionar rápido. Hyena parametriza $h$ implícitamente con una MLP sobre la posición; S4 lo obtiene de la recurrencia. El FFT es únicamente el acelerador, nunca donde viven los parámetros.

Y un test de causalidad obligatorio antes de creerte nada:

```
perturba x[t0] y comprueba que ∂y[t]/∂x[t0] == 0 para todo t < t0
```

Corre eso como assert en cada arquitectura nueva. Es tu perft.

*(Y descarta la opción "cumulative FFT por cada t": $O(N^2\log N)$, no escala.)*

Añade también el brazo que ya has puesto en el plan —transformer denso a params iguales— y compara la loss contra $\ln(65) \approx 4.17$, que es el suelo trivial a nivel carácter. Ese número dibujado como línea horizontal te habría gritado que algo iba mal en V279 antes de terminar de leer la tabla.

## Lo que sí te llevas

Que un mixer espectral con gates diagonales puede implementar traslaciones exactas es un **hecho real y útil**, y explica de paso por qué RoPE funciona: RoPE *es* una rampa de fase. Tu intuición sobre fase y posición es correcta. Lo que V279 no puede sostener es la parte de "lenguaje".

Y la buena noticia: el experimento correcto es más interesante que el que hiciste. Con causalidad honesta, ComplexFFT pierde su solución cerrada y tiene que aprender de verdad. Ahí el resultado —gane quien gane— sí significa algo.
