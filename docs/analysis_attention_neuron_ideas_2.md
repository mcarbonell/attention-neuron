Análisis por GPT 5.6 Earth.
Fecha: 20/07/2026.

---


Mi lectura transversal en una frase: la idea más potente no es “usar DCT/FFT”, sino aprender una representación de baja dimensión —coordenadas, fase, gates y routing— que convierta un sustrato fijo en una estructura útil. Para secuencias, además, hay que separar mezcla temporal estacionaria de memoria asociativa dependiente del contenido.

## Taxonomía

### A. Mecanismos respaldados por datapoints

- **La reindexación de canales cambia radicalmente la compresibilidad sin cambiar la función.** V290 verifica que permutar internamente los canales de GPT-2 preserva la PPL hasta error numérico, y que TSP/Fiedler mejora mucho la poda/low-pass DCT. Es una señal fuerte: la “alta frecuencia” de una matriz depende de unas coordenadas de canales arbitrarias.

- **La fase compleja conserva información posicional que la amplitud sola destruye.** En V278, quitar la fase FFT deja una tarea de posición en azar; en V281, el mixer FFT complejo realmente causal queda cerca de attention con ~55% de parámetros y el PE explícito aporta prácticamente cero en ese setup. Esto es un mecanismo concreto, no sólo una metáfora.

- **Una convolución/filtro espectral estacionario no da recall asociativo por contenido.** V292 es un negativo muy valioso: gating local multiplicativo sobre FFT/Walsh no resuelve MQAR. El FFT estático gana sólo cuando el desplazamiento es fijo. Esto delimita con precisión qué no puede resolver el mixer fase-espectral solo.

- **El binding/unbinding holográfico por fase sí abre una vía lineal para memoria asociativa**, pero aún con capacidad baja: V293–V296 pasan de azar (~3%) a ~19–24% en MQAR; la normalización causal de masa mejora estabilidad y convergencia. Es una señal de mecanismo, no una solución de memoria todavía.

- **La selección estructurada sobre sustratos congelados funciona.** La Attention Neuron rank-4 llega prácticamente al MLP denso en MNIST en V2/Fase 2, y las variantes ternarias/gated muestran que inhibición y selección importan. La lectura correcta es: un backbone grande puede controlarse con pocos grados de libertad; no que los pesos o el cómputo hayan desaparecido.

- **Los priors geométricos son muy efectivos cuando el dominio los justifica.** Stroke/matchstick neurons en V50–V57 y DCT local en V59–V62 son buenos datos de que parametrizar detectores como geometría/campos lisos puede sustituir muchos pesos libres en visión simple.

- **Soft MoE facilita colusión, no especialización.** V214 lo muestra claramente; V217 y V240 sugieren que competencia dura o asignación de gradiente por competencia es más prometedora cuando los expertos representan leyes incompatibles.

### B. Apuestas arquitectónicas abiertas

- **Phase-Spectral Transformer + núcleo matrix-free.** V281–V284/V283 son el núcleo ejecutable más maduro: FFT causal complejo para dinámica temporal, Walsh/DCT para parametrizar proyecciones. La señal es buena en LM pequeño; falta saber si escala y, sobre todo, si conserva recall por contenido.

- **Memoria holográfica causal con control de colisiones.** V285 es impresionante como prueba sintética, pero “O(1) memory” no implica memoria exacta arbitraria. El problema pendiente es capacidad frente a número de pares, longitud, interferencia y queries adversarias.

- **Aprender coordenadas antes de comprimir.** V290, V151–V161, V284 y V287 apuntan a una familia común: ordenar/deformar el espacio para que las variables se vuelvan suaves y luego aplicar DCT/FFT/quantización.

- **Router de familias funcionales.** Lineal, logarítmico, resonante, discontinuo, geométrico y simbólico no deberían competir mediante un softmax ordinario. Hay una apuesta seria en un selector duro basado en competencia, residuo, novedad o estructura del input.

- **Sistema híbrido continuo–discreto.** V212, V213 y V240 sugieren que el bloque simbólico pequeño debe tener un optimizador distinto del resto; no merece forzar todo el modelo a Adam ni todo a DGE.

### C. Metáforas/claims que conviene no confundir con mecanismo

No para descartarlos, sino para que guíen sin contaminar la brújula:

- “Los pesos no importan” / “red sin pesos”: el sustrato congelado sigue almacenando parámetros y se computa. Lo que muestran los datos es reducción de parámetros entrenables y una reparametrización estructurada.

- “Lenguaje = bajas frecuencias semánticas; altas = ruido”: V288 contradice una versión simple de eso. El low-pass DCT destruye GPT-2 con poda muy pequeña; lo que importa es energía y suavidad **después de elegir las coordenadas correctas**, no una jerarquía intrínseca baja/alta frecuencia.

- “Walsh = lógica, DCT = semántica”: buena hipótesis de diseño, aún no mecanismo demostrado. El fallo de causalidad de Walsh en V281 es del procedimiento `FWHT → mask → FWHT`, no una imposibilidad general de todo operador Walsh causal.

- “Contexto infinito O(1)”: sólo puede significar resumen/recall con capacidad finita, no memoria exacta ilimitada. La curva de colisiones es la prueba central que falta.

- “XOR demuestra Turing-completitud”, “conciencia”, “hipocampo”, “cerebelo”: nombres fértiles para diseñar módulos, no conclusiones empíricas.

## Conexiones nuevas que me parecen más fértiles

### 1. El programa real puede ser “fijación de gauge” de redes

V290 da la pieza más importante del repositorio: los canales ocultos tienen una simetría de permutación. Por tanto, una matriz aparentemente rugosa puede ser una función lisa escrita en coordenadas malas.

Esto conecta directamente:

- Permutación TSP/Fiedler de canales en V290.
- Ordenar clanes/manifolds antes de superponer memoria en V151–V161.
- Regularización de continuidad de fase en V284.
- Warps conformes de V287.
- Geometría hiperbólica de V286.

La versión fuerte de la hipótesis sería:

> Antes de comprimir pesos, memoria o activaciones, encuentra las coordenadas en las que son lisos.

Un experimento muy limpio sería aplicar permutaciones funcionalmente equivalentes capa a capa para minimizar la entropía espectral de \(P_\ell W_\ell P_{\ell-1}^T\), y sólo después cuantizar/prunear. Es más profundo que “DCT quantization”: sería compresión mediante elección de coordenadas. V288→V290 ya es un mini-arco que lo respalda.

### 2. Separar dos motores temporales

Los documentos mezclan a veces “contexto largo” y “recall asociativo”, pero V292 los separa muy bien:

- **FFT causal complejo**: excelente candidato para dinámica temporal, patrones estacionarios y mezcla secuencial barata.
- **Memoria holográfica de fase/SSM**: candidata para “key → value” dependiente de contenido.
- **No son sustitutos entre sí.**

La arquitectura interesante no sería “FFT reemplaza attention” sino:

```text
FFT causal: dinámica/sintaxis/señal temporal
        +
memoria de fase normalizada: eventos recuperables por contenido
        +
router de novedad: decide cuándo escribir y consultar
```

Esto es bastante más prometedor que seguir intentando que un filtro de convolución resuelva MQAR.

### 3. Memoria fase–hiperbólica–por clanes

La debilidad de V293–V296 es la diafonía. V151–V161 ya descubrieron que la superposición mejora cuando los recuerdos se dividen en clanes y se ordenan. V286 ofrece una pista distinta: la geometría hiperbólica representa jerarquías de forma más eficiente.

Una apuesta genuinamente nueva sería usar:

- radio/posición hiperbólica para nivel, escala temporal o familia semántica;
- fase compleja para binding de identidad;
- clanes/buckets para evitar colisiones;
- normalización causal de masa de V296;
- cleanup no lineal sólo dentro del bucket relevante.

No afirmaría que eso funcionará, pero sí que combina exactamente los fallos y éxitos observados, no sólo metáforas.

### 4. El “controlador” importa más que el tipo de experto

Attention Neuron, stage-gating, Oligarchy, V214–V217, Atlas de familiaridad y PID cuentan la misma historia: no basta con una biblioteca rica de operaciones; hay que controlar cuándo se activa cada una.

La versión interesante sería un controlador que use señales observables:

- entropía/esparsidad espectral;
- familiaridad estructural;
- residuo entre predicción y reconstrucción;
- estabilidad de fase/gates;
- distancia geodésica al atlas.

Eso evita que un router softmax aprenda a compensar errores entre expertos. Haría que “MoE de bases” sea descubrimiento de representación, no sólo mezcla de redes.

## Ranking provisional por novedad × palanca

| Dirección | Novedad | Palanca | Estado |
|---|---:|---:|---|
| Coordenadas/permutaciones para compresión y representación | 5/5 | 5/5 | Señal fuerte y transversal |
| Memoria causal fase–clanes–jerarquía | 5/5 | 5/5 | Muy abierta; upside máximo |
| FFT causal complejo + proyecciones matrix-free | 4/5 | 5/5 | La rama más madura para construir |
| Router duro entre bases/espacios funcionales | 4/5 | 4/5 | Buena hipótesis, evidencia inicial |
| Gating sobre sustratos congelados/ternarios | 3/5 | 4/5 | Mecanismo útil, no aún una teoría general |
| Neuro-simbólico con microbloques DGE | 4/5 | 3/5 | Especialmente atractivo para world models |
| Neuronas geométricas/deformaciones | 4/5 | 3/5 | Muy buenas como frontend con prior correcto |
| PID/optimización por fases | 2/5 | 3/5 | Buen satélite, no el centro |

Mi apuesta principal no sería unificarlo todo bajo “resonancia”. La unificación más precisa es:

> **La red aprende mejor cuando primero se le da —o aprende— un sistema de coordenadas donde el problema sea liso, disperso o separable; después un controlador de baja dimensión decide qué estructura activar.**

Eso explica conjuntamente DCT, Walsh, fase, permutaciones, matchsticks, mapas conformes, Poincaré, gating y memoria holográfica.