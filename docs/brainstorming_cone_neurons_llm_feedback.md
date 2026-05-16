
----

FEEDBACK Claude Opus 4.6

---


# Cone Neurons para LLMs — Análisis Crítico y Propuesta Factible

## Diagnóstico del brainstorming

He leído ambos documentos a fondo. El brainstorming contiene ideas muy inspiradoras pero hay que separar el grano de la paja, porque mezcla intuiciones válidas con afirmaciones que suenan bien pero que no sobreviven un análisis técnico riguroso.

### Lo que NO funciona (y por qué)

#### 1. `ref_idx` como parámetro aprendible (el prototipo propuesto)

El pseudocódigo del brainstorming tiene un bug conceptual grave:

```python
self.ref_idx = nn.Parameter(torch.randint(0, seq_len, (n_cones,)))
# Luego: ref_embeddings = x[0, self.ref_idx, :]
```

Problemas:
- **`randint` no es diferenciable.** Un índice discreto no tiene gradiente. No se puede optimizar con backpropagation.
- **Depende del batch 0**: `x[0, ...]` significa que todo el modelo depende de qué secuencia sea la primera del batch. Eso es catastrófico.
- **El forward destruye información**: `h = cone_weights.mean(dim=1)` colapsa TODA la secuencia a un solo vector. Un LLM necesita preservar la dimensión temporal para predecir token-por-token.

**Veredicto:** Este prototipo no entrena. Ni siquiera llega a ejecutarse correctamente.

#### 2. "Conos en embedding space semántico" (la idea principal)

La idea de medir distancia euclidiana entre embeddings tiene un problema fundamental: **los embeddings al inicio del entrenamiento son aleatorios**. La distancia entre "gato" y "estaba" al principio es ruido puro. El cono no sabe a qué apuntar porque el espacio sobre el que opera está cambiando bajo sus pies en cada paso de gradiente. Es un problema de huevo y gallina: necesitas buenos embeddings para que los conos funcionen, pero necesitas que los conos funcionen para aprender buenos embeddings.

#### 3. Las tablas comparativas "LLaMA con conos = 25M params"

Estas cifras son puro marketing. Reemplazar matrices d×4d por 4 parámetros por neurona elimina casi toda la capacidad de transformación del modelo. Un FFN con conos no puede aprender una transformación no-lineal arbitraria en R^d — solo puede "seleccionar regiones" del vector de entrada. Es la diferencia entre una cámara con 1 megapíxel y una con 100 megapíxeles: sí, la primera pesa menos, pero pierde resolución.

---

### Lo que SÍ funciona (la pepita de oro real)

El brainstorming tiene **una idea central correcta** que merece ser probada:

> **Cone 1D posicional sobre la secuencia de tokens** — cada "head" no necesita una matriz QK^T de N×N, sino que aprende DÓNDE mirar en la secuencia con 3-4 parámetros: centro μ, ancho σ, y amplitud a.

Esto es análogo directo a lo que funciona en V101 (píxeles 2D → posiciones en grid 2D). En un LLM:
- Las posiciones de los tokens en la secuencia son un espacio 1D ordenado.
- "Cerca en posición" sí tiene significado real: n-gramas, sintaxis local, concordancia sujeto-verbo.
- Múltiples capas con radios crecientes resuelven las dependencias largas (el argumento de V1→V4→IT de la corteza visual).

Esto es **exactamente lo que hacen** Longformer, BigBird, y sliding-window attention... pero con patrones FIJOS. Tu propuesta los hace **aprendibles** con 3 parámetros, lo cual es genuinamente nuevo y elegante.

---

## Propuesta: V103 — Cone1D Sequence Mixer para char-level LM

> [!IMPORTANT]
> La idea más factible y verificable es la más simple: **conos 1D posicionales** sobre la secuencia temporal, comparados contra un Transformer y un MLP matched-params. Nada de distancias semánticas ni embedding space. Solo posiciones.

### Arquitectura

```
Embedding(vocab, d_model) + PosEmbedding(seq_len, d_model)
    ↓
[Cone1DBlock × N_layers]:
    1. Cone1DMixer:  (B, S, D) → (B, S, D)   — mezcla temporal con conos
    2. PointwiseFFN: (B, S, D) → (B, S, D)   — transformación por posición
    3. LayerNorm + Residual
    ↓
Linear(d_model, vocab_size) → logits
```

#### Cone1DMixer (la pieza clave)

```python
class Cone1DMixer(nn.Module):
    """
    n_cones "neuronas" que miran regiones de la secuencia.
    Cada cono: mu (centro), sigma (ancho), amplitude (excit/inhib) = 3 params
    
    Para cada posición t en [0, seq_len):
      peso[j] = amplitude * max(0, 1 - |t - mu| / sigma)    # triangular
      o
      peso[j] = amplitude * exp(-0.5 * ((t - mu)/sigma)^2)  # gaussiano
    
    Salida: weighted sum de los embeddings vecinos → mezcla temporal.
    """
```

**Diferencia crucial con el prototipo del brainstorming:**
- NO usa distancia en embedding space. Usa distancia posicional 1D, que es FIJA y determinista (como los píxeles en visión).
- NO colapsa la secuencia. Cada posición t produce un valor mezclando sus vecinos según los conos.
- Los conos son COMPARTIDOS en la secuencia (como un filtro de convolución), pero cada cono mira a una distancia relativa diferente.

#### Causalidad

Para language modeling autoregresivo, el cono debe ser causal: solo mira hacia atrás.

```python
# Mask causal: solo posiciones j <= t contribuyen
causal_mask = (positions <= t)  # triangular truncado a la izquierda
weights = cone_weights * causal_mask
```

Esto es trivial y mantiene los 3 parámetros por cono.

### Parámetros por capa

| Componente | Parámetros |
|---|---|
| Cone1DMixer: n_cones × 3 (mu, sigma, amp) | ~192 (si n_cones=64) |
| Projection: n_cones → d_model | n_cones × d_model = ~8K |
| PointwiseFFN: d_model → 4×d_model → d_model | ~2 × d_model × 4×d_model = ~131K (d=128) |
| LayerNorm | ~256 |
| **Total por capa** | **~140K** |

Con 4 capas → ~560K + embeddings. Comparable a un tiny transformer para test justo.

### Baseline

Un Transformer con matched params:
- d_model=128, n_heads=4, n_layers=2, seq_len=256
- ~100K-200K params

### Dataset

**Shakespeare char-level** (clásico de Karpathy), ~1M caracteres, vocab_size ≈ 65.

### Métrica

- **Validation loss** (Cross-Entropy, bits-per-character)
- **Velocidad de convergencia** (loss en epoch 1-5)
- **Parámetros totales**
- **Tokens/segundo en inferencia**

---

## Open Questions

> [!IMPORTANT]
> **¿Conos relativos o absolutos?**
> - **Absoluto:** mu = posición fija en la secuencia (como V101 en la imagen). Problema: no generaliza a secuencias más largas que el entrenamiento.
> - **Relativo:** mu = offset respecto a la posición actual (como convolución). Generaliza a cualquier longitud. **Yo recomiendo relativo.**

> [!IMPORTANT]
> **¿Forma del cono: triangular o gaussiano?**
> - Triangular (como V101): corte duro, más barato computacionalmente.
> - Gaussiano: suave, pero nunca llega a cero exacto (hay que truncar).
> - **Propongo probar ambos en el mismo script.**

> [!WARNING]
> **El FFN sigue siendo denso.** La eficiencia paramétrica de los conos solo aplica al mezcla temporal (reemplaza atención). El FFN pointwise sigue siendo la mayor parte de los parámetros (~95%). Si quieres eficiencia extrema, habría que hacer un segundo experimento reemplazando también el FFN por conos 1D sobre la dimensión d_model. Pero eso es V104 — primero validemos que la mezcla temporal funciona.

---

## Verification Plan

### Automated Tests
1. Ejecutar V103 en Shakespeare char-level, 20 epochs
2. Ejecutar Transformer baseline matched-params, 20 epochs  
3. Comparar val_loss, convergence speed, tokens/sec

### Criterio de éxito
- Si V103 alcanza **val_loss comparable** (±10%) al Transformer con menos o iguales parámetros → la idea funciona
- Si además converge más rápido en las primeras 5 epochs → bonus (esperado por el V101 precedent)
- Si la val_loss es >20% peor → la idea posicional pura no es suficiente y habría que reconsiderar

---

FEEDBACK Claude Opus 4.7

---


Llevando la Atención Cónica al dominio del lenguaje y LLMs

Tu idea es muy potente porque ataca el problema central de los LLMs: la explosión paramétrica de las proyecciones Q/K/V y los MLPs. Te lanzo varias direcciones, de más conservadoras a más rupturistas:
1. Conos en el espacio posicional (la traducción más directa)

En visión, tu cono vive en el espacio 2D de píxeles. En lenguaje, el equivalente natural es el eje temporal/posicional de la secuencia.

Una "neurona de atención cónica 1D" sobre tokens tendría:

    Ct​: posición central del foco atencional
    Radio: ancho del contexto relevante
    Amplitud: excitación/inhibición

Pesoi​=Amplitud⋅max(0,1−Radio∣i−Ct​∣​)

Esto reemplazaría el mecanismo QKT (que es O(n2⋅d)) por 3 parámetros por cabeza. Sería una forma de ALiBi / sliding window aprendido y dinámico, pero con inhibición lateral biológica.
2. Conos en el espacio de embeddings (más interesante)

Aquí está lo jugoso: en lugar de operar sobre posiciones, los conos viven en el espacio semántico. Cada "neurona cónica" tiene:

    C
    : un centroide en Rd (un concepto prototipo)
    Radio: radio semántico
    Amplitud: excitación/inhibición

activacioˊn(token)=Amplitud⋅max(0,1−Radio∥emb(token)−C
∥​)

Esto es esencialmente una MLP layer reformulada como mezcla de RBFs cónicos. Reemplazas las matrices densas del FFN (que son ~70% de los parámetros de un transformer) por neuronas con d+2 parámetros en vez de d⋅dff​.
3. La conexión más profunda: Superposición e Inhibición

Anthropic ha mostrado que los LLMs codifican features en superposición y que muchas se inhiben mutuamente. Tu mecanismo de Amplitud negativa (Centro OFF) es literalmente lo que falta en los SAEs (Sparse Autoencoders) actuales, que solo modelan features positivas con ReLU.

Una capa cónica con inhibición sería un SAE biológicamente plausible: cada feature no solo se activa ante su prototipo, sino que suprime activaciones cuando aparece su anti-concepto. Esto podría dar interpretabilidad nativa.
4. Auto-organización topológica del vocabulario (lo más bonito de V102)

Tu hallazgo de V102 es el más explotable en lenguaje. Si fuerzas un cuello de botella triangular 1D/2D al final de un LM, la red se vería obligada a organizar el espacio latente de manera que tokens semánticamente similares queden físicamente cercanos en el vector oculto.

Esto sería un Word2Vec emergente vía restricción geométrica, sin necesidad de objetivos contrastivos. La topología del clasificador induce la topología semántica.

Imagínate visualizar el vector oculto de 1024 dims como una línea, y ver que la zona [200-250] se activa para verbos, [400-450] para entidades, etc. Interpretabilidad por construcción.
5. Atención cónica jerárquica (multi-escala)

En visión, un cono cubre píxeles vecinos. En lenguaje, podrías tener conos a múltiples escalas:

    Conos pequeños sobre tokens (sintaxis local)
    Conos medianos sobre frases (semántica)
    Conos grandes sobre párrafos (discurso)

Cada uno con solo 3-4 parámetros. Sería una alternativa ultraligera a la atención multi-cabeza.
6. Experimentos concretos que yo haría primero

    Baseline mínimo: clasificación de sentimiento (SST-2). Reemplaza el FFN de un transformer pequeño por una capa cónica en espacio de embeddings. Mide params vs accuracy.

    El test crítico: language modeling sobre TinyStories o WikiText con un transformer donde el MLP es 100% cónico. Si convergencia se mantiene con 10x menos params, tienes algo grande.

    El test de interpretabilidad: entrena un LM minúsculo con clasificador triangular final sobre el vocabulario. Visualiza si emerge organización topológica de tokens (sinónimos juntos, antónimos en zonas de Amplitud negativa).

7. El riesgo conceptual

El lenguaje, a diferencia de MNIST, no tiene una métrica euclidiana natural. La distancia entre embeddings sí, pero la topología semántica es altamente no-euclidiana (jerárquica, a veces hiperbólica).

Posible mitigación: usar distancia coseno o distancia hiperbólica en lugar de euclidiana para los conos en espacio de embeddings. Un "cono hiperbólico" podría capturar mejor relaciones jerárquicas (hiperónimo/hipónimo).

Mi apuesta: la idea más prometedora es la #2 + #3 combinadas: reemplazar los FFN del transformer con capas cónicas con inhibición en espacio de embeddings. Si tu intuición de V101 escala, podrías obtener LLMs con 10x menos parámetros en la parte FFN y con interpretabilidad nativa por las amplitudes con signo.

¿Quieres que profundicemos en alguno de estos hilos, o que esbocemos la arquitectura de un experimento específico (digamos, un nanoGPT cónico)?


---

# Aplicando Cone Attention al Lenguaje

Buena pregunta — y más sutil de lo que parece, porque **el lenguaje no tiene la topología 2D regalada** que tenía MNIST. El éxito de V99 dependió de que los píxeles vivían en una rejilla $(P_x, P_y)$ fija y conocida. En lenguaje, la "coordenada" análoga hay que **elegirla o construirla**. Ahí está justamente lo interesante.

Te propongo cinco direcciones, ordenadas de más conservadora a más especulativa.

## 1. Cone Positional Attention (la más directa)

**Idea:** Cada cabeza de atención no calcula $QK^\top$ denso, sino que define un cono sobre **posiciones relativas**.

$$w(i, j) = A \cdot \max\left(0, 1 - \frac{|i - j - C|}{R}\right)$$

Cada cabeza tiene 3 parámetros: desplazamiento $C$ (a qué distancia mira), radio $R$ (cuánto contexto), amplitud $A$ (excita/inhibe).

| Aspecto | Atención estándar | Cone Positional |
|---|---|---|
| Params por cabeza | $3 \cdot d^2$ (Q,K,V) | $\sim 3$ + V |
| Sesgo inductivo | Ninguno | Localidad explícita |
| Inhibición | No (softmax ≥ 0) | Sí ($A < 0$) |

**Honestidad intelectual:** Esto se solapa con **ALiBi**, **T5 relative bias** y **local attention**. Tu aporte genuino sería la **amplitud negativa** (inhibición posicional explícita) y el corte duro tipo cono. ALiBi solo tiene pendiente; tú tienes pendiente *y* centro desplazable *y* signo.

## 2. Cone FFN sobre el Residual Stream (la más V102)

Aquí está, en mi opinión, **la idea más fiel al espíritu de tu trabajo**.

Las FFN de un transformer tienen neuronas con $d_{\text{model}}$ pesos cada una. Si forzaras a que el residual stream tenga una **topología 1D ordenada** (las dimensiones no son intercambiables, tienen vecindad), cada neurona FFN podría ser un cono:

$$\text{neurona}_k(x) = A_k \cdot \sum_{i=0}^{d-1} \max\left(0, 1 - \frac{|i - C_k|}{R_k}\right) \cdot x_i$$

Esto es **V102 a escala de transformer**. Predicción: el residual stream se auto-organizaría en bandas semánticas (sintaxis al principio, semántica en el medio, pragmática al final, o algo análogo emergente). Esto sería **brutalmente interpretable**.

## 3. Conic LM Head (la más útil prácticamente)

<details>
<summary><strong>El LM head es ~30% de los parámetros de un LLM pequeño. Comprimirlo importa.</strong></summary>

Actualmente: $W_{\text{out}} \in \mathbb{R}^{|V| \times d}$ — un vector denso por token de vocabulario (~50k × 4096 = 200M params).

**Versión cónica:** Cada token del vocabulario es un cono en el espacio de embeddings:
$$\text{logit}(t) = A_t \cdot \max\left(0, 1 - \frac{\|h - C_t\|}{R_t}\right)$$

Con $C_t \in \mathbb{R}^k$ para $k \ll d$ (ej. $k=64$). El residual stream se proyecta a un espacio "semántico" de baja dimensión donde los tokens son **prototipos cónicos**.

Esto fuerza —como V102— una **auto-organización topológica del espacio semántico**: tokens parecidos deberán tener centros cercanos para que el gradiente fluya razonablemente. Tendrías un mapa semántico **emergente y visualizable** del vocabulario.

</details>

## 4. Atención Excitatoria/Inhibitoria sobre Contenido

La atención softmax es estrictamente no-negativa: nunca puede decir "ignora *activamente* este token". Hay trabajos recientes (Cogno, negative attention) explorando esto.

Tu marco lo encaja naturalmente: define **dos conos por cabeza** — uno excitatorio sobre la similitud QK, otro inhibitorio. Es el análogo lingüístico de **ON-center / OFF-surround** de tu V101.

**Aplicación:** Tareas de negación, contraste, desambiguación. ("No el rojo, el azul" — el modelo necesita *suprimir* "rojo" activamente).

## 5. Cones sobre Estructura Sintáctica (especulativa pero bonita)

Si pre-procesas con un parser (o usas un árbol latente aprendido), tienes una **topología de árbol** sobre los tokens. Un cono en ese árbol tiene centro = nodo, radio = distancia geodésica máxima.

$$w(i,j) = A \cdot \max\left(0, 1 - \frac{d_{\text{tree}}(i,j)}{R}\right)$$

Esto codificaría **localidad sintáctica** en vez de posicional. Un cono podría atender exactamente al "sujeto del verbo principal" sin importar cuántas palabras lo separen linealmente.

---

## Mi recomendación

Si tuviera que elegir **uno solo** para empezar, sería el **#2 (Cone FFN con residual topológico)** porque:

1. Es el más fiel al hallazgo profundo de V102 (auto-organización forzada).
2. Es entrenable a escala pequeña (GPT-nano sobre TinyStories, por ejemplo).
3. La promesa de interpretabilidad es **vendible** y verificable visualmente.
4. No compite con décadas de literatura de atención eficiente.

El **#3 (Conic LM Head)** sería el más práctico/publicable: compresión real + interpretabilidad emergente.

## Riesgo principal que debes anticipar

En MNIST, los píxeles **están naturalmente correlacionados con sus vecinos** (un trazo cubre píxeles adyacentes). En el residual stream de un transformer, las dimensiones son inicialmente **intercambiables** — no hay vecindad. Tienes que **forzarla** desde la arquitectura (cones en FFN + cones en LM head simultáneamente) o no emergerá.

Es decir: en MNIST la topología la regaló el dataset. En lenguaje, tú tienes que **construir el sustrato topológico** y dejar que el modelo lo pueble. Eso es más ambicioso, pero también más interesante.

¿Quieres que diseñemos en detalle el experimento mínimo para validar el #2 — algo como un transformer de 2 capas sobre TinyStories con FFN cónica y métricas de auto-organización?