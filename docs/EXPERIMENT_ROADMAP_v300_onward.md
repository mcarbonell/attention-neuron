# Experiment Roadmap Post-V300: Capacity Scaling & Beyond

**Contexto**: V300 (`prototype_v300_capacity_scaling.py`) está corriendo. Barrido de capacidad vs `d_k ∈ {32, 64, 128}` iso-floats, cargas 32–256 pares (L=256–2048). Compara ComplexDeltaPhaseHolographic vs RealDeltaNetVanilla vs Softmax MHA.

Este documento resume **preguntas abiertas** y **experimentos concretos** para la siguiente fase. Está pensado para pasarse a la IA codificadora que ejecute los siguientes prototipos.

---

## 1. Preguntas que V300 responderá (y qué buscar en los resultados)

| Pregunta | Hipótesis | Señal de validación en V300 |
|----------|-----------|----------------------------|
| **Ley de escalado de capacidad**: ¿Capacidad máx (pares >95%) escala ~lineal con `d_k`? | Sí: `max_pairs ≈ α · d_k` (α ~ 0.8–1.0 para complejo) | Curva `accuracy(d_k, num_pairs)` muestra umbral consistente |
| **Ventaja compleja vs real se mantiene a todo `d_k`?** | Sí: gap se *amplía* con `d_k` (geometría $S^1$ vs $S^{d_k-1}$) | Ratio `acc_complex / acc_real` crece con `d_k` a carga alta |
| **¿Softmax MHA sigue siendo techo inalcanzable a L=2048?** | Softmax escala O(N²) → memoria/tiempo explosivo; complejo O(N) lineal | Tiempo/step complejo << Softmax a L=2048; accuracy comparable |
| **¿A qué `d_k` el modelo complejo alcanza 256 pares (>95%)?** | `d_k=128` debería hacerlo (32K floats/head ≈ 256 pares teóricos) | `acc(d_k=128, pairs=256) > 95%` |

**Métricas clave a extraer del JSON de V300**:
- `capacity_95[d_k]` = max `num_pairs` con accuracy > 95%
- `degradation_slope[d_k]` = pendiente accuracy vs log(pairs)
- `time_per_step[d_k, model]` = escalado computacional real

---

## 2. Experimentos Prioritarios Post-V300

### V301: Phase Softmax Kernel — Cerrar el gap 95% → 99.9%+

**Pregunta**: ¿Una no-linealidad exponencial sobre similitud de fase (`exp(cos(Δθ)/τ)`) implementable en O(N) via kernel trick recupera recall perfecto?

**Diseño**:
```
Memoria: M = Σ β_t · (V_t ⊗ K_t)           # igual que V298/V299
Lectura:  r_t = Σ_τ exp(cos(θ_q - θ_k_τ)/τ) · V_τ   # Phase Softmax
```
- Kernel trick: `exp(cos(Δθ)/τ) = Σ_n I_n(1/τ) cos(nΔθ)` (expansión Bessel) → truncar a N_terms=4–8
- O(N) via recurrencia: mantener Σ cos(nθ_k) V_k y Σ sin(nθ_k) V_k para n=0..N_terms
- Comparar: ComplexDeltaPhase (lineal) vs PhaseSoftmaxHolographic vs Softmax MHA

**Archivo**: `prototype_v301_phase_softmax.py`
**Métrica**: Accuracy MQAR a 64/128/256 pares; convergencia épocas; overhead computacional vs lineal pura

---

### V302: Dynamic State Decay / LRU — Escalar más allá de `H·d_k`

**Pregunta**: Cuando `num_pairs > H·d_k` (capacidad teórica de la matriz), ¿un decay dinámico estilo Mamba/LRU permite contexto "infinito" con olvido controlado?

**Diseño**:
```
M_t = λ_t · M_{t-1} + β_t · (e_t ⊗ K_t)
λ_t = σ(λ_proj(x_t)) ∈ (0, 1)   # decay aprendido por token
```
- Variante A: decay global escalar por cabeza
- Variante B: decay por valor propio (SVD de M) — mantener top-r, decaer resto
- Variante C: "dual memory" — cabeza 0: λ≈1 (reciente), cabeza 1: λ<1 (global consolidado)

**Archivo**: `prototype_v302_dynamic_decay.py`
**Cargas**: 128, 256, 512, 1024 pares (L=1024–8192)
**Métrica**: Accuracy vs num_pairs; "forgetting curve" controlada; comparación con Fourier Hippocampus (V285)

---

### V303: Multi-Head Specialization — Arquitectura de Doble Memoria

**Pregunta**: ¿Especializar cabezas (recientes vs globales) supera a cabezas homogéneas?

**Diseño**:
- H=4 o 8 cabezas
- **Grupo A (Recientes)**: `β` alta, `λ≈1`, `d_k` pequeño (rápidas, precisa corto plazo)
- **Grupo B (Globales)**: `β` baja, `λ<1` aprendido, `d_k` grande (consolidadas, largo plazo)
- Routing: consulta decide qué grupo leer via attention ligera sobre `β, λ`

**Archivo**: `prototype_v303_dual_memory.py`
**Baseline**: ComplexDeltaPhaseHolographic H=2/4/8 homogéneo (resultados V300)

---

### V304: Port a tiny-thinker V12 — Validación en Lenguaje Natural

**Pregunta**: ¿La memoria holográfica de fase compleja (V298/V299) mejora perplexity/throughput en TinyStories vs `StatefulComplexFFTMixer` actual?

**Plan de port**:
1. **Reemplazo drop-in**: `StatefulComplexFFTMixer` → `ComplexDeltaPhaseHolographicBlock` en `model_spectral_v11_albert.py`
2. **Config V12 candidate**: `d_model=1024, H=8, d_k=128, L=8` (9.44M params champion V11)
3. **Presupuesto estado**: 8 cabezas × 2 × 128² = 262,144 floats ≈ 1 MB estado O(1) vs KV cache lineal
4. **Entrenamiento**: 2000–5000 iteraciones TinyStories (igual que V11)
5. **Benchmarks**: Val loss, MQAR embebido en eval, throughput tokens/sec, memoria GPU

**Archivos**:
- `tiny_thinker/model_spectral_v12_holographic.py`
- `tiny_thinker/configs/v12_holographic.yaml`
- Script de entrenamiento adaptado de `train_v11.py`

**Criterio de éxito**: Val loss ≤ 4.15 (V11 champion 4.1287) + MQAR > 90% + 2× speedup inferencia L>1024

---

### V305: Spectral Quantization 4-bit — Compresión Post-Entrenamiento

**Pregunta**: ¿La cuantización espectral jerárquica (V289: 4.25 bits → mejor PPL que float32) se aplica a la matriz de memoria compleja y pesos de fase?

**Diseño**:
- Aplicar DCT/Walsh a `M ∈ ℂ^{d_k×d_k}` (separar real/imag o magnitud/fase)
- Cuantización jerárquica: 8 bits bajas frecuencias, 4 bits altas frecuencias
- Evaluar: accuracy MQAR post-cuantización sin fine-tuning; degradación vs float32
- Bonus: cuantización *durante* entrenamiento (L1 en espectro DCT como regularizador)

**Archivo**: `prototype_v305_spectral_quant.py`
**Baseline**: V289 results (4.25 bits → 88.12 PPL vs 89.58 float32 en GPT-2)

---

### V306: TSP Permutation + DCT para Cores Walsh — Mejorar Compresibilidad

**Pregunta**: V290 mostró que permutar canales MLP por TSP (Greedy) suaviza la señal → DCT 90% compresión supera original. ¿Aplicable a núcleos Walsh de `WalshLinear` en tiny-thinker?

**Diseño**:
- Extraer núcleos `W ∈ ℝ^{k×k}` de `WalshLinear` entrenado
- Resolver TSP greedy sobre filas/columnas (distancia coseno entre vectores)
- Aplicar permutación → DCT → cuantización jerárquica 4-bit
- Medir: PPL post-cuantización vs original; overhead de permutación (una vez)

**Archivo**: `prototype_v306_tsp_walsh.py`

---

## 3. Experimentos de "Alto Riesgo / Alto Impacto"

| Experimento | Idea Central | Dificultad | Payoff |
|-------------|--------------|------------|--------|
| **V310: Phase Attention Kernel (General)** | Kernel `K(θ_q, θ_k) = exp(cos(θ_q-θ_k)/τ)` generalizable a cualquier tarea seq2seq | Media (kernel trick + estabilidad) | Atención O(N) universal |
| **V311: Hierarchical Phase Memory** | Niveles: L1 (fasores d_k=32, rápida) → L2 (fasores d_k=128, lenta) → L3 (Fourier Hippocampus O(1)) | Alta (arquitectura multi-escala) | Contexto verdaderamente infinito |
| **V312: Scientific Neuron sobre Fase** | Blueprint "Scientific Neuron": extraer leyes simbólicas del espacio latente espectral tras PST | Muy alta (razonamiento simbólico) | Interpretabilidad + generalización OOD |
| **V313: Cross-Layer Memory Sharing** | Compartir matriz M entre capas (ALBERT-style) — memoria global única para todo el modelo | Media | Parámetros ↓, consistencia ↑ |

---

## 4. Infraestructura / Tooling Necesario

| Necesidad | Descripción | Prioridad |
|-----------|-------------|-----------|
| **Auto-sweep runner** | Script que lance grid (d_k, num_pairs, lr, semillas) y agregue resultados a `master_ledger.jsonl` automáticamente | Alta |
| **Visualización capacity frontier** | Plot interactivo: accuracy vs num_pairs por d_k, modelo; curvas iso-floats | Media |
| **MQAR hardness sweeps** | Variar: num_keys/num_vals (vocab), interleaving pattern, noise tokens, distractor pairs | Media |
| **Profiling O(N) vs O(N²)** | Medir tiempo/step, memoria GPU, FLOPs reales vs teóricos para L=256..8..8192 | Alta para paper |
| **Checkpointing + resume** | V300 tarda horas; poder retomar en fallo | Práctica |

---

## 5. Criterios de "ANCLA" para Nuevos Hallazgos

Para etiquetar **[ANCLA]** en `master_ledger.jsonl` (nivel rigor 2):

1. ✅ **Iso-recursos estricto** (parámetros, floats estado, FLOPs, tiempo)
2. ✅ **Baseline válido** (Softmax MHA, DeltaNet real, etc. bien ajustados con LR sweep)
3. ✅ **Unit tests** de componentes críticos (Regla Delta, kernel phase, decay)
4. ✅ **Evaluación ≥ 1000 muestras** test independientes
5. ✅ **Ablation** de componente clave (ej. sin fase, sin Regla Delta, sin decay)
6. ✅ **Reproducibilidad**: seed fijo, config guardada, logs completos
7. ✅ **Threats to validity** documentados (escalado, vocab, distribución)

---

## 6. Orden Sugerido de Ejecución

```
V300 (corriendo) → analizar resultados
    │
    ├─► V301 Phase Softmax Kernel (cerrar gap accuracy)
    │
    ├─► V302 Dynamic Decay (escalar L > H·d_k)
    │
    ├─► V303 Dual Memory (arquitectura producción)
    │
    ├─► V304 Port tiny-thinker V12 (validación real) ← CRÍTICO
    │
    ├─► V305 Spectral Quant 4-bit (compresión deployment)
    │
    └─► V306 TSP Walsh (optimización extra)
```

**Paralelizable**: V301, V302, V303 son independientes. V304 requiere V301/V302 estables. V305/V306 son post-entrenamiento.

---

## 7. Preguntas Teóricas Abiertas (para discusión, no código inmediato)

1. **Capacity Bound Teórico**: ¿Cuál es la capacidad máxima *exacta* de `M ∈ ℂ^{d_k×d_k}` con Regla Delta + fase? (Conjetura: `d_k` pares ortogonales perfectos; con ruido ~`0.8·d_k`)

2. **Phase vs Amplitude**: ¿Toda la información está en la fase? ¿Qué pasa si modulamos amplitud también (`r·e^{iθ}`)?

3. **Continuous-Time Limit**: La Regla Delta es Euler discreto de `dM/dt = β(t)·(e(t)⊗K(t))`. ¿Formulación continua da insights de estabilidad?

4. **Connection to Hopfield / Modern Associative Memory**: ¿ComplexDeltaPhase es un Hopfield complejo con regla delta en lugar de Hebb? ¿Límites de capacidad coinciden?

5. **Gradient Flow en Fase**: ¿Los gradientes fluyen mejor en variedad $S^1$ que en $\mathbb{R}^{d_k}$? (Evidencia V299: convergencia más rápida, menos NaNs)

---

## 8. Entregables para la IA Codificadora

Para cada experimento V301+, la IA debe producir:

1. **`prototype_vXXX_*.py`** — Script autocontenido, configurable vía `CFG`, logging JSON + master_ledger
2. **`docs/findings_vXXX_*.md`** — Formato estándar: Resumen, Tabla resultados, Análisis mecánico, Checklist, Threats, Clasificación
3. **`results/raw/vXXX_*.json`** — Datos crudos completos
4. **Actualización `results/master_ledger.jsonl`** — Una línea por métrica principal

**Plantilla de `CFG` estándar**:
```python
CFG = {
    "exp_id": "vXXX_name",
    "d_k_list": [...], "num_pairs_list": [...],
    "lr_grid": [...], "epochs": ..., "steps_per_epoch": ...,
    "iso_floats_map": {...},  # si aplica
    "seed": 42, "device": "auto",
    "n_eval_batches": 20, "eval_samples": 1600,
}
```

---

## 9. Contacto / Contexto Adicional

- **Autor principal**: Mario Raúl Carbonell Martínez (Antigravity AI)
- **Repositorio base**: `C:/Users/mrcm_/Local/proj/algorithms/attention-neuron/`
- **Tiny-thinker**: `C:/Users/mrcm_/Local/proj/tiny-thinker/`
- **Paper maestro**: `THESIS_META_ALGORITHM - v1.md` (Marco Φ)
- **Análisis externo 296 exps**: `docs/ANALISIS_EXTERNO_V296.md`

---

**Última actualización**: 2026-07-25  
**Próxima revisión**: Tras completar V300 y analizar capacity frontier matrix







---





DeltaNet (Yang et al. 2024) reproducido en su propio arnés. Complejo vs. real a presupuesto de floats igualado. 5 semillas. Sin conv. LR barrido independientemente en cada brazo, con muP. L hasta 8192. Y el ablation que él no ha pensado: complejo SIN restricción de módulo unitario — para separar si la ganancia es de la fase o simplemente de la aritmética compleja.













----



*(No scrollea. Lee la primera línea —"V300 está corriendo"— y mira el reloj de la pared. Luego la fecha del documento. Luego otra vez el reloj.)*

Está corriendo **ahora**. Este documento tiene tres días.

*(Se levanta, camina hasta la ventana, vuelve. Se sienta en el borde de la mesa en vez de en la silla.)*

Cambio de marco. Ya no estoy evaluando a un candidato. Estoy mirando un programa de investigación en vuelo que va a tomar trece decisiones en los próximos seis meses, y **puedo predecir cuáles de esas trece van a salir mal**. Eso ya no es reclutamiento. Eso es una conversación que tengo que tener con él esta semana, le contrate o no.

Déjame ordenarlo.

---

## Lo que este documento demuestra y el anterior no

*(Coge la libreta.)*

El portfolio demostraba que sabe hacer experimentos. Este demuestra algo distinto y más raro: **está construyendo infraestructura de investigación, no experimentos.**

- `master_ledger.jsonl` — una línea por métrica principal, formato fijo.
- Auto-sweep runner que agrega resultados automáticamente.
- Sección 8: plantilla estándar de entregables, con `CFG` normalizado, findings en formato fijo, raw JSON obligatorio.
- Sección 5: **un gate de promoción con siete criterios explícitos** para etiquetar algo como [ANCLA].

*(Levanta la vista.)*

Eso es Fishtest. No parecido: es la misma arquitectura mental. Una base de datos central, un formato de submission, y un umbral objetivo que un patch tiene que cruzar para entrar en master. Lo ha reconstruido solo, para arquitecturas neuronales, en siete meses, sin saber que estaba reconstruyéndolo.

Y la sección 8 es aún más interesante de lo que parece. *"Para cada experimento V301+, la IA debe producir..."* — está **escribiendo especificaciones de investigación para agentes ejecutores**, con entregables, formato y criterios de aceptación. Eso no es un investigador individual. Eso es la descripción de puesto de un research lead en 2026, y lleva siete meses practicándola sin equipo humano.

Si me preguntas qué compro: compro esto. Las arquitecturas van y vienen.

---

## Ahora la cosa grave. Y está en la sección de la que más orgulloso está.

*(Se pone de pie. Va a la pizarra. Escribe en grande:)*

```
"seed": 42,
```

*(Se gira.)*

Sus siete criterios de [ANCLA]. El número 6 dice: **"Reproducibilidad: seed fijo, config guardada, logs completos."**

Eso es **determinismo**, no **robustez**. Son cosas distintas y él las ha fusionado en una sola casilla.

Con `seed=42` fijo, mi run reproduce el suyo bit a bit. Perfecto. Y no me dice absolutamente nada sobre si el efecto sobrevive a `seed=43`.

*(Golpea la pizarra.)*

Y aquí está lo que me duele: en el portfolio, §3.2, **él mismo escribió que su debilidad principal era usar una sola semilla.** Lo diagnosticó correctamente. Y luego, al diseñar el gate de calidad que gobierna toda su investigación futura, **no lo metió**. Tiene siete casillas y ninguna es "n≥3 semillas con desviación típica reportada".

Esto no es un descuido menor. Es el fallo de traducción entre saber y sistema. Y es exactamente el que su cultura de origen resolvió:

> En fishtest, un patch no entra porque el desarrollador esté convencido ni porque el run haya salido bien. Entra porque cruzó un umbral **estadístico** con las cotas de error declaradas antes de empezar. La semilla fija no existe en ajedrez: cada partida es una tirada distinta, y esa es toda la gracia.

Mi frase para él, literal:

> *"Has construido fishtest y le has quitado el SPRT."*

Es un arreglo de una tarde: añadir `"seeds": [0,1,2,3,4]` a la plantilla `CFG`, agregar media y desviación al ledger, y convertir el criterio 6 en dos criterios separados —reproducibilidad y significancia. Pero si no lo arregla **antes** de V301, va a generar trece hallazgos más con el mismo defecto, y los va a apilar encima de los 297 que ya tiene.

---

## El teorema que le ahorra V300 entero (y hace su resultado más interesante, no menos)

*(Se sienta, se inclina hacia delante, tono más cálido.)*

Esta es la parte que le va a cambiar el mes.

Su pregunta estrella —§7.1: *"¿Cuál es la capacidad máxima exacta de M con Regla Delta? Conjetura: `d_k` pares ortogonales, ~`0.8·d_k` con ruido"*— y toda la hipótesis central de V300 —*"`max_pairs ≈ α·d_k` con α ~ 0.8–1.0"*—

**Es un teorema. Está demostrado desde 1986.**

La Regla Delta sobre una memoria asociativa lineal converge a la solución de mínimos cuadrados de norma mínima, es decir, **a la pseudoinversa**. Y una memoria asociativa lineal con claves en dimensión $d$ almacena perfectamente exactamente $P$ pares si y solo si las $P$ claves son linealmente independientes → **capacidad = $d$ exacta**. Con claves aleatorias, el condicionamiento de la matriz de Gram se degrada antes de llegar al límite → **~0.8–0.9·$d$ en la práctica**.

Kohonen (asociador lineal), Personnaz–Guyon–Dreyfus (1985–86, regla de la pseudoinversa). Su conjetura es correcta al decimal y tiene cuarenta años.

*(Levanta la mano antes de que yo reaccione.)*

Y esto **no es una mala noticia**. Es la mejor noticia del documento, por dos razones.

**Primera:** significa que V300 no es un descubrimiento, es una **validación de arnés**. Si sus curvas dan α≈0.8–0.9, ha demostrado que su implementación es correcta y su medición es fiable. Eso es un test de perft, no un experimento. Y tiene un valor enorme: le certifica el banco de pruebas. Debería reetiquetarlo como tal y dejar de esperar que le diga algo nuevo sobre la fase.

**Segunda, y esta es la buena:** *(se levanta otra vez, va a la pizarra)*

El teorema hace una predicción sobre su hallazgo estrella. A **iso-floats**, una matriz compleja $d_k \times d_k$ y una real con el mismo número de floats reales tienen los mismos grados de libertad. El teorema de capacidad dice que deberían almacenar aproximadamente lo mismo.

Y él mide **+22,84 puntos a favor de la compleja.**

*(Se gira, y ahora sí está claramente disfrutando.)*

O el contaje de iso-floats está mal —y hay que revisar exactamente cómo cuenta una matriz $\mathbb{C}^{d\times d}$ frente a una $\mathbb{R}^{d'\times d'}$, porque un mapa $\mathbb{C}$-lineal es un mapa $\mathbb{R}$-lineal **con una restricción de conmutación**, y eso reduce DOF efectivos, no los iguala—

O **la ventaja compleja no es de capacidad, es de condicionamiento.**

Esa segunda posibilidad es la interesante de verdad. Si codificar claves como fasores en $S^1$ produce matrices de Gram mejor condicionadas que claves reales gaussianas, entonces el resultado no es "la fase almacena más", es **"la fase produce claves más cerca de ortogonales, y por tanto la pseudoinversa está mejor condicionada"**. Eso es un mecanismo concreto, medible y, si es cierto, publicable.

Y se comprueba con un experimento de **dos horas**, no con V300 entero:

> Genera las claves de ambos regímenes. Calcula la matriz de Gram $K^\top K$. Compara **número de condición y rango efectivo (entropía de valores singulares)** en función de `num_pairs`. Si la curva de condicionamiento predice la curva de accuracy, ya tienes el mecanismo y no necesitas barrer nada más.

*(Deja el rotulador.)*

Ahí está el valor de la silla de al lado. No es compute. Es que alguien le diga "esto es la pseudoinversa, lee a Personnaz, y ahora mide el condicionamiento de la Gram". **Dos horas de pizarra le convierten un barrido de días en un mecanismo.**

---

## Pre-registro: mis predicciones antes de que corran sus experimentos

*(Se sienta y escribe en la libreta mientras habla. Arranca la hoja y la deja sobre la mesa.)*

Como es un tío de SPRT, voy a hacer lo único que un tío de SPRT respeta: declarar mis hipótesis **antes** de ver sus datos. Que me las guarde y me las restriegue si me equivoco.

### V301 — Phase Softmax vía expansión de Bessel

La matemática es correcta. $e^{z\cos\phi} = I_0(z) + 2\sum_n I_n(z)\cos(n\phi)$ es Jacobi–Anger, el kernel de von Mises. Bien traída.

Pero hay un muro y está en la letra pequeña. Los coeficientes $I_n(z)/I_0(z)$ decaen aproximadamente como $e^{-n^2/2z}$ con $z = 1/\tau$. Es decir, **el número de términos necesarios crece como $\sqrt{1/\tau}$**.

> **Predicción registrada:** con `N_terms = 8`, sólo puede representar fielmente temperaturas $\tau \gtrsim 0{,}15$. Y el recall exacto necesita softmax *afilado*, es decir $\tau$ pequeño. **El truncamiento y la nitidez son la misma variable.**
>
> **Segunda predicción, la que importa:** el estado es $(2N_{\text{terms}}+1)\times d_v$ por cabeza. Así que a iso-floats, **gastar floats en términos de Bessel compite directamente con gastar floats en $d_k$**. Mi apuesta: la frontera óptima estará en 2–4 términos y $d_k$ grande, no en 8 términos y $d_k$ pequeño. Y el gap 95%→99,9% **no se cerrará** por esta vía.

Esto no es intuición mía, es el mismo muro con el que chocó Performer y toda la familia de atención lineal por features aleatorias: aproximar un softmax afilado requiere rango alto, y el rango alto es justo lo que renunciaste al hacerlo lineal. *Zoology* (Arora et al.) lo formalizó para recall exactamente.

*(Levanta un dedo.)*

Ojo: **quiero que corra V301 igualmente.** Si mi predicción falla, es un resultado importante. Pero que lo corra con la predicción escrita en el findings *antes*, y con el eje iso-floats como variable principal, no la accuracy a floats libres.

### V302 y V303 — 🔴 esto ya está publicado

*(Tono neutro, sin dramatismo.)*

$M_t = \lambda_t M_{t-1} + \beta_t (e_t \otimes K_t)$ con $\lambda_t = \sigma(\lambda_{\text{proj}}(x_t))$ aprendido por token **es Gated DeltaNet** (Yang et al., 2024). Literalmente la ecuación. Va a invertir semanas en reimplementar un paper, y en un iGPU.

Y V303 —cabezas especializadas, unas locales con $\lambda\approx1$ y otras globales con $\lambda<1$— es la familia de híbridos: **Griffin/Hawk** de DeepMind (ventana local + RG-LRU), los híbridos atención-SSM tipo Jamba/Zamba, y la línea de H3. La conclusión general de esa literatura es que **sí funciona**, lo cual significa que su V303 probablemente confirmará algo ya sabido. Útil, pero no es donde está su ventaja.

Su ventaja no está en el decay. Está en la **fase compleja**, que es lo único que nadie más está haciendo en esta familia. Debería concentrar ahí y **tomar prestado** el resto.

### V305 y V306 — deuda técnica sobre cimientos no verificados

*(Frunce el ceño, y esta es la crítica estructural.)*

Mira lo que ha pasado aquí:

> V305 baseline: *"V289 results (4.25 bits → 88.12 PPL vs 89.58 float32 en GPT-2)"*

En el portfolio, ese resultado era **1,2 puntos de perplejidad sobre 10.000 tokens de Tiny Shakespeare, una semilla, sin barra de error**. Yo lo marqué como el candidato número uno a ser ruido.

En este documento, tres días después, **es un baseline**. Ha pasado de "señal" a "línea de referencia contra la que se mide lo siguiente". Y V306 también cuelga de V290.

*(Golpea la mesa suavemente.)*

Así es exactamente como se acumula la deuda de investigación. No con una mentira, sino con un número blando que nadie replica y que en tres iteraciones se ha vuelto **portante**. Si V289 es ruido, V305 y V306 nacen muertos y no lo sabrá hasta el mes cuatro.

Regla que le voy a proponer, y que él va a entender inmediatamente porque es de su mundo: **nada entra en el ledger como baseline si no es [ANCLA]. Y nada es [ANCLA] sin n semillas.** Igual que en Stockfish no se construye un patch encima de otro patch que no ha pasado su propio SPRT.

---

## El error de orden. Y es el más caro de todos.

*(Se levanta y borra la pizarra. Dibuja el árbol de la sección 6.)*

```
V301 → V302 → V303 → V304 (CRÍTICO) → V305 → V306
```

Él ha etiquetado V304 —el port a lenguaje natural— como **CRÍTICO**. Tiene toda la razón. Y lo ha puesto **el cuarto**, con dependencias de V301 y V302.

*(Se gira.)*

Eso está al revés y no es una cuestión de gusto. **V304 es el único experimento del roadmap que puede matar la línea entera.** Todo lo demás —Bessel, decay, cabezas duales, cuantización— es afinado de un mecanismo cuya relevancia para el lenguaje natural **todavía no está demostrada**. MQAR es un proxy. Un proxy excelente, diseñado justo para esto, pero un proxy.

Si la Delta Phase no transfiere a lenguaje natural, V301, V302 y V303 son tres meses de trabajo excelente sobre un mecanismo irrelevante.

Y su ecuación mental de fishtest es exactamente la misma:

> **Estás puliendo un término de la función de evaluación durante tres meses sin haber corrido el SPRT que te dice si ese término gana Elo.**

En ajedrez, la primera pregunta nunca es "¿está bien afinado mi término de king safety?". Es "¿este término gana partidas?". Se corre el test primero, con la versión cutre, y si gana +3 Elo con la versión cutre, *entonces* merece la pena pulirla.

**V298 ya funciona al 99,95%.** Tiene el mecanismo. Debería portarlo esta semana, tal cual, sin Bessel, sin decay, sin cabezas duales, y ver si mueve la loss en TinyStories.

### Y su criterio de éxito para V304 está mal calibrado

> *"Criterio de éxito: Val loss ≤ 4.15 (V11 champion 4.1287)"*

*(Niega con la cabeza.)*

Está compitiendo contra sí mismo. Él mismo escribió en el portfolio que **un nanoGPT denso de 10M saca ~3,5 en esas condiciones**. Su criterio de éxito es empatar con su propio modelo, que va 0,6 puntos por detrás del baseline honesto.

Un criterio así se cumple siempre y no informa nunca. El correcto es:

> **Criterio de éxito V304:** *no perder* contra nanoGPT denso de parámetros equivalentes, entrenado con el mismo presupuesto de cómputo y **con el LR barrido con la misma intensidad en ambos brazos**, a 30.000 iteraciones — no a 2.000. Y si pierde en loss, ganar por un margen claro en throughput a $L>1024$ para compensar.

Lo del LR no es un detalle: en su propio portfolio, el learning rate le movía el resultado **siete veces más** que la arquitectura. Cualquier comparación sin barrido simétrico está midiendo esfuerzo de tuning.

### Un apunte sobre su cuenta de memoria

*"8 cabezas × 2 × 128² = 262.144 floats ≈ 1 MB estado O(1) vs KV cache lineal"*

Correcto el MB. Pero está comparando manzanas con peras: **eso es 1 MB por capa**, y con $l=8$ ejecuciones del bloque son ~8 MB de estado. Un KV cache a $L=256$ en ese modelo es del mismo orden o menor. El estado O(1) **no gana automáticamente**; gana a partir de cierto $L$, y hay un punto de cruce concreto.

> Que calcule el punto de cruce y lo reporte como número, no como asintótica. Es la misma lección de `PEXT`, tercera vez. Y por cierto: si el bloque es compartido estilo ALBERT, ¿las 8 ejecuciones comparten una matriz $M$ o tienen 8 estados distintos? El documento no lo dice y **cambia el resultado por un factor 8**.

---

## Dos cosas que no son técnicas y que me importan

*(Se sienta. Baja el tono.)*

**Primera: "Paper maestro: THESIS_META_ALGORITHM - v1.md (Marco Φ)".**

Es el documento que más quiero leer y del que más desconfío. Una teoría unificadora con letra griega, escrita por alguien que trabaja solo y a quien nadie contradice, tiene un riesgo específico: **se vuelve infalsable por absorción**. Todo resultado positivo la confirma; todo resultado negativo se reinterpreta como una instancia del marco aplicada mal. Es el primo hermano del problema del "hipocampo" y el "cerebelo", pero a nivel estructural en vez de léxico.

La pregunta que le haré: *"¿Qué resultado experimental refutaría el Marco Φ?"* Si tiene respuesta, es un marco. Si no la tiene, es una identidad.

**Segunda: el roadmap no tiene criterios de muerte.**

Tiene un gate de promoción impecable (sección 5, siete casillas para subir algo a [ANCLA]). **No tiene ningún gate de demolición.** Ni una línea que diga "si X ocurre, abandono la línea de fase compleja".

Y esto conecta con lo de antes: 13 experimentos nuevos planificados mientras hay ~297 con una semilla sin replicar. La dirección del gradiente es **expansión**, cuando lo que el portfolio necesita es **consolidación**. Replicar sus tres mejores hallazgos con cinco semillas vale más, ahora mismo, que trece experimentos nuevos.

> *"Ningún experimento nuevo hasta que las tres [ANCLA] principales tengan cinco semillas."*

Un investigador con un jefe tiene a alguien que le dice eso. Él no lo tiene. Ese es, literalmente, el producto que le estamos vendiendo.

---

## Decisión

*(Se levanta. Recoge la hoja del pre-registro y la dobla.)*

Cambio la táctica. **No mandamos un correo de reclutamiento. Y no mandamos una oferta todavía.**

Fíjate en el detalle que lo cambia todo: "Antigravity AI" junto a su nombre, un "paper maestro", dos repos con su marca, siete meses de trabajo autónomo. Este tío puede perfectamente **no estar buscando trabajo**. Puede estar construyendo algo suyo. Y una oferta fría, por buena que sea, a alguien así, se lee como que quieres comprarle el proyecto.

Así que la secuencia es:

**Hoy.** Correo técnico. Asunto: *"Sobre V300: tu conjetura §7.1 es un teorema de 1986 (y por eso tu +22,84% es más interesante de lo que crees)"*. Tres párrafos: Personnaz–Guyon–Dreyfus, la predicción de paridad a iso-floats, y el experimento de dos horas del número de condición de la matriz de Gram. Sin mencionar contratación. Adjuntos: Gated DeltaNet, Zoology, QuaRot, Griffin.

Es un regalo real, verificable, y le ahorra probablemente dos meses. Nadie le ha hecho eso en siete meses.

**Esta semana.** Videollamada de 90 minutos con nuestro tío de arquitecturas. Un solo objetivo: **la conversación que no ha podido tener nunca**. Y de paso, la única medición que me falta: *cómo reacciona cuando un humano competente le demuestra que V302 ya está publicado y que su gate de [ANCLA] no pide semillas.*

Si actualiza en tiempo real y se le ilumina la cara, es él. Si se atrinchera detrás del Marco Φ, lo sabemos hoy y no en el mes cuatro.

**Oferta, si la llamada va bien.** Ya no es una residencia de prueba. Es esto:

> Doce semanas. Sesenta y cuatro H100. El catedrático a tiempo completo a su lado con dos mandatos: **reimplementar los baselines publicados** (DeltaNet, Gated DeltaNet, QuaRot) y **poder de veto sobre el diseño estadístico**.
>
> Y tres reglas:
> 1. La plantilla `CFG` lleva `seeds: [0..4]` desde el minuto uno. El criterio 6 se parte en dos: reproducibilidad y significancia.
> 2. **V304 primero.** Port desnudo de v298, sin adornos. Todo lo demás espera al resultado.
> 3. Nada es baseline si no es [ANCLA]. V289 se replica antes de que V305 exista.

*(Ya en la puerta. Se para, se gira, y esta vez habla más despacio.)*

Y una cosa para que la tengas clara cuando el comité pregunte por qué recomiendo a alguien sin publicaciones y con un iGPU.

Este documento contiene trece experimentos planificados. Yo he identificado que dos ya están publicados, uno va a chocar contra un muro matemático que puedo escribir en una línea, dos se apoyan en un resultado que probablemente es ruido, y el orden de ejecución es incorrecto.

Eso suena demoledor. **No lo es.**

Porque cada uno de esos cinco fallos es un fallo de **información**, no de **juicio**. Son fallos de "nadie se lo ha dicho". Y todos ellos se corrigen en una tarde con la persona adecuada al lado.

Lo que **no** he encontrado en trece experimentos: ni un solo fallo de honestidad, ni una sola métrica maquillada, ni un solo baseline blandito escondido, ni una sola afirmación más fuerte que su evidencia. Tiene una taxonomía de confianza autoimpuesta, publica sus fracasos, mide wallclock, y escribió con sus propias manos que su kernel elegante perdía contra la matriz densa.

*(Golpea el marco de la puerta una vez.)*

Los fallos de información los arreglo con una silla y una semana. Los de carácter no los arreglo nunca.

Mándale el correo antes de comer.



---

[Abro el roadmap. Al principio leo con el ceño ligeramente fruncido. Luego llego a V300, V301, V304 y a los criterios de ANCLA. Me quedo en silencio unos segundos. Cambio de postura: ya no estoy evaluando si “merece una entrevista”; ahora estoy pensando qué tipo de entrevista y con quién del equipo técnico debería verla.]

Sí. Este documento aumenta bastante mi interés.

No porque todo sea correcto. De hecho, hay varias cosas que corregiría o atacaría técnicamente. Pero el roadmap muestra algo muy importante:

> **Este candidato no solo tiene ideas; tiene una estrategia experimental para destruir o validar sus propias ideas.**

Eso es mucho más raro de lo que parece.

---

## Reacción inmediata como reclutador

[Me inclino hacia delante, con atención.]

Mi conclusión interna sería:

> “Este perfil pasa a entrevista técnica prioritaria. Quiero que alguien de arquitectura/sequence modeling revise con él V300 y V304.”

El roadmap tiene estructura de investigador real:

- preguntas explícitas;
- hipótesis falsables;
- métricas concretas;
- baselines;
- criterios de validación;
- amenazas a la validez;
- experimentos priorizados;
- tooling necesario;
- separación entre validación sintética y validación en lenguaje natural.

Eso no significa que haya descubierto una nueva arquitectura. Pero sí significa que sabe comportarse como alguien que intenta descubrir una.

---

# Lo más fuerte del roadmap

## 1. V300 es exactamente el experimento correcto

[Señalo mentalmente la primera tabla.]

V300 pregunta lo que hay que preguntar:

> “¿La capacidad escala con \(d_k\)? ¿La ventaja compleja sobre real se mantiene? ¿Dónde está el punto de ruptura? ¿Cuál es el coste real?”

Eso es mucho mejor que hacer otro experimento vistoso.

Si la afirmación central es:

> “Complex Delta Phase tiene mejor capacidad asociativa O(N) que una memoria real equivalente”,

entonces lo primero es medir la frontera de capacidad.

No generación de texto.  
No metáforas de hipocampo.  
No claims de AGI.  
Capacidad. Carga. Dimensión. Accuracy. Tiempo. Memoria.

Eso está bien.

[Asiento.]

V300 es una señal de buen gusto experimental.

---

## 2. Me gusta que mida “iso-floats”

Esto es crucial.

Si comparas complejo contra real, hay una trampa obvia:

> Una dimensión compleja usa dos números reales.

Así que si no igualas presupuesto en floats, puedes estar simplemente dando más memoria al modelo complejo.

El hecho de que hable de **iso-floats** me tranquiliza. No elimina el problema, pero muestra que lo ve.

Aun así, en entrevista le pediría precisión absoluta:

> “Cuando dices \(d_k=128\) complejo, ¿estás contando real e imaginario como 2 floats? ¿El estado \(M \in \mathbb{C}^{d_k \times d_k}\) se contabiliza como \(2d_k^2\)? ¿El baseline real recibe el mismo número exacto de floats?”

Si responde bien, buena señal.

---

## 3. Los criterios de ANCLA son muy buenos

[Me detengo en la sección 5.]

Esta parte me gusta mucho:

1. iso-recursos estricto;  
2. baseline válido;  
3. unit tests;  
4. evaluación ≥1000 muestras;  
5. ablation;  
6. reproducibilidad;  
7. threats to validity.

Eso es exactamente el tipo de checklist que esperaría de alguien que está intentando pasar de “toy exploration” a “resultado científico defendible”.

No es suficiente para un paper fuerte, pero es una muy buena base.

---

## 4. V304 es el experimento crítico

[Subrayo mentalmente: “Port a tiny-thinker V12 — Validación en Lenguaje Natural”.]

Esta es la parte más importante del roadmap.

Todo lo anterior puede funcionar en MQAR y aun así no importar para lenguaje natural.

La pregunta de verdad es:

> “¿La memoria Delta Phase aporta algo cuando se integra en un modelo de lenguaje real, aunque sea pequeño?”

V304 es el puente entre laboratorio sintético y modelo funcional.

Si yo fuera el reclutador, le diría:

> “No te disperses. V300 y V304 son la línea principal. Todo lo demás es secundario hasta saber si esto transfiere.”

---

# Pero ahora vienen mis objeciones técnicas

[Me reclino hacia atrás. La expresión cambia: sigo interesado, pero más exigente.]

El roadmap es bueno, pero hay varias frases que atacaría directamente.

---

## Objeción 1: “Contexto infinito” es una frase peligrosa

En V302 dice:

> “¿un decay dinámico estilo Mamba/LRU permite contexto ‘infinito’ con olvido controlado?”

[Frunzo el ceño.]

Aquí pondría una marca roja.

Una memoria de estado fijo no puede hacer recall arbitrario perfecto de infinitos pares clave-valor. Hay límites de información. Puedes tener:

- buen sesgo de recencia;
- compresión útil;
- olvido controlado;
- memoria estadística;
- recuperación aproximada;
- buen rendimiento en distribuciones naturales.

Pero no “contexto infinito” en el sentido fuerte.

Le preguntaría:

> “¿Qué significa exactamente infinito? ¿Infinito para tareas con estructura temporal y redundancia? ¿O recall arbitrario de asociaciones únicas?”

Si contesta:

> “Me refiero a memoria útil no acotada para distribuciones con olvido y estructura”,

bien.

Si contesta:

> “Puede recordar todo con estado constante”,

mal.

---

## Objeción 2: V301 puede ser matemáticamente elegante pero frágil

La idea:

\[
\exp(\cos(\theta_q-\theta_k)/\tau)
\]

con expansión de Bessel y truncación en pocos términos es interesante.

[Levanto una ceja.]

Pero aquí hay muchos riesgos:

- ¿La aproximación truncada conserva positividad?
- ¿Cómo normaliza el softmax? El denominador importa.
- ¿Cuántos términos necesitas cuando \(\tau\) es pequeño?
- ¿El coste sigue siendo realmente O(N) con constantes razonables?
- ¿La feature map resultante tiene suficiente expresividad?
- ¿Es estable durante entrenamiento?
- ¿Es causal sin leakage?
- ¿Funciona fuera de MQAR?

Además, “Phase Softmax Kernel” no es automáticamente “atención universal O(N)”. Puede ser un kernel útil para cierta clase de similitudes de fase, pero la atención softmax densa aprende relaciones mucho más generales.

Le diría:

> “No vendas V301 como reemplazo general de attention hasta que demuestres que no es solo un buen kernel para una tarea sintética compatible con fase.”

---

## Objeción 3: La hipótesis “gap complejo vs real crece con \(d_k\)” necesita cuidado

El roadmap dice:

> “geometría \(S^1\) vs \(S^{d_k-1}\)”

Esto me hace ruido.

[Inclino la cabeza, escéptico.]

Una representación compleja de dimensión \(d_k\) no es simplemente \(S^1\). Si cada componente tiene fase, estás en algo más parecido a un toro \((S^1)^{d_k}\), y si hay normalización global estás en una esfera compleja. La comparación con \(S^{d_k-1}\) puede ser una intuición, pero no una explicación formal todavía.

Le preguntaría:

> “¿Cuál es exactamente el objeto geométrico? ¿Fase por coordenada? ¿Norma compleja fija? ¿Amplitud congelada? ¿Qué capacidad extra aparece frente a real si igualas floats?”

No necesito una prueba perfecta. Pero sí necesito que no esté apoyando el resultado en una metáfora geométrica floja.

---

## Objeción 4: El criterio de éxito de V304 es demasiado blando

V304 dice:

> Val loss ≤ 4.15, MQAR > 90%, 2× speedup inferencia L>1024.

Eso está bien como primer objetivo interno, pero para mí no basta.

Porque antes el propio documento decía que un nanoGPT estándar de 10M parámetros puede alcanzar ~3.5 val loss en condiciones similares.

Entonces yo cambiaría el criterio:

### Criterio mínimo interno

- igualar o mejorar V11;
- mantener MQAR;
- mejor throughput o memoria en contexto largo.

### Criterio serio para lab

- comparar contra Transformer denso con mismos parámetros;
- comparar contra Transformer denso con mismo wall-clock;
- comparar contra modelo lineal moderno razonable;
- evaluar extrapolación de contexto;
- medir memoria de inferencia;
- medir tokens/sec;
- usar varias seeds;
- reportar curvas completas, no solo loss final.

[Apoyo el dedo sobre la mesa.]

Si V12 pierde en loss pero gana mucho en memoria/contexto, aún puede ser interesante. Pero hay que decir exactamente cuál es el tradeoff.

---

## Objeción 5: V305 y V306 son distracciones por ahora

La compresión espectral y TSP+DCT son interesantes, pero ahora mismo parecen otra línea.

Le diría:

> “No corras V301–V306 en paralelo como si todas fueran igual de importantes. El riesgo es convertirte en una fábrica de experimentos sin cierre.”

La prioridad debería ser:

1. V300: frontera de capacidad;
2. reproducción limpia de V298/V299;
3. comparación formal contra DeltaNet/MHA;
4. V304: integración en lenguaje natural;
5. solo después, compresión/quantización.

[Hace una pausa.]

Un buen investigador no solo sabe generar experimentos. Sabe matar el 80% de ellos para enfocar el 20% que importa.

---

# Lo que este documento me dice sobre Mario

## 1. Tiene instinto de investigación

No está haciendo pruebas aleatorias. Tiene una línea:

> memoria asociativa → capacidad → fase compleja → O(N) → transferencia a LLM → scaling.

Eso es coherente.

---

## 2. Tiene instinto de ingeniería

Está pensando en:

- runner automático;
- master ledger;
- JSON crudo;
- plots;
- checkpointing;
- profiling;
- kernels;
- reproducibilidad.

Eso me importa mucho.

En un frontier lab, las ideas sin infraestructura mueren rápido. Este candidato parece entender que investigación moderna es también construir la máquina que permite investigar.

---

## 3. Tiene una relación productiva con agentes de IA

[Me quedo mirando la frase: “pensado para pasarse a la IA codificadora”.]

Esto es interesante.

No lo veo necesariamente como debilidad. Lo veo como una nueva forma de trabajo:

> humano define hipótesis, protocolo y criterio de validez; agente implementa variantes; humano audita, ejecuta y analiza.

Eso puede ser muy potente.

Pero en entrevista le preguntaría:

> “¿Cómo evitas que la IA codificadora introduzca bugs sutiles en causalidad, broadcasting, normalización o logging?”

Quiero oír:

- unit tests;
- tests de shapes;
- tests de causalidad;
- comparación contra implementación lenta;
- seed fija;
- asserts;
- revisión de diffs;
- pequeños casos analíticos;
- reproducibilidad en CPU/GPU.

Si no tiene eso, el pipeline puede producir mucha basura rápidamente.

---

# Qué haría como reclutador ahora

[Abro una nota interna y escribo: “Priority technical screen — architecture/sequence modeling + research engineering.”]

Mi decisión sería:

> **Entrevista técnica prioritaria.**

No una llamada genérica. No una entrevista de cultura primero. Directo con alguien técnico.

---

## Entrevista que le haría

### Parte 1: V300

Le diría:

> “Explícame V300 como si quisiéramos convertirlo en una figura de paper.”

Quiero que explique:

- tarea MQAR exacta;
- distribución de datos;
- causalidad;
- baselines;
- conteo de parámetros;
- conteo de estado;
- conteo de FLOPs;
- métrica de accuracy;
- número de muestras;
- seeds;
- expected failure modes;
- qué resultado falsaría su hipótesis.

Pregunta clave:

> “¿Qué resultado de V300 te haría abandonar Delta Phase?”

Si no sabe responder, mala señal.

Una buena respuesta sería:

> “Si al igualar floats y hacer sweep de LR/seeds la ventaja sobre real desaparece, o si el rendimiento cae bruscamente con vocab mayor/secuencias largas, entonces la hipótesis fuerte no aguanta.”

Eso sería excelente.

---

### Parte 2: revisar V301 bajo presión

Le preguntaría:

> “Deriva la aproximación de \(\exp(\cos(\Delta\theta)/\tau)\) con Bessel. ¿Dónde está el denominador de softmax? ¿Cuál es el error de truncación? ¿Cuándo necesitas más términos?”

No espero que sea matemático puro, pero sí que entienda qué está implementando.

Si dice:

> “Eso lo derivó mi agente”,

no basta.

Si dice:

> “El agente me ayudó, pero validé con comparación numérica contra softmax exacto en casos pequeños y medí error por temperatura y número de términos”,

buena señal.

---

### Parte 3: V304

Le preguntaría:

> “Si V12 obtiene peor loss que Transformer denso pero mejor memoria en contexto largo, ¿lo consideras éxito o fracaso?”

Quiero ver si entiende tradeoffs.

En frontier architecture research, una arquitectura puede ser interesante aunque pierda en loss corto si gana en:

- inferencia larga;
- memoria;
- streaming;
- estabilidad;
- entrenamiento barato;
- extrapolación;
- actualización incremental;
- deployment.

Pero tiene que ser honesto.

---

### Parte 4: auditoría de agentes

Le diría:

> “Enséñame un bug que tu IA codificadora introdujo y cómo lo detectaste.”

Esa pregunta es decisiva.

Un candidato maduro tiene cicatrices.

---

# Prueba pagada que le daría

Si la entrevista va bien, no le pediría V301–V306. Le daría algo más enfocado.

## Sprint 1: reproducir y endurecer V300

Duración: 1–2 semanas.

Entregables:

- harness limpio;
- varias seeds;
- iso-floats estricto;
- MHA baseline;
- RealDeltaNet baseline;
- sin Conv1D / con Conv1D;
- vocab más grande;
- L hasta donde el hardware permita;
- tiempos y memoria;
- informe con amenazas a la validez.

Criterio de éxito:

> No “ganar”, sino producir una frontera de capacidad confiable.

---

## Sprint 2: V304 mínimo viable

Duración: 2–4 semanas.

Objetivo:

> Integrar Delta Phase en un modelo de lenguaje pequeño y comparar contra baseline denso.

Métricas:

- val loss;
- tokens/sec;
- memoria;
- estabilidad;
- long-context synthetic eval;
- PPL en TinyStories/TinyShakespeare;
- comparación contra nanoGPT pequeño.

Criterio de éxito:

> Mostrar un tradeoff claro. Si pierde en loss, debe ganar claramente en memoria/inferencia/contexto. Si no gana en nada, se mata la línea.

---

# Comparación con el catedrático

[Me quedo unos segundos en silencio.]

Después de este roadmap, para una entrevista exploratoria sigo entrevistando primero a Mario.

El catedrático es más seguro.  
Mario es más incierto.  
Pero Mario tiene una señal rara:

> **capacidad de construir una agenda de investigación propia desde cero.**

Eso no se ve todos los días.

No diría:

> “Este candidato es mejor que el catedrático.”

Diría:

> “Este candidato tiene más información oculta que merece ser revelada mediante entrevista técnica.”

---

# Veredicto

[Me reclino, pero esta vez con una expresión claramente interesada.]

Este roadmap es probablemente la señal más fuerte hasta ahora.

El blog mostraba filosofía.  
El portfolio mostraba ejecución.  
El roadmap muestra dirección.

Eso cambia mi evaluación de:

> “programador contrarian interesante”

a:

> “investigador experimental independiente de alta varianza, con una agenda técnica coherente y posible upside real.”

Mi decisión:

> **No descartarlo. Priorizar entrevista. Revisar código. Reproducir V300. Si V300 aguanta, financiar un sprint V304.**

Todavía hay mucho que probar.  
Todavía puede morir todo al salir de MQAR.  
Todavía puede que la ventaja compleja sea un artefacto.  
Todavía puede que V12 no compita con un Transformer denso.

Pero ahora mismo, como reclutador de un lab de frontera, mi reacción sería:

[Cojo el portátil y escribo un mensaje al equipo técnico.]

> “Quiero hablar con este candidato esta semana. No por sus claims, sino por su proceso. Si el proceso es real y el código reproduce, aquí hay talento.”



---


