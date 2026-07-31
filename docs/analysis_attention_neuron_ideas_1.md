Análisis por Claude Opus 4.6.
Fecha: 20/07/2026. Actualizado: 22/07/2026 (integración de v298/v299). Actualizado: 23/07/2026 (integración de v297 — pieza intermedia de la serie holográfica).

---

# Análisis Completo del Repositorio Attention-Neuron
## 297 Experimentos, 6 Eras, una Tesis

He leído en profundidad: la idea original (V1-V9), los brainstormings filosóficos (caverna de Platón, geometría de gauge, cone neurons para LLMs), los blueprints (DCT-LLM, Phase-Spectral Transformer, Holographic Hippocampus, Spectral Cerebellum, Scientific Neuron), los findings experimentales de cada era (V50 stroke neurons, V65 text JPEG, V101 cone neurons, V208 resonancia multiplicativa, V251 oligarquía, V282 Phase-nGPT, V286 Poincaré, V287 óptica conforme, V288-V290 compresión espectral, V292-V297 holografía y Phase Softmax, **V298-V299 Regla Delta de Fase y Frontera de Capacidad**), y los documentos maestros.

---

## La Tesis Central (La que vertebra todo)

> **"Los pesos son sombras. Lo que la red realmente aprende es dónde mirar."**

Expresado de formas distintas a lo largo de las 6 eras:
- **Era 1 (V1-V9):** "La neurona es la variable, no el peso"
- **Era 2 (V50-V101):** "4 parámetros por neurona son suficientes si codifican geometría"
- **Era 3 (V35-V65):** "Walsh/DCT son bases fijas; solo los coeficientes de atención se aprenden"
- **Era 4 (V208-V251):** "Los pesos congelados son el reservorio; los gates son la inteligencia"
- **Era 5 (V278-V290):** "La fase compleja codifica posición; las bases espectrales comprimen sin pérdida"
- **Era 6 (V292-V296):** "La conjugación de fase holográfica reemplaza la atención cuadrática"
- **Era 7 (V298-V299):** "La Regla Delta Matricial Compleja elimina la diafonía de raíz y demuestra capacidad superior por parámetro"

Es la misma idea expresada con vocabulario cada vez más preciso. Y eso le da **coherencia interna** al proyecto.

---

## Las Ideas por Orden de Potencial Disruptivo

### Tier S — Potencial de Cambio de Paradigma

#### 1. Memoria Holográfica O(N) por Regla Delta Matricial de Fase (v293-v299)

**Evolución completa de la línea holográfica:**

| Experimento | MQAR Acc | vs Softmax MHA | Complejidad | Etiqueta |
|---|---|---|---|---|
| v293 (Single Head, Hebbiana) | 18.94% (max 21.72%) | +3.63% | O(N) | [SEÑAL] |
| v294 (Multi Head H=16, Hebbiana) | 17.81% (max 22.34%) | +4.37% | O(N) | [SEÑAL] |
| v296 (Mass Normalized, Hebbiana) | 19.69% (max 23.59%) | +8.12% | O(N) | [SEÑAL] |
| **v297 (Phase Softmax + Forget Gate)** | **49.59%** | **+34.12%** | **O(N)** | **[SEÑAL]** |
| **v298 (Regla Delta Matricial Compleja)** | **99.95%** | **≡ Softmax MHA** | **O(N)** | **[ANCLA]** |

**Frontera de capacidad (v299) — iso-floats (~2,048 floats/cabeza):**

| Modelo | 8 Pares ($L=64$) | 64 Pares ($L=512$) | Degradación |
|---|---|---|---|
| **Compleja $\mathbb{C}$ (Delta Phase)** | **99.80%** | **95.98%** | **-3.82%** |
| Real $\mathbb{R}$ (DeltaNet Vanilla) | 99.67% | 73.14% | -26.53% |
| Vectorial $\mathbb{C}$ (Diagonal) | 89.36% | 4.48% | -84.88% |
| Softmax $O(N^2)$ (Control) | 99.63% | 99.73% | -0.10% |

**Por qué es Tier S — ahora confirmado empíricamente:** Es la única línea que ha demostrado empíricamente **igualar** a Softmax Attention (99.95% vs 99.95%) en recall asociativo con complejidad estrictamente $O(N)$. Y bajo presión de capacidad (64 pares), la codificación compleja retiene **+22.84%** más que la real con el mismo presupuesto de memoria de estado.

**Importancia del escalón v297:** La progresión 23% → 49% → 99.95% no es accidental. v297 demostró que el Forget Gate Selectivo Causal ($\lambda_t = \text{sigmoid}(W_\lambda x_t)$) y la normalización por fuerza de coincidencia de fase duplicaban el rendimiento respecto a la acumulación Hebbiana pura, pero también reveló un **techo infranqueable en ~50%**: la diafonía espacial de la suma lineal de fasores no se elimina con gating ni normalización. Este diagnóstico preciso — identificar que el cuello de botella estaba en el *write* (acumulación Hebbiana), no en el *read* (normalización) — fue exactamente lo que motivó la formulación de la Regla Delta en v298.

---

##### Análisis retrospectivo: por qué la Regla Delta es superior a la propuesta de Softmax Phase Spiking

En la versión original de este análisis (20/07/2026), la propuesta para romper el techo del 23% era:

```python
# Propuesta original (softmax post-retrieval):
R = softmax(Re(conj(Q) * M) / tau)
```

La intuición era correcta — la suma lineal de cosenos no tiene el "spiking" del softmax — pero el diagnóstico era incompleto. **v297 confirma empíricamente esta incompletitud:** al implementar exactamente un mecanismo análogo (normalización por fuerza de coincidencia de fase $S_t = |\bar{Q}_t M_{K,t}|$ como proxy de la función de partición del softmax, combinada con un forget gate selectivo), el resultado fue 49.59% — una mejora sustancial sobre el 23%, pero con un techo infranqueable en ~50%. El problema real no estaba en el **read** (unbinding), sino en el **write** (binding). La acumulación Hebbiana $M_t = \sum K_\tau V_\tau$ inyecta ruido de diafonía cuadráticamente con cada nuevo par, y ningún post-procesamiento en el read puede recuperar una señal que ya fue destruida en el almacenamiento.

v297 también probó **Power Sharpening** ($\text{sign}(u)|u|^\gamma$, $\gamma=3.0$), que degradó el rendimiento del 49.59% al 38.12%, confirmando que la causa del fallo no era la falta de nitidez en la atención sino la interferencia acumulada irreversible en el estado de memoria.

La Regla Delta Matricial de v298 ataca la raíz:

$$M_t = M_{t-1} + \frac{\beta}{d_k}(e_t \otimes K_t), \quad e_t = V_t - \text{Re}(M\bar{K}_t)/d_k$$

**Comparación mecanística:**

| Aspecto | Softmax post-retrieval (propuesta original) | Regla Delta Matricial (v298) |
|---|---|---|
| **Dónde actúa** | En el *read* — suprime interferencia después de leer | En el *write* — impide que la interferencia se escriba |
| **Diafonía acumulada** | Se acumula libremente en $M$; el softmax la enmascara pero no la elimina | Se corrige en cada paso; el residuo $e_t = 0$ para claves ya almacenadas |
| **Escalado con $N_{pares}$** | Degrada suavemente (el softmax necesita un contraste creciente $\tau$ para separar señal de ruido) | Se mantiene estable (la ortogonalización dinámica compensa la interferencia) |
| **Parámetros extra** | Añade $\tau$ como hiperparámetro sensible | Solo $\beta$ (learning rate de la regla delta), robusto |
| **Convergencia** | Incierta — el gradiente del softmax sobre una señal ruidosa puede ser inestable | Ultrarrápida — 2-4 épocas, igualando a Softmax MHA |
| **Elegancia** | Parche aplicado a la salida | Corrección aplicada a la causa |
| **Evidencia empírica** | **v297 confirmó: Phase Norm + Forget Gate → 49.59% (techo ~50%)** | **v298 confirmó: Delta Rule → 99.95% (sin techo observable)** |

**Veredicto:** La Regla Delta es la solución **correcta**, no solo la solución que funciona. v297 proporcionó la evidencia empírica directa: el softmax post-retrieval (o su proxy Phase Norm) mejoró el 23% al 49.59% pero se estancó — exactamente como predecía el diagnóstico de diafonía Hebbiana. La Regla Delta cierra la herida de raíz.

> [!NOTE]
> **Lección metodológica:** Cuando un sistema falla, la tentación es arreglar la salida (post-procesamiento). Pero la solución robusta casi siempre está en arreglar el almacenamiento. Esto es análogo a la diferencia entre "filtrar ruido en la lectura" vs "no escribir ruido". v298 eligió lo segundo.

---

#### 2. Compresión Espectral con Reordenación de Canales (v288 + v290)

| Método | Ratio 90% | Ratio 70% | Ratio 50% |
|---|---|---|---|
| Sin ordenar (Lowpass) | 163.95 PPL | Explosión | Explosión |
| **Greedy TSP + Lowpass** | **88.36 PPL** ✅ | 1302.39 | Explosión |
| Sin ordenar (Energía) | 90.61 | 98.95 | 155.61 |
| **PCA + Energía** | 90.17 | **96.20** | 158.20 |
| **Fiedler + Energía** | 90.76 | 101.56 | **114.71** |

**Por qué es Tier S:** El resultado de Greedy TSP (88.36 < 89.58 baseline float32) es genuinamente sorprendente. Estás mejorando la perplejidad eliminando el 10% de los coeficientes de alta frecuencia. Esto sugiere que esos coeficientes son **ruido de sobreajuste**, no señal.

Y lo hace sin:
- Dataset de calibración (a diferencia de GPTQ/AWQ)
- Reentrenamiento
- Hardware especial

**Lo que sugiero:** Probar en LLaMA-7B o Mistral-7B. Si Greedy TSP + lowpass al 90% mejora o mantiene la perplejidad en un modelo de 7B parámetros, tienes una herramienta de compresión que es trivial de implementar y compite en calidad con GPTQ.

---

### Tier A — Ideas Empíricamente Fuertes

#### 3. Gated Frozen Networks + Hipótesis de la Oligarquía (v251 series)

Las "leyes" descubiertas son el resultado más limpio del repositorio:

| Ley | Hallazgo | Implicación |
|---|---|---|
| Anti-Regulador | WD=0 es óptimo | La amplificación extrema de pocas neuronas es la solución, no la regularización |
| Discovery (Zero-Init) | Init=0 ≈ Init=1 | La red converge al mismo atractor independientemente del punto de partida |
| Oligarquía | $N_{eff}/D \approx 48\%$ siempre | La sparsity no es un artefacto, es una propiedad intrínseca de la solución |
| Escalado Sublineal | $N_{eff} \propto D^{0.9}$ | Más neuronas → menor fracción activa |

Y lo de v291 es sólido: se mantiene en Fashion-MNIST (85.32%), CIFAR-10 (42.99%), y 3 capas (96.48%). La universalidad del atractor de ~48% es robusta.

**Propuesta nueva:** ¿Has probado la Oligarquía con un backbone espectral (Walsh/DCT) en lugar de random? Si las gates seleccionan sobre un backbone ortogonal, la Participación Ratio podría bajar aún más (más compresión) porque las bases espectrales son más informativas que las random.

---

#### 4. Cone Neurons / Retinal Attention (v98-v102)

**94.30% en MNIST con 3,850 parámetros y 4 params por neurona** ($C_x, C_y, R, A$). Es el resultado más elegante del repositorio en cuanto a PEI.

Lo que me interesa más que MNIST es la **propuesta de Cone Attention para LLMs** del brainstorming: cada "head" de atención se reduce a un cono 1D sobre la secuencia con 4 parámetros (centro μ, ancho σ, amplitud, forma). La tabla comparativa con Mamba/RWKV/xLSTM es provocativa:

| Método | Params por token | ¿Aprende dónde mirar? | ¿Resolution invariant? |
|---|---|---|---|
| Full attention | N² | No (mira todo) | No |
| Mamba/SSM | O(d) | Sí (implícito) | Parcial |
| **Cone-1D** | **4-6** | **Sí (explícito)** | **Sí** |

**La pregunta clave no respondida:** ¿Se ha probado Cone-1D en una tarea de secuencia real (copy task, MQAR, Shakespeare character-level)? El brainstorming tiene código pero no vi un `findings_cone_attention_llm.md`. Esto debería probarse.

---

#### 5. Phase-nGPT (v282) — LLM Ligero Extremo

| Modelo | Params | Val Loss | Wall Time |
|---|---|---|---|
| Standard Transformer | 610K | **1.5630** | 1725s |
| **CausalPhase_nGPT_Narrow** | **116K** (19.2%) | 1.6762 | **735s** |

80% menos parámetros, pérdida competitiva, mitad de tiempo. La trinidad Hiperesfera + Resonancia de Fases + Gating Lineal es viable.

**Pero:** v292 demostró que el FFT mixer no puede hacer recall asociativo dependiente del contenido. Así que esta arquitectura es óptima para **tareas donde la posición temporal importa más que el contenido** (generación de lenguaje con patrones rítmicos, código con indentación estructurada), pero no para tareas que requieren "encontrar la aguja en el pajar".

---

#### 6. Neuronas de Trazos (v50) — 97.88% con Bézier

**99.2% de reducción de parámetros** en la capa de entrada (de 200K a 1,536). Los filtros visuales son curvas de Bézier diferenciables con contraste On-Center/Off-Surround.

**Lo más interesante:** Invarianza a la resolución. Al estar basada en funciones continuas (distancia a una curva), esta capa es teóricamente agnóstica a la resolución. Podría escalar de 28x28 a 1024x1024 sin añadir un solo parámetro.

---

### Tier B — Ideas Teóricamente Ricas, Empíricamente Incipientes

#### 7. Óptica Conforme (v287) — Lentes Gravitacionales

39.06% en MNIST con 3,082 params. Bajo, pero el concepto de deformar una textura congelada mediante mapas conformes es el más creativo del repositorio. Los pesos resultantes muestran patrones armónicos y regulares — es un regularizador espacial implícito.

**Conexión no explorada:** Combinar Óptica Conforme con bases Walsh. En lugar de una textura random congelada, usar una base Walsh como $W_{base}$ y deformar conformalmente sobre ella. Unificaría las eras 3 y 5.

#### 8. Poincaré Hyperbolic Attention (v286)

Superioridad sistemática de la atención geodésica hiperbólica sobre la euclidiana en **todas** las dimensiones testadas (de d=2 a d=64). En d=64, la brecha es +11.63% (43.49% vs 31.86%).

**Lo interesante:** La ventaja de Poincaré crece con la dimensión del embedding. En modelos grandes, el espacio hiperbólico podría dar ventajas aún mayores para representar jerarquías semánticas.

#### 9. Text JPEG (v65) — El "Alma" del Lenguaje

La observación de que al truncar los coeficientes DCT de una secuencia de embeddings, las bajas frecuencias preservan la **estructura gramatical** ("Subject Block → Action Block → Emotion Block") es profundamente sugerente. La propuesta de generación "coarse-to-fine" (primero la onda semántica, luego los detalles léxicos) es radical y potencialmente superior al autoregressive.

#### 10. Cerebelo Espectral + Early Exit (Blueprint)

Inferencia asimétrica basada en entropía predictiva. No hay datos experimentales pero la idea es sólida y práctica.

---

### Tier C — Ideas Que Abrieron Caminos Pero Se Toparon con Muros

#### 11. Neurona de Resonancia Multiplicativa (v208)

Fracasó por el Fenómeno de Gibbs (las funciones continuas no pueden aproximar discontinuidades sin infinitos armónicos). Pero el diagnóstico es valioso: necesitas **bases de discontinuidad** (floor, round) para funciones como el módulo. Esto limita las arquitecturas espectrales puras a funciones continuas.

#### 12. Neurona Polimórfica / Rosetta (v14/v22 via NEW_ALGORITHMS_BRAINSTORMING)

La idea de que una neurona pueda cambiar dinámicamente entre SUM, VAR, PRODUCT, SIN es intelectualmente atractiva. Es un "Mixture of Experts" a nivel de operación primitiva. No vi datos experimentales recientes.

#### 13. Multi-Pass Inference / Feedback Loops (NEW_ALGORITHMS_BRAINSTORMING §6)

Una red plana de 5 capas que se ejecuta 3 veces en bucle antes de responder. Es esencialmente lo que hacen los "thinking tokens" modernos (o1, Claude Sonnet), pero a nivel de arquitectura interna. Conceptualmente sólido.

---

## Resultados Negativos Importantes (Anclas Negativas)

| Experimento | Resultado | Lección |
|---|---|---|
| **v292** | Gating multiplicativo element-wise + FFT = ruido aleatorio en MQAR | La FFT no puede emular $QK^T$ content-based |
| **v294 Candidato 3** | Decaimiento exponencial LRU = colapso (4.06%) | El olvido temporal indiscriminado destruye memorias útiles |
| **v297 Power Sharpening** | $\text{sign}(u)|u|^\gamma$ ($\gamma=3$) degradó de 49.59% a 38.12% | La agudización no lineal del read no compensa la diafonía del write; la confirma |
| **v299 Elementwise** | Memoria diagonal compleja = colapso a 32+ pares (7.18% → 4.48%) | La estructura matricial de producto exterior es **indispensable** para recall de alta densidad |
| **v208** | Resonancia multiplicativa = Gibbs explosion en OOD | Las bases continuas (cos) no aproximan discontinuidades |
| **v5** | Red masiva 4096x4096 con 2 params/neurona = 11.35% | La "fuerza bruta dimensional" no funciona con modulación escalar |
| **v288 Lowpass** | DCT lowpass sin reordenar = colapso inmediato | Las altas frecuencias de los pesos de LLMs son esenciales... a menos que reordenes |

---

## Meta-Patrones que Emergen del Repositorio Completo

### 1. La Invarianza Paramétrica como Norte Verdadero
Desde V1 (2 params/neurona, 76%) hasta V101 (4 params/neurona, 94.3%) hasta V251 (1 gate/neurona, 94.27%), el proyecto ha demostrado repetidamente que la eficiencia paramétrica extrema es alcanzable. La pregunta ya no es "¿funciona con pocos parámetros?" sino "¿cuál es el prior geométrico correcto para cada dominio?".

### 2. Selección > Representación
Los gates multiplicativos (v251), los conos (v101) y las máscaras de atención espectral (v35) convergen en la misma verdad: **el acto de seleccionar qué activar es más importante que la calidad de lo activado**. La Oligarquía (48% activo, power-law) es la expresión estadística de esta verdad.

### 3. Compresión Espectral Funciona para Estado Estático, No para Routing Dinámico
- ✅ DCT/Walsh para sintetizar pesos (v64, v288, v290)
- ✅ DCT/Walsh para mixing temporal de posición fija (v282 Phase-nGPT)
- ❌ DCT/Walsh + gating para recall asociativo dependiente de contenido (v292)
- ✅ Conjugación de fase holográfica para recall asociativo O(N) (v293+)
- ✅ **Regla Delta Matricial Compleja = recall perfecto O(N) igualando Softmax MHA (v298)** ← nuevo
- ✅ **Fase compleja > real bajo presión de capacidad: +22.84% a 64 pares (v299)** ← nuevo

### 4. Write-side Correction > Read-side Filtering (meta-patrón confirmado por v297 + v298)
La línea holográfica demostró con evidencia empírica progresiva que la acumulación Hebbiana era el cuello de botella, no el mecanismo de retrieval. **v297 proporcionó la demostración negativa**: mejorar el read-side (Phase Norm + Forget Gate) llevó al 49.59% pero con un techo infranqueable en ~50%. **v298 proporcionó la demostración positiva**: corregir el write-side (Regla Delta) alcanzó el 99.95% sin techo observable. La combinación de ambos resultados convierte "Write-side Correction > Read-side Filtering" de hipótesis a **patrón confirmado empíricamente** con evidencia en ambas direcciones. Este patrón probablemente se generaliza a otros dominios de memoria asociativa.

### 5. Las Ideas Más Fuertes Son las Más Simples
V50 (Bézier = 4 params → 97.88%), V101 (Cono = 4 params → 94.3%), V251 (gate = 1 param → 94.27%), **V298 (Regla Delta = 1 corrección residual → 99.95%)**. Las ideas complicadas (óptica conforme, Poincaré, neuronas polimórficas, e incluso el Power Sharpening de v297) obtienen peores resultados con más complejidad. v297 lo ilustra a la perfección: el Forget Gate simple (+26%) superó al Power Sharpening elaborado (-11.47%). Y la Regla Delta confirma la tendencia a escala máxima: la modificación más pequeña posible al mecanismo de escritura (restar la predicción antes de escribir) produjo el salto más grande del repositorio (+50% sobre v297, +76% sobre la versión Hebbiana pura).

---

## Cruces que No He Visto en el Repositorio (Propuestas Nuevas)

| Cruce | Idea | Justificación | Estado |
|---|---|---|---|
| ~~Holografía + Softmax Phase Spiking~~ | ~~$R = \text{softmax}(\text{Re}(\text{conj}(Q) \cdot M) / \tau)$~~ | ~~Romper el techo del 23%~~ | **SUPERADA por v298** — La Regla Delta resolvió el problema de forma más elegante atacando el write en lugar del read. Ver análisis en §Tier S. |
| **Cone-1D + MQAR** | Probar Cone Attention 1D en la tarea MQAR que ya tienes montada | Verificar si 4 params/neurona funcionan para secuencias | **PENDIENTE** |
| **Greedy TSP + LLaMA-7B** | Reordenación + compresión DCT en un modelo de producción | Validar el hallazgo v290 a escala real | **PENDIENTE** |
| **Oligarquía + Backbone Walsh** | Gates sobre un backbone ortogonal en lugar de random | ¿Baja la Participation Ratio? ¿Sube el PEI? | **PENDIENTE** |
| **Phase-nGPT + Delta Phase Memory** | Reemplazar el CausalFFT Mixer por el `DeltaPhaseHolographic` de v298 | El mixer espectral no hace recall → el Delta Phase sí, y ahora al 99.95% | **VIABLE** — Prioridad alta tras v298 |
| **Cerebelo + Delta Phase** | Early-exit basado en entropía donde el "cerebelo" es el Delta Phase O(N) | Sistema 1 ultraligero con memoria real al 99.95% | **VIABLE** — Prioridad alta tras v298 |
| **Delta Phase + Decay Dinámico** | Añadir un mecanismo de olvido selectivo (no LRU global) a la Regla Delta | Escalar a $L > 2048$ donde el rango de $M$ se satura ($N_{pares} > H \times d_k$) | **NUEVO** — Amenaza 1 de v298 |
| **Delta Phase + Vocabulario Real** | Validar la ortogonalidad de fase con $N_{vocab} = 50,000$ | La proyección $\theta = W_k x$ debe mantener separación angular con vocabularios de LLM | **NUEVO** — Amenaza 2 de v298 |

---

## Resumen para Ti

**Actualización post v298/v299:** El cuello de botella que identifiqué en la versión original de este análisis (el techo del 23% en MQAR) ha sido **resuelto**. Y la solución fue más elegante que mi propuesta: en lugar de parchear la salida con un softmax, la Regla Delta Matricial de Fase Compleja eliminó la diafonía de raíz corrigiendo el mecanismo de escritura.

Tu repositorio tiene ahora **4 resultados empíricos [ANCLA]**:
1. **Memoria Holográfica Delta Phase O(N)** — 99.95% en MQAR, igualando Softmax MHA (v298)
2. **Superioridad de Fase Compleja sobre Real** — +22.84% a 64 pares bajo iso-floats (v299)
3. **Compresión espectral con reordenación** — Greedy TSP + lowpass mejora PPL eliminando el 10% de coeficientes (v290)
4. **Gated Frozen Networks + Oligarquía** — 48% de neuronas activas como atractor universal (v251)

Más **1 concepto geométrico elegante** (cone neurons) que aún no se ha probado en secuencias.

Las fronteras abiertas ahora son: (a) integrar Delta Phase en una arquitectura LLM completa (Phase-nGPT + Delta Phase), (b) escalar a secuencias largas $L > 2048$ con decay selectivo, y (c) validar con vocabularios reales de $N = 50,000$. La base matemática y empírica para hacerlo es sólida.
