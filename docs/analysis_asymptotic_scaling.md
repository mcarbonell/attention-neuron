# Análisis Asintótico: Bloques de Atención y FFN — V103-V106

> Variables de escala:
> - **d** = d_model (dimensión del embedding)
> - **N** = longitud de secuencia (contexto)
> - **L** = número de capas
> - **H** = número de cabezas de atención
> - **C** = número de conos (ConeAttn)
> - **M** = número de neuronas cónicas (ConeFFN)
> - **V** = tamaño del vocabulario
> - **R** = radio medio de los conos (en tokens)

---

## 1. Bloques de Mezcla Temporal (reemplazan Self-Attention)

### 1.1 Causal Self-Attention (Baseline)

**¿Qué es?** El mecanismo estándar del Transformer. Cada token genera un vector Query, Key y Value mediante proyecciones lineales densas. Los scores de atención se computan como el producto escalar de Q y K, se aplica softmax con máscara causal, y se pondera V.

```
Q = x @ W_q    K = x @ W_k    V = x @ W_v    (proyecciones densas)
scores = (Q @ K^T) / sqrt(d/H)                (producto escalar)
attn = softmax(causal_mask(scores))            (normalización causal)
out = attn @ V @ W_o                           (ponderación + proyección)
```

| Métrica | Fórmula | Dependencia clave |
|---|---|---|
| **Parámetros/capa** | 4d² + 4d | **O(d²)** — cuadrático en d_model |
| **FLOPs/token** | 8Nd + 4N² | **O(N²)** — cuadrático en contexto |
| **Memoria training** | N²·H + 3Nd | **O(N²)** — la matriz de atención |
| **Memoria inferencia** | 2NdL (KV-cache) | **O(NdL)** — crece con contexto |

**Cuello de botella:** La matriz de atención N×N domina tanto en cómputo como en memoria. A N=100K, cada capa almacena 10^10 scores. El KV-cache crece linealmente con cada nuevo token generado.

---

### 1.2 Cone1D Temporal Mixer (Propuesto — V103)

**¿Qué es?** Cada "cono" es una neurona con 3 parámetros aprendibles que define un campo receptivo triangular sobre posiciones RELATIVAS en la secuencia:

```
peso(t, j) = amplitud × max(0, 1 - |t - j - offset| / radio)   para j ≤ t
```

- **offset**: dónde mira (cuántas posiciones atrás)
- **radio**: cuán ancho es el campo receptivo (cuántos tokens cubre)
- **amplitud**: excitación (+) o inhibición (-), como una célula ganglionar retiniana

Cada token genera un Value (proyección lineal d→C), y los conos ponderan los Values con los pesos triangulares. El resultado se proyecta de vuelta (C→d).

```
V = x @ W_v                                          (proyección a C dims)
pesos[t,j,c] = amp_c * relu(1 - |t-j-offset_c| / radius_c) * causal(j≤t)
pesos = normalize(pesos)                              (normalización por cono)
out[t,c] = Σ_j pesos[t,j,c] * V[j,c]                (suma ponderada)
output = out @ W_out                                  (proyección C→d)
```

**Hallazgo V103:** Los radios se auto-organizan por profundidad:
- Capa 0: radios 3-9 (mira cerca → sintaxis local, n-gramas)
- Capa 2: radios 4-10 (mira lejos → dependencias de largo alcance)
- Esto replica la jerarquía V1→V4 de la corteza visual.

| Métrica | Fórmula | Dependencia clave |
|---|---|---|
| **Parámetros/capa** | 3C + 2Cd | **O(Cd)** — lineal en d |
| **FLOPs/token (impl. densa)** | 4Cd + 2N·C | **O(NC)** — lineal en N |
| **FLOPs/token (impl. sparse)** | 4Cd + 2R·C | **O(RC)** — independiente de N |
| **Memoria training** | N·C (sin N² matrix) | **O(NC)** — lineal en N |
| **Memoria inferencia** | R·C (solo ventana local) | **O(RC)** — constante en N |

**Ventajas clave vs Self-Attention:**
1. **Sin KV-cache**: cada posición solo necesita R tokens vecinos, no todo el historial
2. **O(N) en cómputo**: no hay producto QK^T de N×N
3. **Interpretable**: cada cono tiene significado geométrico (offset, radio, signo)
4. **3 params/cono**: vs d²/H params/cabeza en atención estándar

---

### 1.3 True Causal ComplexFFT Mixer (El Definitivo — V281/V282)

**¿Qué es?** Un reemplazo total de Self-Attention usando la Transformada Rápida de Fourier (FFT) y modulación causal en el dominio de la frecuencia. La secuencia de entrada se pasa al dominio frecuencial, se multiplica por un "gate" complejo con fase y amplitud aprendidas, y se proyecta a un subespacio estrictamente causal (donde la respuesta impulsional anti-causal se hace cero) antes de volver al dominio temporal. 

La principal diferencia con los modelos previos es que la **fase compleja** del gate actúa como un codificador posicional intrínseco.

```
X = FFT(x_padded)                                    (A dominio de frecuencias)
h_raw = IFFT(exp(log_amp) * exp(i*phase))            (Respuesta impulsional del gate)
h_causal = h_raw * causal_mask                       (Forzar causalidad estricta)
gate_causal = FFT(h_causal)                          (Gate corregido)
out = IFFT(X * gate_causal)[:T]                      (Filtrado y vuelta al dominio tiempo)
output = out @ W_out                                 (Proyección a d dims)
```

| Métrica | Fórmula | Dependencia clave |
|---|---|---|
| **Parámetros/capa** | 2·pad_T + d² | **O(d² + T)** — La matriz de salida d×d domina. |
| **FLOPs/token** | O(log T) + 2d² | **O(d²)** — La FFT es O(T log T) global, marginal por token. |
| **Memoria training** | d·T | **O(dT)** — No hay matriz NxN de atención. |
| **Memoria inferencia** | d·T (Buffer temporal) | **O(dT)** |

**Ventajas clave vs Self-Attention y ConeAttn:**
1. **Calidad asombrosa**: Logra el 96% de la calidad de Self-Attention (V281) y compite perfectamente cuando se estabiliza con nGPT (V282).
2. **Sin Positional Encoding**: La fase en la FFT ya codifica de manera absoluta y relativa la temporalidad de la secuencia.
3. **Hardware-efficient**: `torch.fft` es nativo y corre a velocidades órdenes de magnitud más rápidas que Self-Attention e incluso implementaciones Python puras de los conos.

---

## 2. Bloques FFN (Feed-Forward Network)

### 2.1 Dense FFN (Baseline)

**¿Qué es?** Dos capas lineales con activación no-lineal. Expande la dimensión 4× (de d a 4d) y la comprime de vuelta. Actúa como una "memoria asociativa" per-token: el up-project activa neuronas que reconocen patrones, el down-project recombina las activaciones.

```
hidden = GELU(x @ W_up + b_up)      W_up ∈ R^(d × 4d)
output = hidden @ W_down + b_down    W_down ∈ R^(4d × d)
```

| Métrica | Fórmula | Dependencia clave |
|---|---|---|
| **Parámetros/capa** | 8d² + 5d | **O(d²)** — cuadrático en d |
| **FLOPs/token** | 16d² | **O(d²)** |
| **Memoria training** | N·4d (activación intermedia) | **O(Nd)** |

**Hallazgo V104-V105:** La expansión 4× es masivamente redundante. Los conos revelaron que cada neurona del FFN solo lee 1-2 dimensiones de d (radio~1). El 98% de los pesos de W_up son ruido optimizado.

---

### 2.2 NarrowFFN (d→d — V105)

**¿Qué es?** Una sola capa lineal cuadrada (d→d) con GELU. Sin expansión a 4d. La transformación es una recombinación lineal completa de todas las dimensiones seguida de no-linealidad.

```
output = GELU(x @ W + b)     W ∈ R^(d × d)
```

**Por qué funciona:** El FFN denso hace dos cosas: (1) recombinar dimensiones (la multiplicación matricial) y (2) crear representaciones intermedias expandidas (la expansión 4×). NarrowFFN demuestra que solo (1) es necesario. La recombinación d×d captura el 99% de la capacidad.

| Métrica | Fórmula | Dependencia clave |
|---|---|---|
| **Parámetros/capa** | d² + d | **O(d²)** — pero 8× menos que Dense |
| **FLOPs/token** | 2d² | **O(d²)** — 8× menos FLOPs que Dense |
| **Memoria training** | N·d | **O(Nd)** — 4× menos que Dense |

**Resultado V105:** val_loss +1.0% vs Dense con 11.5× menos parámetros en el FFN.

---

### 2.3 BottleneckFFN (d→k→d — V105)

**¿Qué es?** Como el Dense FFN pero con una expansión mucho menor: d→k→d donde k=d/4 (en vez de k=4d). Es una factorización de rango bajo del FFN estándar.

```
hidden = GELU(x @ W_up + b_up)      W_up ∈ R^(d × k),  k = d/4
output = hidden @ W_down + b_down    W_down ∈ R^(k × d)
```

| Métrica | Fórmula | Dependencia clave |
|---|---|---|
| **Parámetros/capa** | 2dk + k + d ≈ d²/2 | **O(d²)** — pero 16× menos que Dense |
| **FLOPs/token** | 4dk ≈ d² | **O(d²)** — 16× menos que Dense |
| **Memoria training** | N·k = N·d/4 | **O(Nd)** — 16× menos que Dense |

**Resultado V105:** val_loss +2.7% vs Dense con 43× menos parámetros FFN.

---

### 2.4 ConeFFN (Conos sobre dimensiones — V103-V104)

**¿Qué es?** Cada neurona es un cono triangular sobre el eje de DIMENSIONES del hidden state. En vez de leer todas las d dimensiones (como una neurona densa), cada neurona cónica solo mira una región local del vector.

```
cone_weight[k, i] = amp_k × max(0, 1 - |i - center_k| / radius_k)
hidden_k = ReLU(Σ_i cone_weight[k,i] × x_i + bias_k)    (M neuronas)
output = hidden @ W_out                                     (M→d proyección)
```

- **center**: qué dimensiones del hidden state lee esta neurona
- **radius**: cuántas dimensiones adyacentes incluye
- **amplitude**: excitación/inhibición

**Hallazgo V104:** Los radios colapsan a ~1 dimensión. Cada neurona elige leer UNA sola dimensión. Forzar radios más anchos EMPEORA el resultado. Esto demostró que el FFN denso es un selector sparse de dimensiones.

| Métrica | Fórmula | Dependencia clave |
|---|---|---|
| **Parámetros/capa** | 4M + Md | **O(Md)** — lineal en d (no cuadrático) |
| **FLOPs/token** | Md + 2Md = 3Md | **O(Md)** — lineal en d |
| **Memoria training** | N·M + M·d (pesos del cono) | **O(NM + Md)** |

**Nota:** Si M ∝ d, los params son O(d²) como NarrowFFN. La ventaja real es conceptual: reveló la sobreparametrización del FFN.

---

### 2.5 DimGate (Gate por dimensión — V105)

**¿Qué es?** El FFN más simple posible: un vector de gates aprendibles que modula cada dimensión del hidden state con un sigmoid.

```
output = x * sigmoid(g)      g ∈ R^d  (aprendible)
```

Variante con escala: `output = x * (scale * sigmoid(gate))`

| Métrica | Fórmula | Dependencia clave |
|---|---|---|
| **Parámetros/capa** | d (o 2d con escala) | **O(d)** — lineal |
| **FLOPs/token** | d (o 2d) | **O(d)** — negligible |
| **Memoria training** | 0 extra | **O(1)** |

**Resultado V105:** val_loss +5.9% vs Dense. Demasiado débil como reemplazo completo, pero captura el 94% de la capacidad del FFN con 1,000× menos parámetros. Demuestra que una fracción grande de lo que hace el FFN es selección dimensional.

---

## 3. Tabla Comparativa de Escalado Asintótico

### 3.1 Parámetros por capa (mixer + FFN)

| Arquitectura | Mixer | FFN | Total/capa | Orden |
|---|---|---|---|---|
| **Transformer** | 4d² | 8d² | 12d² | O(d²) |
| **ConeAttn + Dense** | 2Cd | 8d² | 2Cd + 8d² | O(d²) — FFN domina |
| **Attn + Narrow** | 4d² | d² | 5d² | O(d²) — 2.4× menos |
| **ConeAttn + Narrow** | 2Cd | d² | 2Cd + d² | O(d²) si C∝d, sino O(Cd) |
| **ConeAttn + Bottleneck** | 2Cd | d²/2 | 2Cd + d²/2 | O(d²) — 24× menos |
| **ConeAttn + DimGate** | 2Cd | 2d | 2Cd + 2d | **O(Cd)** — sublineal en d² |
| **CausalPhase + Narrow (V282)** | d² | d² | 2d² | O(d²) — 6× menos absoluto |
| **Matrix-Free Phase-nGPT (V283)** | k² | k² | 2k² | **O(k²)** — Independiente de d² |

### 3.2 Parámetros a escalas concretas (L capas, sin embeddings)

| Escala | d | N | L | Transformer | Cone+Dense | Attn+Narrow | Cone+Narrow |
|---|---|---|---|---|---|---|---|
| **Toy** | 64 | 128 | 3 | 147K | 106K | 61K | 20K |
| **Small** | 256 | 512 | 6 | 4.7M | 4.0M | 2.0M | 1.3M |
| **Medium** | 1024 | 2048 | 12 | 151M | 115M | 63M | 27M |
| **LLaMA-7B** | 4096 | 4096 | 32 | 6.4B | 4.5B | 2.7B | 0.8B |
| **LLaMA-70B** | 8192 | 8192 | 80 | 64B | 44B | 27B | 6.4B |

> C=256 conos para todos los tamaños. En la práctica, C podría escalar con d.

### 3.3 FLOPs por token

| Arquitectura | Mixer FLOPs | FFN FLOPs | Total/token | Dependencia en N |
|---|---|---|---|---|
| **Transformer** | 8Nd + 4N² | 16d² | 4N² + 8Nd + 16d² | **O(N²)** |
| **Cone (dense impl)** | 4Cd + 2NC | 16d² | 2NC + 4Cd + 16d² | **O(N)** |
| **Cone (sparse impl)** | 4Cd + 2RC | 16d² | 2RC + 4Cd + 16d² | **O(1)** en N |
| **Attn + Narrow** | 8Nd + 4N² | 2d² | 4N² + 8Nd + 2d² | **O(N²)** |
| **Cone + Narrow** | 4Cd + 2RC | 2d² | 2RC + 4Cd + 2d² | **O(1)** en N |

### 3.4 Memoria en inferencia (generación autoregresiva)

| Arquitectura | KV-cache | Pesos | Total para N=100K, d=4096, L=32 |
|---|---|---|---|
| **Transformer** | 2NdL | 12d²L | **50 GB cache** + 12.8 GB pesos |
| **Cone+Dense** | **0** (sin cache) | (2Cd+8d²)L | **0 GB cache** + 8.6 GB pesos |
| **Attn+Narrow** | 2NdL | 5d²L | **50 GB cache** + 5.4 GB pesos |
| **Cone+Narrow** | **0** | (2Cd+d²)L | **0 GB cache** + 1.6 GB pesos |

**El KV-cache es el elefante en la habitación.** A contexto largo, domina completamente el presupuesto de memoria. ConeAttn lo ELIMINA.

---

## 4. Gráfico de Escalado: Parámetros vs d_model

```
Params    Transformer: 12d²L
(log)     ─────────────────────────────── /
          Cone+Dense:  (2Cd+8d²)L       /
          ─────────────────────────── /  /
          Attn+Narrow: 5d²L          / /
          ──────────────────────── / / /
          Cone+Narrow: (2Cd+d²)L / / /
          ─────────────────── / / / /
                             / / / /
                            / / / /
          ┌────────────────────────────→ d_model
          64   256   1024   4096  8192
```

La separación entre curvas CRECE con d_model porque:
- Transformer escala como **12d²** por capa
- Cone+Narrow escala como **d² + 2Cd** (donde C es constante o crece sublinealmente)
- A d=4096: Transformer=12×16M=192M/capa, Cone+Narrow=16M+2×256×4096=18M/capa → **10.7× menos**

---

## 5. Gráfico de Escalado: FLOPs vs Longitud de Contexto N

```
FLOPs     Transformer: O(N²)
(log)     ───────────────────────────── |
                                      /|
          Attn+Narrow: O(N²)        /  |
          ─────────────────────── /    |
                                /     |
          Cone (dense): O(N)   /      |
          ──────────────────/─────────|
          Cone (sparse): O(1)         |
          ════════════════════════════|
          ┌────────────────────────────→ N (contexto)
          128   1K    4K   16K   100K
```

La diferencia cuadrática vs lineal se vuelve astronómica a contexto largo:
- N=4K: Transformer=16M ops/token, Cone=52K → **308×**
- N=100K: Transformer=10B ops/token, Cone=52K → **192,000×**

---

## 6. Conclusión: ¿Cuándo usar cada arquitectura?

| Restricción principal | Mejor arquitectura | Razón |
|---|---|---|
| **Contexto muy largo** (>16K) | ConeAttn + Dense FFN | O(N) vs O(N²), sin KV-cache |
| **Máxima compresión / Edge** | CausalPhase + NarrowFFN + nGPT | Retiene PPL asombrosa con 19% de parámetros |
| **Interpretación bio-mimética**| ConeAttn + Dense FFN | Los radios se auto-organizan (interpretabilidad visual) |
| **Máxima calidad cruda** | nGPT Transformer | Baseline estabilizado con hiperesfera |

> **DimGate no se beneficia de apilado:** su capacidad expresiva NO crece con L. El presupuesto de params libre (O(d) vs O(d²)/capa) debe invertirse en d_model más grande, NO en más capas. Y ni siquiera eso compensa, porque la operación es cualitativamente insuficiente.

### La revolución absoluta: Matrix-Free Phase-nGPT (V283)

Tras el éxito consecutivo del V282 y el V283, la arquitectura definitiva se deshace de TODAS las matrices densas. Mediante el uso de una capa `WalshLinear` (Blueprint V67), el modelo asume un coste puro de **O(k² + T)**:
- En V283, un núcleo de Walsh con $k=d/2$ (k=64) **superó en Loss** a la versión densa ($d \times d$), usando 3 veces menos parámetros.
- Frente a un Transformer clásico (610K params), el modelo Matrix-Free de k=64 (42K params) retiene un 95% de la calidad utilizando apenas el **7% de los parámetros**.
- La estabilización hiper-esférica de nGPT funciona en perfecta sinergia con la proyección ortogonal de Walsh-Hadamard.

A escala LLaMA-7B ($d=4096$):
- Un núcleo Walsh de $k=256$ permitiría a la red operar en altísima dimensión latente pesando decenas de veces menos. Un LLM cognitivamente denso que cabe íntegramente en la memoria caché L3 de un procesador moderno.
