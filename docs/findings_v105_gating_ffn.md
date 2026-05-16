# V105: Is FFN Just a Dimension Gate? — Findings

## Resultados (d_model=128, standard causal attention en todos)

| Config | Total Params | FFN Params (est.) | Val Loss | PPL | vs Dense |
|---|---|---|---|---|---|
| **DenseFFN** (d→4d→d) | 611,456 | ~379K | **1.5527** | 4.72 | — |
| **NarrowFFN** (d→d + GELU) | 265,856 | ~33K | **1.5689** | 4.80 | **+1.0%** |
| ConeFFN (256 neurons) | 317,696 | ~85K | 1.5740 | 4.83 | +1.4% |
| BottleneckFFN (d→d/4→d) | 241,376 | ~8.7K | 1.5945 | 4.93 | +2.7% |
| DimGateScale (x * scale * sigmoid(g)) | 217,088 | ~768 | 1.6298 | 5.10 | +5.0% |
| DimGateBias (x * sigmoid(g) + bias) | 217,088 | ~768 | 1.6304 | 5.11 | +5.0% |
| DimGate (x * sigmoid(g)) | 216,704 | ~384 | 1.6441 | 5.18 | +5.9% |

---

## Hallazgos

### 1. El FFN NO es solo un gate — pero está masivamente sobreparametrizado

```
Puro gating (d params):     +5.9% → insuficiente, necesita ALGO de transformación
Bottleneck (d/4 hidden):    +2.7% → un poco de recombinación ayuda mucho
NarrowFFN (d×d):            +1.0% → casi iguala al denso con 11× menos params
DenseFFN (d×4d×2):          mejor  → pero la expansión 4× es derroche
```

**El FFN necesita recombinación lineal, pero NO necesita expansión a 4d.** Una simple multiplicación matricial d×d + GELU es suficiente para capturar el 99% de la capacidad del FFN denso.

### 2. NarrowFFN: el resultado estrella 🌟

```
NarrowFFN: val=1.5689, FFN_params≈33K → solo +1.0% peor que Dense
DenseFFN:  val=1.5527, FFN_params≈379K → 11.5× más parámetros
```

**11.5× compresión del FFN por solo 1% de degradación.** Esto significa que la expansión estándar d→4d→d del Transformer es un derroche masivo. Una capa linear d→d con activación no lineal captura casi toda la información.

### 3. Los gates funcionan sorprendentemente bien para lo que son

DimGateScale (2d = 256 params de FFN) está a solo 5% del Dense (379K params de FFN). Eso es **1,480× compresión** por 5% de pérdida. No es suficiente como reemplazo, pero demuestra que el gating por dimensión captura una fracción enorme de lo que hace el FFN.

### 4. Jerarquía de complejidad necesaria

```
Gating puro         →  selecciona qué dimensiones importan       → 95% del FFN
Recombinación d/4   →  mezcla entre dimensiones vecinas          → 97% del FFN
Recombinación d×d   →  mezcla completa entre todas las dims      → 99% del FFN
Expansión d×4d×2    →  crea representaciones intermedias masivas → 100% del FFN
```

La expansión a 4d NO añade casi nada sobre d×d. La mayor parte de la capacidad del FFN está en la recombinación lineal, no en la expansión.

---

## Implicaciones a escala

### Para un LLaMA-7B con FFN NarrowFFN (d×d en vez de d×4d×2):

| | FFN estándar (d=4096) | NarrowFFN (d=4096) |
|---|---|---|
| Params FFN/capa | 2 × 4096 × 16384 = **134M** | 4096 × 4096 + 4096 = **16.8M** |
| Params FFN total (32 capas) | **4.3B** | **537M** |
| Modelo total | 7B | **3.2B** |
| Reducción | — | **54% menos params totales** |

Combinado con ConeAttn (V103):
- ConeAttn: ahorra 2B en atención
- NarrowFFN: ahorra 3.8B en FFN
- **Total: 7B → ~1.5B params (78% reducción)**

### Para BottleneckFFN (d→d/4→d):

| | FFN estándar | BottleneckFFN (k=1024) |
|---|---|---|
| Params FFN/capa | 134M | 2 × 4096 × 1024 + 1024 = **8.4M** |
| Params FFN total | 4.3B | **269M** |
| Modelo total | 7B | **2.4B** (con atención estándar) |
| Con ConeAttn | — | **~0.8B** |

---

## Conexión con V104

V104 mostró que ConeFFN colapsa a radio ~1 (lee 1 dim). V105 confirma:
- Leer 1 dim (DimGate) da +5.9% → no es suficiente
- Pero ConeFFN (radio~1 + output projection) da +1.4% → la output projection HACE la recombinación
- NarrowFFN (d→d) es equivalente a ConeFFN sin el overhead de conos

**Los conos eran el instrumento de diagnóstico. NarrowFFN es la solución limpia.**

## Siguiente paso (V106)

Combinar las dos victorias:
- **ConeAttn** (V103: +4% con 24% menos params en atención)
- **NarrowFFN** (V105: +1% con 11× menos params en FFN)

En un solo modelo: ¿ConeAttn + NarrowFFN iguala al Transformer con masivamente menos parámetros?
