# V106: ConeAttn + NarrowFFN Combined — Findings

## Resultados (d_model=128)

| Config | Params | Val Loss | PPL | vs Baseline | Ahorro |
|---|---|---|---|---|---|
| **Attn+Dense** (baseline) | 611,456 | **1.5527** | 4.72 | — | — |
| **Cone+Dense** (V103) | 438,176 | 1.5668 | 4.79 | **+0.9%** | 28% |
| **Attn+Narrow** (V105) | 265,856 | 1.5689 | 4.80 | **+1.0%** | 57% |
| **Cone+Narrow** (PROPOSED) | 92,576 | 1.7172 | 5.57 | +10.6% | **85%** |
| Cone+Bottleneck | 68,096 | 1.8515 | 6.37 | +19.2% | 89% |

### d_model=64

| Config | Params | Val Loss | vs Baseline d64 |
|---|---|---|---|
| Attn+Dense d64 (baseline) | 158,272 | 1.5918 | — |
| Cone+Narrow d64 | 34,144 | 1.8580 | +16.7% |

---

## Hallazgos

### 1. Los ahorros NO se suman linealmente

```
ConeAttn sola:     +0.9% (vs baseline)
NarrowFFN sola:    +1.0% (vs baseline)
Esperado aditivo:  ~+2.0%
Observado combo:   +10.6% → HAY INTERACCIÓN
```

Cuando ambos componentes están simultáneamente debilitados, el modelo no puede compensar. Necesita al menos UN componente fuerte (o atención completa o FFN denso) para funcionar bien.

### 2. PEI del combo es MAYOR que baseline 🌟

```
Cone+Narrow:  PEI = 0.1173 (92K params)
Baseline:     PEI = 0.1113 (611K params)
```

**El modelo combinado es más eficiente POR PARÁMETRO.** Simplemente es demasiado pequeño en esta escala. La implicación: al escalar d_model, la brecha debería cerrarse porque el modelo gana capacidad más rápido que el baseline (PEI > baseline PEI).

### 3. Los ganadores individuales son los realmente impresionantes

| Arquitectura | Delta vs Baseline | Param Savings |
|---|---|---|
| **Cone+Dense** d128 | **+0.9%** | **28%** |
| **Attn+Narrow** d128 | **+1.0%** | **57%** |

Ambos componentes funcionan espectacularmente por separado. El mejor trade-off depende de qué importa más:
- **¿Ahorrar en inferencia?** → ConeAttn (O(N) vs O(N²), sin KV-cache)
- **¿Ahorrar en params?** → NarrowFFN (57% menos params, mantiene toda la atención)

### 4. Cone+Dense d128 SUPERA al Transformer d64

```
Cone+Dense d128:   val=1.5668, params=438K
Attn+Dense d64:    val=1.5918, params=158K
```

Si te interesa la CALIDAD y no los params absolutos, Cone+Dense a mayor d_model es la mejor opción: más calidad que un transformer pequeño, con O(N) scaling.

---

## Conclusión de la serie V103-V106

### Los dos descubrimientos independientes se mantienen:

1. **ConeAttn funciona como reemplazo de atención** (+0.9% a d=128)
   - Radios crecen con profundidad (V1→V4)
   - O(N) en contexto, sin KV-cache
   - 3 params por cono vs d² para Q/K/V

2. **FFN está masivamente sobreparametrizado** (V104-V105)
   - Los conos colapsan a 1 dim = el FFN es un selector sparse
   - NarrowFFN (d→d) captura el 99% con 11× menos params
   - La expansión 4× estándar es derroche

### Lo que NO funciona (aún):
- Combinar ambos a esta escala: la pérdida es superaditiva (+10.6% vs +2% esperado)
- El modelo necesita al menos un componente fuerte para compensar las limitaciones del otro

### Arquitectura recomendada:
Para producción, elegir UNA de:
- **ConeAttn + Dense FFN**: cuando el cuello de botella es la inferencia y el contexto largo
- **Attn estándar + NarrowFFN**: cuando el cuello de botella es el tamaño del modelo
