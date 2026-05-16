# V107: Iso-Budget Comparisons — Findings

## Resultados

### Budget A (~158K params)

| Config | Params | Layers | d | Val Loss | PPL | Tiempo |
|---|---|---|---|---|---|---|
| **Baseline** Attn+Dense d=64, L=3 | 158,400 | 3 | 64 | **1.5768** | 4.84 | 370s |
| NarrowFFN d=96, L=9 | 156,096 | 9 | 96 | 1.6132 | 5.02 | 829s |
| DimGate d=96, L=20 | 148,992 | 20 | 96 | 1.9720 | 7.19 | 1614s |
| DimGate d=160, L=12 | 156,672 | 12 | 160 | 1.9955 | 7.36 | 1143s |
| DimGate d=192, L=6 | 106,560 | 6 | 192 | 2.0672 | 7.90 | 596s |

### Budget B (~612K params)

| Config | Params | Layers | d | Val Loss | PPL | Tiempo |
|---|---|---|---|---|---|---|
| **Baseline** Attn+Dense d=128, L=3 | 611,712 | 3 | 128 | **1.5544** | 4.73 | 1027s |
| Attn+Narrow d=192, L=3 | 583,488 | 3 | 192 | 1.5769 | 4.84 | 858s |
| NarrowFFN d=192, L=12 | 627,840 | 12 | 192 | 1.5778 | 4.84 | 1616s |
| DimGate d=256, L=30 | 574,272 | 30 | 256 | 1.9238 | 6.85 | 4022s |
| DimGate d=128, L=30 | 288,576 | 30 | 128 | 1.9277 | 6.87 | 2873s |
| DimGate d=256, L=12 | 249,984 | 12 | 256 | 1.9822 | 7.26 | 1389s |

---

## Hallazgos

### 1. DimGate NO escala con profundidad — es matemáticamente colapsable

La razón fundamental: `DimGate(x) = x * sigmoid(g)` es una transformación **multiplicativa diagonal**. Componer L capas de esta operación es equivalent a UNA sola capa con gate acumulado:

```
x → x*g₁ → x*g₁*g₂ → ... → x * ∏ᵢ gᵢ = x * G_efectivo
```

**20 capas DimGate ≡ 1 capa DimGate** (mismo poder expresivo, diferente gate). La profundidad no crea representaciones nuevas — solo reescala las existentes. El modelo no puede "aprender" nada nuevo en las capas adicionales que no pudiera aprender en una.

Esto contrasta con NarrowFFN (d→d + GELU), donde la no-linealidad rompe la colapsabilidad: componer L capas NarrowFFN sí crea representaciones cualitativamente más ricas.

### 2. El aprendizaje se vuelve más lento con más capas — y por qué

Datos claros del budget A:

```
L=3  (baseline):  370s  → val=1.5768 ← ganador
L=9  (NarrowFFN): 829s  → val=1.6132 ← 2.2× más lento, peor
L=20 (DimGate):  1614s  → val=1.9720 ← 4.4× más lento, mucho peor
```

Tres mecanismos explican esto:

**A) Gradient dilution:** El gradiente atraviesa L LayerNorms y L conexiones residuales. A L=20, la señal efectiva por parámetro se diluye aunque los residuals mitigan el vanishing gradient.

**B) Saturación del sigmoid:** En DimGate, `sigmoid(g)` satura cuando `g → ±∞`, produciendo gradiente ≈ 0. Con 20 capas, la probabilidad de que al menos un gate esté saturado en la cadena de backprop es muy alta.

**C) Overhead sin capacidad:** Los parámetros de LayerNorm (2d por capa) son una fracción creciente del budget a L grande, y no aportan capacidad expresiva.

**D) Misma cantidad de steps, menos aprendizaje útil:** 200 steps/epoch × 20 epochs = 4,000 updates. Un modelo L=20 necesita muchos más updates para que el gradiente alcance todas las capas con suficiente señal.

### 3. NarrowFFN escala mejor con profundidad que DimGate

Comparación limpia (Budget B):
```
NarrowFFN d192 L3  (583K): val=1.5769  → 858s
NarrowFFN d192 L12 (628K): val=1.5778  → 1616s  ← misma calidad, 2× tiempo
```

NarrowFFN con más capas mantiene la calidad (1.5778 ≈ 1.5769) pero es 2× más lento. Esto es mucho mejor que DimGate (que empeora mucho con más capas), pero peor que el baseline (que logra mejor calidad en menos tiempo).

**La conclusión:** Para este task y esta escala (20 epochs, seq=128), el transformer d=128 L=3 es más eficiente en tiempo-de-entrenamiento que cualquier alternativa más profunda con el mismo presupuesto. La receta "pocas capas, mayor d" gana.

### 4. DimGate no es un FFN completo — es un componente de modulación

DimGate tiene valor real, pero como **gate auxiliar** dentro de una arquitectura mayor, no como reemplazo standalone del FFN. Una arquitectura viable:

```
h = NarrowFFN(x)          ← recombinación (O(d²) params)
h = h * sigmoid(g_dim)    ← supresión dimensional (O(d) params extra)
```

Esto añade solo `2d` params al NarrowFFN pero le da capacidad de suprimir dimensiones irrelevantes por capa. El gate aprende a decir "en esta capa, la dimensión 47 no importa".

---

## Revisión de la Tabla de Conclusiones

La sección 6 de `analysis_asymptotic_scaling.md` decía "DimGate: O(d) lineal". Eso sigue siendo correcto asintóticamente, pero hay que añadir el disclaimer:

> **DimGate no se beneficia de apilado:** su capacidad expresiva NO crece con L. El presupuesto de params libre (O(d) vs O(d²)/capa) debe invertirse en d_model más grande, NO en más capas. Y ni siquiera eso compensa, porque la operación es cualitativamente insuficiente.

## Conclusión Final de la Serie V103-V107

| Arquitectura | Viabilidad | Mejor uso |
|---|---|---|
| **ConeAttn** | ✅ Excelente | Mixer temporal, escala O(N), sin KV-cache |
| **NarrowFFN** | ✅ Excelente | FFN comprimido 11×, casi misma calidad |
| **ConeFFN** | ⚠️ Diagnóstico | Reveló sobreparametrización, no usar solo |
| **DimGate** | ⚠️ Auxiliar | Gate de modulación en combo con NarrowFFN |
| **DimGate standalone** | ❌ Inviable | Colapsable, no escala en profundidad |
