# Blueprint: Delta Rule — Validation Experiments (pre-v301)

Seis experimentos ligeros. Cada uno responde UNA pregunta y tiene un **go/no-go gate** explícito.

**Regla metodológica para TODOS:** n≥3 semillas. Reportar media ± desviación típica. Nada al ledger sin eso.

---

## EXP-1: Ablación Conv1D

**Pregunta:** ¿El éxito de v298 viene de la regla delta o del preprocesamiento Conv1D?

**Setup:** Factorial 2×2 sobre v298:

| | Sin Conv1D | Conv1D k=4 |
|---|---|---|
| Delta rule | A | v298 |
| Trivial (media V) | B | C |

- C: reemplazar delta rule por `V_pred = mean(V_{t-3..t-1})` — predice el próximo valor como promedio local
- Si C ≈ v298 → Conv1D lo explicaría todo
- Si A ≈ v298 → Conv1D es irrelevante
- Si A > C y A ≈ v298 → la delta rule aporta capacidad real

**Métrica:** Accuracy MQAR 64 pares, d_k=32 complejo, 5 semillas.

**Gate:** Si A pierde >5 pts contra v298, Conv1D es componente crítico — documentar y seguir, pero sabiendo que el resultado depende de preprocessing local.

---

## EXP-2: Condicionamiento de Gram (mecanismo de la ventaja compleja)

**Pregunta:** ¿La ventaja compleja (+22% en v299) es de *capacidad* o de *condicionamiento numérico*?

**Setup:** No entrena modelos. Solo genera claves y mide:

1. Genera N={8,16,32,64,128,256} claves en dos regímenes:
   - **Fase:** `k = exp(i·θ)` con θ ~ Uniform(0, 2π) — complejo unitario
   - **Real:** `k = L2_normalize(N(0,1))` — real en S^{d-1}
2. Iguala dimensión de floats: complejo d_k=32 (64 floats reales), real d_k'=45 (45 floats → 64 modos de Fourier ≈ comparable)
3. Calcula matriz de Gram G = K^H K (compleja) o G = K^T K (real)
4. Mide: **número de condición** κ(G) = σ_max/σ_min y **rango efectivo** (entropía de valores singulares)

**Métrica:** Curvas κ(G) vs num_pairs para ambos regímenes. Si κ_complex << κ_real a mismo num_pairs y las curvas correlacionan con accuracy de v299, la ventaja es de condicionamiento, no de capacidad.

**Duración:** ~2 horas CPU. No requiere GPU.

**Gate:** Si κ_complex ≈ κ_real a iso-floats, la ventaja no es de condicionamiento → buscar otro mecanismo (¿gradiente?, ¿representación?). Si κ_complex << κ_real, el mecanismo está identificado y puedes publicar: "fase → Gram mejor condicionada → más pares recuperables".

---

## EXP-3: MQAR Semántico (puente a texto real)

**Pregunta:** ¿La codificación de fase captura similitud semántica o solo identidad exacta?

**Setup:** MQAR modificado con estructura semántica en las claves:
- Vocabulario de 500 palabras con embeddings pre-entrenados (GloVe 50d)
- Cada clave se genera: `θ_k = W_proj @ e_{palabra}`, fase del embedding
- Test: para una query "perro", ¿recupera el valor asociado a "perro" (exacto)? ¿Y a "can" (sinónimo)? ¿A "hueso" (relacionado)?
- Control: mismo modelo con claves aleatorias (v298 original)

**Métrica:** Recall exacto vs recall semántico (sinónimos, hiperónimos, palabras relacionadas). Recall semántico medio > 2× baseline aleatorio (>12.5% para vocab 8 valores).

**Gate:** Si recall semántico ≤ aleatorio → las fases codifican identidad exacta, no estructura. La transferencia a texto real será pobre — reconsiderar o aceptar como mecanismo de cache exacto O(N).

---

## EXP-4: Escalado de Gradientes (profundidad temporal)

**Pregunta:** ¿A qué profundidad temporal muere el gradiente, y lo arregla Truncated BPTT?

**Setup:** Sobre v298 sin Conv1D (resultado de EXP-1) con secuencia larga (L=2048):
1. Mide `||∂L/∂M_0||` para L en {64, 128, 256, 512, 1024, 2048}
2. Repite con TBPTT K={8, 16, 32, 64}
3. Mide accuracy final en cada configuración

**Métrica:** Norma de gradiente vs L (debe decaer exponencialmente). Accuracy vs K.

**Gate TBPTT:** Si TBPTT K=32 mantiene >95% accuracy, el problema de gradientes está resuelto para ingeniería. Si ni K=64 funciona, la delta rule requiere repensar la arquitectura (Skip-Mem o similar).

---

## EXP-5: Mecanismos de Olvido (capacidad > d_k)

**Pregunta:** ¿Qué estrategia da mejor accuracy cuando M opera en sobrecapacidad (256 pares, d_k=32)?

**Setup:** 5 variantes sobre la misma base (complejo, d_k=32, H=2, 256 pares, L=2048):
1. **Sin olvido** (v298 original) → baseline
2. **Decay exponencial:** `M = λ·M + β·outer(err,k)`, λ∈{0.9, 0.99, 0.999}
3. **Clipping espectral:** power iteration para σ_max, si σ_max > τ, escala M
4. **Erasura por contenido:** antes de escribir, detecta la clave más similar ya almacenada en M y resta esa dirección (pseudoinversa online)
5. **Skip-Mem (vía doble):** M_fast (delta, truncada K=16, β alta) + M_slow (LRU λ=0.999, β baja)

**Métrica:** Accuracy por tramo (pares early vs mid vs late). La ideal mantiene accuracy plana en todos los tramos.

**Gate:** Si ninguna variante supera 70% en el tramo late (pares 193-256), la sobrecapacidad no se resuelve solo con olvido → necesitas jerarquía (múltiples M a diferentes escalas).

---

## EXP-6: Perfil MFU

**Pregunta:** ¿Cuánto del pico teórico de GPU aprovecha la delta rule vs softmax?

**Setup:** Modelo pequeño (d_model=256, H=4, d_k=64, 6 capas). Secuencias L={512, 1024, 2048, 4096}. Mide FLOPs reales / tiempo → MFU. Compara DeltaRule vs FlashAttention.

**Métrica:** MFU en función de L.

**Gate:** Si MFU(delta) < 0.25 × MFU(softmax) a L=4096, el cuello de banda secuencial está confirmado. Plan de mitigación: caching en SRAM, overlapped compute, o aceptar como coste de la compresión O(N).

---

## Orden y dependencias

```
EXP-2 (Gram, 2h CPU) ────────────────── paralelo
                                              │
EXP-1 (Conv1D ablation) ── gate ──────        │
                                              ↓
EXP-3 (Semántico) ─────── gate (¿generaliza?) │
                                              ↓
EXP-4 (Gradientes) ────── gate (¿es entrenable?)
                                              ↓
EXP-5 (Olvido) ────────── gate (escala?)      │
                                              ↓
EXP-6 (MFU) ───────────── gate (eficiencia?)  │
```

Los 4 primeros deciden **si** la delta rule es viable. Los 2 últimos deciden **cómo** escalarla.

---

## Metodología

- **3 semillas mínimo** por condición. Reportar μ ± σ, no solo best.
- **LR barrido independiente** por condición (el LR afecta 7× más que la arquitectura en tus datos).
- **Conv1D fijo o ausente** en todos menos EXP-1 (sabemos qué hace una vez corrido EXP-1).
- **Resultados al ledger** solo si cumplen: ≥3 seeds, baseline riguroso, threats documentados.
- **Si un gate dice "no"**: el experimento siguiente no se corre hasta decidir si se modifica el enfoque o se abandona la línea.