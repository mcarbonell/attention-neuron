# Findings V280: Causal Phase LM — Análisis de Leakage y Hallazgo Profundo

## Resultados

| Modelo | Val Loss | PPL | Wall Time |
|--------|----------|-----|-----------|
| C_CausalComplexFFT_noPE | **0.0171** | 1.0 | 140s |
| D_CausalComplexFFT_PE   | 0.0180 | 1.0 | 169s |
| B_CausalWalsh_PE        | 0.1979 | 1.2 | 517s |
| A_CausalWalsh_noPE      | 0.5006 | 1.6 | 524s |

## ⚠️ Problema Persistente: El Zero-Padding No da Causalidad Real

El "truco" de zero-padding `[x, 0...0] → FFT → gate → IFFT → take T` evita
la **convolución circular** (aliasing temporal), pero **no impone causalidad**.

La respuesta impulsional del filtro es `h = IFFT(gate)`. Si `h[t] ≠ 0` para
`t > 0`, el filtro sigue viendo el futuro. El gate `amp * exp(i*phi)` es
completamente arbitrario, por lo que `h` tiene componentes en ambas direcciones.

**Val loss = 0.017 sigue siendo imposible para un LM honesto con 59K params.**

## Hallazgo Profundo: Simetría Real vs Asimetría Compleja

A pesar del leakage, los datos revelan algo genuinamente importante.
Los modelos tienen acceso **igualmente no-causal** al contexto, pero sus
performances difieren en un factor ~10x. ¿Por qué?

### La respuesta impulsional de los gates reales es SIMÉTRICA

Para Walsh con gates reales positivos (`g_k = exp(log_amp_k) > 0`):

```
h[t] = IFWHT(g)[t]    (respuesta impulsional del filtro Walsh)
```

Como la FWHT es real y simétrica (la matriz de Hadamard es simétrica),
y los gates son reales positivos, la respuesta impulsional satisface:

```
h[t] = h[-t]    (filtro par / simétrico)
```

Un **filtro simétrico** da exactamente el mismo peso a la posición `t-k`
(pasado) que a `t+k` (futuro). **No puede preferir el pasado sobre el futuro.**
Es estructuralmente incapaz de ser un buen predictor causal.

### La respuesta impulsional de los gates complejos puede ser ASIMÉTRICA

Para FFT con gates complejos (`g_k = amp_k * exp(i*phi_k)`):

```
h[t] = IFFT(g)[t]    (respuesta impulsional compleja)
```

Con gates complejos, `h[t]` puede ser **completamente asimétrica**:
puede poner más peso en posiciones pasadas que futuras. El modelo APRENDE
a sesgarse hacia el pasado porque eso es lo que minimiza la loss.

```
Walsh (real): h[t] = h[-t]    → simétrico → future/past equally weighted
FFT (complex): h[t] ≠ h[-t]  → asimétrico → puede ser causal-biased
```

**Esta es la diferencia fundamental, no la explotación diferencial del leakage.**

## Jerarquía Actualizada con Causalidad

```
ComplexFFT: 0.017  (asimétrico, aprende bias hacia el pasado)
Walsh + PE: 0.198  (simétrico, siempre ve igual pasado y futuro)
Walsh sin PE: 0.500 (simétrico + sin info posicional)
```

El gap ComplexFFT → Walsh es **~12x en loss** en V280 (vs ~4x en V279).
Con zero-padding (convolución lineal vs circular), el Walsh es aún MÁS perjudicado
porque su simetría se impone más limpiamente.

## Resumen del Arco V278 → V280

| Experimento | Hallazgo |
|-------------|----------|
| V278 | La FASE es el mecanismo posicional. Sin fase (B_RealFFT), el modelo es ciego. |
| V279 | ComplexFFT domina a Walsh en mixing secuencial (~10x). Non-causal leakage. |
| V280 | El leakage persiste pero el mecanismo real es simetría: Walsh (real gates) = filtro par = no puede predecir bien. ComplexFFT (complex gates) = filtro asimétrico = puede sesgarse hacia el pasado. |

## Plan V281: Causalidad Real

Para eliminar completamente el leakage, necesitamos imponer que `h[t] = 0` para `t > 0`:

**Opción A — Causal Filter Enforcement:**
```python
h = torch.fft.irfft(gate, n=pad_T)  # respuesta impulsional
h_causal = h.clone()
h_causal[..., pad_T//2:] = 0        # zeroing de la parte no-causal
gate_causal = torch.fft.rfft(h_causal, n=pad_T)  # gate causal
```

**Opción B — Time-Domain Exponential Gates:**
Parametrizar el filtro directamente en dominio del tiempo como una
exponencial decayente: `h[t] = exp(-alpha * t)` para `t >= 0`, `0` para `t < 0`.
Esto garantiza causalidad por construcción y es diferenciable.

**Opción C — STFT Causal (Short-Time Fourier Transform):**
Aplicar FFT en ventanas deslizantes causales de tamaño W < T.
La posición t solo ve tokens en `[max(0, t-W), t]`.

**Hipótesis para V281**: Con causalidad real, ComplexFFT seguirá superando a Walsh
porque la asimetría de los filtros complejos sigue siendo una ventaja estructural,
independientemente del leakage.

## Archivos
- **Script**: `scratch/prototype_v280_causal_phase_lm.py`
- **Datos**: `scratch/data/tiny_shakespeare.txt`
- **Previos**: `findings_v278_phase_spectral_mixer.md`, `findings_v279_phase_lm_text.md`
