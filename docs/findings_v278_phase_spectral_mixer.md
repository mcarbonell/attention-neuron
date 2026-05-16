# Findings V278: Complex Phase Spectral Mixer — Positional Sensitivity

## Resumen del Experimento

Experimento diseñado para validar empíricamente la hipótesis del `brainstorming_3.md`:
> *"La fase de los coeficientes espectrales complejos codifica la posición temporal de patrones en una secuencia."*

## Configuración

- **Task**: "Single Spike Half Detection" — secuencias de T=64 ceros con un único 1 en posición aleatoria. Label = ¿está el spike en la primera mitad o la segunda?
- **Dataset**: 8,000 train / 2,000 val (sintético)
- **Arquitectura**: `Embedding(2, 64) → 3×SpectralMixer → CLS_pool(pos=0) → Linear(2)`
- **Parámetros comparables**: ~800 en todos los modelos espectrales

## Resultados

| Modelo | Mecanismo Posicional | Val Acc | Conv. Epoch | Tiempo |
|--------|---------------------|---------|-------------|--------|
| **C_ComplexFFT** | Fase analítica `e^(iφ)` | **100%** | **Ep 1** | 14.7s |
| A_RealWalsh | Signos ±1 del basis | 100% | Ep 2 | 51.2s |
| D_Dense (ref.) | Pesos libres | 100% | Ep 1 | 12.9s |
| **B_RealFFT (ablación)** | **Ninguno (fase=0)** | **53%** | Never | 17.9s |

## Hallazgos Clave

### 1. La Fase es el Mecanismo — Ablación Definitiva (Resultado Principal)

El modelo `B_RealFFT` aplica FFT real pero **zerifica la fase** antes de la síntesis inversa:
```python
X_noPhase = X.abs() * amp + 0j   # fase=0 forzada
```
Resultado: **53% para siempre** (aleatorio). Prueba empírica directa de la hipótesis:
> `|FFT(spike_at_t)[k]| = 1/N` para **todo** `t` y **todo** `k`.
El espectro de amplitudes de un spike es **plano e idéntico independientemente de la posición**.
Sin fase, ninguna arquitectura lineal puede distinguir posiciones.

### 2. Corrección a la Hipótesis Original: Walsh No es Ciego a la Posición

La afirmación inicial "Walsh es position-blind" era incorrecta. El análisis correcto:

- **FFT**: codifica posición a través de **fases complejas** `e^(-i2πkt/N)`
- **Walsh**: codifica posición a través de **patrones de signos** `H[k,t] ∈ {±1}`

La salida en posición 0 del RealWalsh mixer tras aplicar gates reales es:
```
out[0] = Σ_k H[k,t] * gate_k
```
Los patrones de signos `H[k,t]` difieren entre `t ∈ [0,31]` y `t ∈ [32,63]`, por lo que un gate real puede aprender a distinguirlos. Por eso RealWalsh también alcanza 100%.

**Conclusión refinada**: Ambas bases son position-aware, pero por mecanismos distintos. Lo que sí es ciego es el FFT sin fase.

### 3. Ventaja de Convergencia del ComplexFFT

- ComplexFFT converge en **Ep 1** vs Ep 2 de RealWalsh.
- ComplexFFT es **3.5x más rápido en wall time** (14.7s vs 51.2s).
  - Nota: parte del speedup es implementación (`torch.fft.rfft` nativo vs FWHT en Python puro).
- La convergencia más rápida sí es un efecto real: las fases complejas son un representación más directa de la posición (analítica vs combinatorial).

### 4. Implicación para LLM Espectral

En un Transformer estándar, el PE (positional encoding) se añade porque los pesos no codifican posición. Los resultados sugieren:

| Mixer | ¿Necesita PE explícito? | Mecanismo |
|-------|------------------------|-----------|
| ComplexFFT | Potencialmente NO | La fase `φ_k` aprende la localización temporal |
| RealWalsh | Potencialmente NO | Los signos ±1 codifican posición |
| RealFFT (sin fase) | **SÍ, siempre** | Ciego a posición |

> **Hipótesis para V279**: Un LM espectral con ComplexFFT mixing podría no necesitar positional encoding explícito para capturar dependencias posicionales en texto real, a diferencia de la arquitectura PSGT (V260) que requería PE como catalizador.

## Plan para V279

Escalar la hipótesis a texto real:
- **Tarea**: Clasificación de secuencias de texto donde el orden importa (ej. detección de n-gramas en posición específica).
- **Comparación**: ComplexFFT mixer SIN PE vs RealWalsh mixer SIN PE vs ambos CON PE.
- **Pregunta**: ¿Puede ComplexFFT prescindir del PE mientras Walsh lo necesita?

## Archivos

- **Script**: `scratch/prototype_v278_phase_spectral_lm.py`
- **Conceptual origin**: `docs/brainstorming_3.md` — Sección 4: Campos de Atención Cuántica
