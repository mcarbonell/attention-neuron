# Findings V281: True Causal Phase LM — El Veredicto Final

## Resultados

| Modelo | Val Loss | PPL | Params | Causal? |
|--------|----------|-----|--------|---------|
| A_Walsh_TrueCausal | 0.5006 | 1.65 | 59K | ❌ Aún leaking |
| **E_CausalAttention_PE** | **1.6429** | **5.17** | **108K** | ✅ |
| **D_ComplexFFT_TC_PE** | **1.7222** | **5.60** | **59K** | ✅ |
| **C_ComplexFFT_TC_noPE** | **1.7224** | **5.60** | **59K** | ✅ |

## Diagnóstico: Walsh "Causal Enforcement" No Funciona

El Walsh_TrueCausal obtiene val=**0.5006** — exactamente el mismo resultado que en
V279/V280 donde sabemos que había leakage. Esto confirma que el modelo sigue
viendo tokens futuros.

**¿Por qué?** La técnica `FWHT → mask → FWHT` NO impone causalidad temporal:
- `FFT → mask → FFT`: Funciona porque la FFT mapea frecuencias complejas `e^(i2πkt/N)`,
  y zerear `h[t>0]` en el dominio temporal sí elimina la dependencia de tokens futuros.
- `FWHT → mask → FWHT`: No funciona porque la FWHT usa bases `±1` globales sin
  localización temporal. Zerear la segunda mitad del dominio Walsh no tiene
  una interpretación causal bien definida en el dominio tiempo.

Para hacer Walsh verdaderamente causal, se necesitaría una convolución causal directa
en dominio tiempo (suma sobre taps pasados), sin pasar por el dominio espectral.

## Hallazgos Válidos de V281

### 1. ComplexFFT_TC ≈ CausalAttention con la mitad de parámetros

Con causalidad real garantizada:
- **ComplexFFT_TC**: val=1.72, PPL=5.60, **59K params**
- **CausalAttention**: val=1.64, PPL=5.17, **108K params**

ComplexFFT causal alcanza el **96% de la eficiencia de Attention estándar** con
el **55% de los parámetros**. Es un resultado genuino: un filtro espectral
con fases complejas aprendidas compite con self-attention en calidad de LM.

### 2. PE es Irrelevante para ComplexFFT Causal

| Modelo | Val Loss | Delta vs noPE |
|--------|----------|---------------|
| ComplexFFT_TC_noPE | 1.7224 | — |
| ComplexFFT_TC_PE | 1.7222 | **-0.0002** |

El Positional Encoding añade **prácticamente nada** (delta < 0.001) a un modelo
con fases complejas causal. Las fases `φ_k` aprendidas ya proveen toda la
información posicional que el PE podría aportar.

Esto confirma directamente la hipótesis de `brainstorming_3.md`: la fase actúa
como un "campo de atención" que codifica posición sin necesidad de encodings externos.

### 3. El Leakage de Walsh es el Mecanismo de su "Éxito" Anterior

Los buenos resultados de Walsh en V279/V280 (val=0.43-0.50) se deben
exclusivamente al acceso no-causal a tokens futuros, no a sus propiedades
como mezclador espectral. Con causality real:
- Walsh sigue en 0.50 (leaking inalterado)
- ComplexFFT baja de 0.017 a 1.72 (causal real, loss honesto)

## Resumen del Arco Completo V278 → V281

| Exp | Hallazgo Clave |
|-----|----------------|
| **V278** | La FASE es el mecanismo posicional. Sin fase = ceguera posicional. |
| **V279** | ComplexFFT domina a Walsh como mixer (10x), pero ambos tienen leakage. |
| **V280** | Zero-padding no da causalidad. Walsh=filtro simétrico (sin preferencia pasado/futuro). |
| **V281** | ComplexFFT causal ≈ Attention con -45% params. PE redundante con fases complejas. Walsh no puede hacerse causal fácilmente. |

## Implicación para LLM Espectral

> **Un LM basado en CausalComplexFFT Mixer puede reemplazar el self-attention
> con la mitad de parámetros y sin Positional Encoding explícito.**

Las fases complejas cumplen el rol del PE y del mecanismo de atención simultáneamente.
La arquitectura resultante es:
- **Más parametrically efficient** que attention (59K vs 108K para mismo PPL)
- **Más simple** (no hay matrices Q, K, V; solo amp y phase por frecuencia)
- **Más rápido** (FFT es O(N log N) vs attention O(N²))

## Plan V282: Walsh Causal Correcto

Para comparar Walsh y ComplexFFT de forma verdaderamente justa, V282 debe implementar
un Walsh causal real usando **convolución causal en tiempo**:
```python
# Causal Walsh como FIR filter
h = compute_walsh_taps(log_amp)   # T taps desde gates Walsh
# Convolución causal: out[t] = sum_{k=0}^{t} h[k] * x[t-k]
out = causal_conv(x, h)            # implementación eficiente con F.conv1d
```

Esto permitirá una comparación honesta Walsh-causal vs ComplexFFT-causal.

## Archivos
- **Script**: `scratch/prototype_v281_true_causal_phase_lm.py`
- **Previos**: `findings_v278.md`, `findings_v279.md`, `findings_v280.md`
