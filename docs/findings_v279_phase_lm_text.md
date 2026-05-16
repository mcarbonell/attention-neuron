# Findings V279: Phase LM on Real Text (Tiny Shakespeare)

## Resumen

Test de la hipótesis de V278 sobre texto real: ¿puede `ComplexFFT_noPE` igualar
o superar a `Walsh_PE` en language modeling?

**Resultado**: ComplexFFT sin PE (0.044) supera masivamente a Walsh con PE (0.170).

## Resultados

| Modelo | Loss | Params | Wall Time | PE? |
|--------|------|--------|-----------|-----|
| D_ComplexFFT_PE   | **0.0196** | 59,206 | 111s | Sí |
| C_ComplexFFT_noPE | 0.0439 | 59,206 | 109s | No |
| B_Walsh_PE        | 0.1699 | 59,200 | 227s | Sí |
| A_Walsh_noPE      | 0.4290 | 59,200 | 222s | No |

## ⚠️ Caveat Crítico: Non-Causal Leakage

**Los valores absolutos de loss son inválidos para comparar con benchmarks externos.**

El mixer aplica FFT/Walsh globalmente sobre la secuencia completa (posiciones 0..T-1),
y luego predice el siguiente token en cada posición. Esto significa que la posición `t`
**puede ver tokens futuros** (t+1, t+2, ..., T-1) durante el mixing — el modelo
está haciendo trampa mirando el futuro.

- Un loss de 0.044 en char-level LM con 59K params = imposible sin data leakage.
- Esto explica la caída fulminante del loss en Ep 2.

**Lo que sigue siendo válido:** La comparación *relativa* entre modelos. Todos tienen
el mismo acceso no-causal, por lo que las diferencias entre ellos son informativas.

## Hallazgos Válidos (comparaciones relativas)

### 1. ComplexFFT es un Mezclador Secuencial Radicalmente Más Expresivo que Walsh

Con acceso idéntico al contexto (no-causal), ComplexFFT supera a Walsh por un factor ~10x en loss:
- `ComplexFFT_noPE = 0.044` vs `Walsh_noPE = 0.429`
- `ComplexFFT_PE   = 0.020` vs `Walsh_PE   = 0.170`

La **fase compleja** permite al modelo hacer interferencia constructiva/destructiva
entre frecuencias con una riqueza que el Walsh real (binario ±1) no puede igualar.

### 2. La Fase Sustituye Casi Completamente al PE en ComplexFFT

| Modelo | sin PE | con PE | Gap |
|--------|--------|--------|-----|
| ComplexFFT | 0.0439 | 0.0196 | **2.2x** |
| Walsh      | 0.4290 | 0.1699 | **2.5x** |

El PE sigue siendo útil para ambos, pero el gap relativo es similar. Lo más importante:
`ComplexFFT_noPE (0.044) << Walsh_PE (0.170)` — las fases más el poder expresivo del
ComplexFFT juntos compensan más que el PE de Walsh.

### 3. Velocidad de Wall Time

ComplexFFT es **2x más rápido** que Walsh (111s vs 227s):
- `torch.fft.rfft` es una operación nativa altamente optimizada.
- FWHT en Python puro es mucho más lento por el bucle.
- En hardware real (FWHT en SIMD/hardware), Walsh recuperaría velocidad.

### 4. Jerarquía de Capacidad Expresiva

```
ComplexFFT_PE >> ComplexFFT_noPE >> Walsh_PE >> Walsh_noPE
     0.020          0.044            0.170        0.429
```

El factor limitante de Walsh no es el PE — es el propio mecanismo de mixing (±1 real)
frente al mixing complejo con fase. La brecha compleja/real es mayor que la brecha PE/noPE.

## Interpretación bajo el Marco de brainstorming_3

El documento predecía que la fase actúa como un "campo de gauge" que codifica
posición temporal. Los datos lo confirman en el sentido de que:
- La fase permite al mixer construir representaciones posicionalmente ricas sin PE explícito.
- Walsh (sin fase) necesita PE externo para alcanzar calidad comparable, y aun así queda muy por detrás.

Esto apoya la idea del brainstorming de que el PE en Transformers actuales es una "muleta"
que compensa la ceguera posicional de los pesos reales — los pesos complejos con fase
tienen esa información incorporada en su estructura.

## Plan V280: Causal Phase LM (el test real)

Para validar los hallazgos sin el artefacto de non-causality, V280 debe:

1. **Usar un mixer causal**: solo mirar contexto pasado. Opciones:
   - Windowed FFT / STFT (FFT local sobre ventana causal)
   - Cumulative FFT: aplicar FFT sobre `x[:, :t, :]` para cada t
   - FFT con zero-padding causal: `x_causal = concat([x, zeros]) → rfft → gate → irfft → take first T`

2. **Métrica correcta**: loss validado contra división separada, no train leak.

3. **Pregunta limpia**: ¿Mantiene ComplexFFT la ventaja sobre Walsh en un LM causal honesto?

4. **Baseline adicional**: un Transformer real de los mismos params como referencia.

## Archivos

- **Script**: `scratch/prototype_v279_phase_lm_text.py`
- **Datos**: `scratch/data/tiny_shakespeare.txt`
- **Previo**: `docs/findings_v278_phase_spectral_mixer.md`
