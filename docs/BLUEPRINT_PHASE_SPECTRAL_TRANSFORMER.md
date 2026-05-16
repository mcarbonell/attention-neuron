# Blueprint: Phase-Spectral LLM (FFT Mixer + Walsh/DCT Weights)

> **Estado**: Validado experimentalmente en `attention-neuron` (V278–V281, Mayo 2026)
> **Destino**: `tiny-thinker` repository
> **Supera a**: `BLUEPRINT_DCT_LLM.md` — este documento lo extiende y reemplaza parcialmente.

---

## 1. La Idea Central: Dos Problemas, Dos Transformadas

La arquitectura DCT-LLM (blueprint anterior) resolvía un problema: **compresión de pesos**.
Este blueprint resuelve un segundo problema, más fundamental: **el mecanismo de atención**.

| Problema | Solución anterior | Solución nueva |
|----------|------------------|----------------|
| Las matrices W son enormes | DCT/Walsh sintetizan W desde pocos coeficientes | ✅ Mantener |
| Self-attention es O(N²) y ciego a posición | Positional Encoding + matrices Q,K,V densas | **FFT Causal Mixer** |

Los transformers actuales tienen dos cuellos de botella:
1. **El FFN**: 66% de los parámetros — ya resuelto con Walsh/DCT (V67)
2. **La atención**: O(N²), necesita PE explícito, usa matrices Q,K,V costosas

El **FFT Causal Mixer** reemplaza completamente el mecanismo de atención.

---

## 2. Base Experimental (lo que los datos dicen)

Los experimentos V278–V281 en `attention-neuron` demuestran:

### Hallazgo 1: La fase FFT codifica posición — sin PE explícito
`ComplexFFT_noPE` vs `ComplexFFT_PE`: delta = **0.0002** en val loss.
Las fases aprendidas `φ_k` en el dominio frecuencial sustituyen completamente al PE.

### Hallazgo 2: FFT Causal Mixer ≈ Attention con -45% de parámetros

| Modelo | Val Loss | PPL | Params | Mecanismo |
|--------|----------|-----|--------|-----------|
| CausalAttention_PE | 1.64 | 5.17 | **108K** | Self-attention estándar |
| **FFT_CausalMixer_noPE** | **1.72** | **5.60** | **59K** | FFT + fase causal |

96% de la calidad con 55% de los parámetros, sin PE. O(N log N) vs O(N²).

### Hallazgo 3: Walsh/DCT son ciegos a posición — inútiles como mixers temporales
Real-valued gates → filtros simétricos h[t] = h[-t] → mismo peso pasado/futuro.
Walsh y DCT son ideales para **síntesis de pesos**, NO para **mixing secuencial**.

---

## 3. La Arquitectura: Phase-Spectral Transformer (PST)

```
 ┌─────────────────────────────────────────────────────────────┐
 │  Input tokens: [t₀, t₁, ..., t_T]                           │
 └──────────────────────────┬──────────────────────────────────┘
                            │
                    ┌───────▼────────┐
                    │  nn.Embedding  │  (estándar, sin PE)
                    └───────┬────────┘
                            │
            ┌───────────────┼──────────────┐
            │               │              │
     ┌──────▼──────┐ ┌──────▼──────┐  ┌────▼──────┐
     │  PST Block  │ │  PST Block  │  │ PST Block │  × N_layers
     └──────┬──────┘ └──────┬──────┘  └────┬──────┘
            └───────────────┼──────────────┘
                            │
                    ┌───────▼────────┐
                    │  LM Head       │  Linear(d_model → vocab)
                    └───────┬────────┘
                            │
                      [logits]


 ┌─── PST Block (detalle) ─────────────────────────────────────┐
 │                                                             │
 │   x → [LayerNorm] → [FFT Causal Mixer] → + residual → x'    │
 │                                                             │
 │   x' → [LayerNorm] → [Walsh FFN]       → + residual → x''   │
 │                                                             │
 └─────────────────────────────────────────────────────────────┘
```

### Componente A: FFT Causal Mixer (reemplaza self-attention)

```python
# Operación (pseudocódigo)
def fft_causal_mixer(x, log_amp, phase):
    # x: (B, T, D)
    x_padded = pad(x, length=2T)          # zero-pad para convolución lineal
    X = rfft(x_padded.T, dim=time)        # FFT sobre eje temporal
    gate_raw = exp(log_amp) * exp(i*phase)  # gate complejo aprendible
    h = irfft(gate_raw)                    # respuesta impulsional
    h_causal = h * [1,...,1, 0,...,0]      # causal enforcement
    gate_causal = rfft(h_causal)           # gate proyectado a causal
    out = irfft(X * gate_causal)[:T].T    # aplicar y tomar T
    return LayerNorm(out + x)
```

**Parámetros**: `2 × (T//2 + 1)` = `T + 2` reales por capa  
**Complejidad**: O(T log T) vs O(T²) de attention  
**PE**: No necesario — las fases aprenden codificación posicional implícita

### Componente B: Walsh FFN (reemplaza las capas densas)

Exactamente como en V67 / `BLUEPRINT_DCT_LLM.md`:

```python
# WalshLinear: sintetiza W desde un núcleo pequeño
# W = H_out @ Core @ H_in  donde H son matrices Hadamard
# Core ∈ ℝ^(k_out × k_in)  con k << d_model
W_synthesized = hadamard_out @ learnable_core @ hadamard_in
output = input @ W_synthesized.T
```

**Compresión**: factor 16x–32x en parámetros del FFN  
**Hardware**: solo sumas/restas en la síntesis (multiplicación libre en inferencia)

---

## 4. Comparativa vs Arquitecturas Anteriores

| Arquitectura | Mixer temporal | Pesos FFN | PE | Complejidad |
|-------------|---------------|-----------|-----|-------------|
| Transformer estándar | Self-attention (dense) | Dense W | Sí | O(N²) |
| V67 Hybrid Spectral GPT | Self-attention (dense) | DCT/Walsh | Sí | O(N²) |
| PSGT (V260) | Espectral (frozen+gates) | Dense | Sí | O(N) |
| **PST (este blueprint)** | **FFT Causal Mixer** | **Walsh** | **No** | **O(N log N)** |

---

## 5. Ventajas Teóricas

### 5.1 Eliminación del PE: Fases como Codificación Implícita
Los RoPE (Rotary Position Embeddings) de LLaMA/Gemini son el estado del arte actual para codificación posicional. El PST propone algo más fundamental: la fase del gate FFT `exp(i·φ_k)` actúa como un RoPE generalizado aprendido, sin necesidad de inyectarlo manualmente.

### 5.2 Contexto Ilimitado O(N log N)
La atención estándar es O(N²) en memoria y cómputo. El FFT Mixer es O(N log N). Para secuencias de 100K tokens:
- Attention: 100K² = 10¹⁰ operaciones
- FFT Mixer: 100K × log₂(100K) ≈ 1.7 × 10⁶ operaciones  
→ Factor 5,900x más eficiente

### 5.3 Interpretabilidad: Qué aprende cada frecuencia
Los gates `(amp_k, phase_k)` tienen interpretación directa:
- `amp_k`: ¿cuánto importa la frecuencia de repetición `k` en el texto?
- `phase_k`: ¿en qué offset temporal resonamos con esa frecuencia?

Esto permite "visualizar" lo que el modelo ha aprendido directamente en el espacio frecuencial.

---

## 6. Limitaciones Conocidas y Mitigaciones

| Limitación | Descripción | Mitigación |
|------------|-------------|------------|
| **No-localidad** | FFT ve toda la ventana de contexto globalmente | Irrelevante para LM: queremos contexto global |
| **Walsh no es causal vía máscara** | `FWHT → mask → FWHT` no impone causalidad | Usar FFT (no Walsh) como mixer; Walsh solo en FFN |
| **Causal enforcement cost** | IFFT → mask → FFT extra en cada forward | O(T log T) añadido, menor que la atención O(T²) |
| **Menos expresivo que attention** | PPL 5.60 vs 5.17 de attention (con -45% params) | Tradeoff explícito: eficiencia vs calidad marginal |
| **Secuencias muy largas** | Con T grande, pad a 2T puede ser costoso | Usar STFT con ventanas solapadas para T > 4096 |

---

## 7. Hoja de Ruta para `tiny-thinker`

### Fase 1: Validación en Benchmark Limpio ← SIGUIENTE PASO
**Objetivo**: Confirmar PPL competitiva en el dataset de `tiny-thinker` (`train_v1.bin`)  
**Script**: Adaptar `prototype_v281_true_causal_phase_lm.py` al pipeline de `tiny-thinker`

```yaml
# config/train_pst_v1.yaml
architecture: PST
d_model: 256
n_layers: 6
seq_len: 512
mixer: fft_causal          # nuevo
ffn: walsh_linear          # de V67
use_pe: false              # novedad: sin PE
k_ffn: 32                  # núcleo Walsh FFN
```

### Fase 2: PST-Mini como Reemplazo de TinyThinker
**Objetivo**: Entrenar PST con ~10M parámetros equivalentes y comparar contra baseline  
**Métrica clave**: Loss en `val_v1.bin` + tokens/sec en inferencia CPU  

### Fase 3: Contexto Largo con FFT (N=8192+)
**Objetivo**: Demostrar la ventaja O(N log N) vs O(N²) en secuencias largas  
**Hipótesis**: PST puede procesar documentos completos donde TinyThinker falla por OOM  

### Fase 4: KV Cache Espectral (del DCT Blueprint)
Combinar el FFT Mixer con la idea del DCT-KV Cache del blueprint anterior:  
Comprimir el contexto pasado en su representación espectral (mantener solo componentes dominantes).

---

## 8. Instrucciones para AI en Nueva Sesión en `tiny-thinker`

**Si eres un AI leyendo esto, implementa el Phase-Spectral Transformer siguiendo estos pasos:**

### Paso 1: Añadir el FFT Causal Mixer
Crea `model/spectral_mixer.py` con la clase `FFTCausalMixer`:
```python
class FFTCausalMixer(nn.Module):
    """Replaces self-attention. O(T log T), no PE needed."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        T = args.max_seq_len
        self.T = T
        # Next power of 2 >= 2T for zero-padded linear convolution
        self.pad_T = 1
        while self.pad_T < 2 * T: self.pad_T *= 2
        self.n_freq = self.pad_T // 2 + 1
        # Learnable complex gate: amplitude + phase per frequency
        self.log_amp = nn.Parameter(torch.zeros(self.n_freq))
        self.phase   = nn.Parameter(torch.zeros(self.n_freq))
        # Causal mask (precomputed buffer, no grad)
        mask = torch.zeros(self.pad_T); mask[:T] = 1.0
        self.register_buffer('causal_mask', mask)
        self.norm = nn.RMSNorm(args.dim)  # or LayerNorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        residual = x
        xt = x.permute(0, 2, 1)  # (B, D, T)
        # Causal zero-pad
        xt_pad = F.pad(xt, (0, self.pad_T - T))  # (B, D, pad_T)
        X = torch.fft.rfft(xt_pad, dim=-1)        # (B, D, n_freq)
        # Build causal gate (projected to causal subspace)
        gate_raw = torch.exp(self.log_amp) * torch.exp(1j * self.phase)
        h_raw    = torch.fft.irfft(gate_raw, n=self.pad_T)
        h_causal = h_raw * self.causal_mask
        gate_c   = torch.fft.rfft(h_causal, n=self.pad_T)
        # Apply and return
        out = torch.fft.irfft(X * gate_c, n=self.pad_T, dim=-1)[..., :T]
        return self.norm(out.permute(0, 2, 1) + residual)
```

### Paso 2: Añadir flags en `ModelArgs`
```python
@dataclass
class ModelArgs:
    # ... existing args ...
    use_fft_mixer: bool = False      # reemplaza attention con FFT Causal Mixer
    use_walsh_ffn: bool = False      # comprime FFN con Walsh (de blueprint anterior)
    use_pe: bool = True              # False cuando use_fft_mixer=True
    k_walsh_ffn: int = 32            # dimensión del núcleo Walsh
```

### Paso 3: Modificar el bloque Transformer
```python
class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.use_fft_mixer:
            self.mixer = FFTCausalMixer(args)   # O(T log T), no PE
        else:
            self.mixer = Attention(args)         # O(T²), con PE
        if args.use_walsh_ffn:
            self.ffn = WalshFFN(args)            # de BLUEPRINT_DCT_LLM
        else:
            self.ffn = FeedForward(args)         # estándar
```

### Paso 4: Config recomendada para el primer test
```yaml
# Phase-Spectral Transformer — PST Mini (para validar)
dim: 256
n_layers: 6
max_seq_len: 512
use_fft_mixer: true
use_walsh_ffn: true
use_pe: false
k_walsh_ffn: 32
# Parámetros esperados: ~8-12M (vs ~30M de un Transformer estándar equivalente)
```

### Paso 5: Validación
Entrenar con `train.py` usando el config de PST y comparar val loss vs baseline `tiny-thinker`.
**Señal de éxito**: PST alcanza val_loss < 1.9 con <60% de los parámetros del baseline.

---

## 9. Conexión con `brainstorming_3.md` — Base Teórica

Este blueprint implementa concretamente las ideas del brainstorming:

| Concepto del brainstorming | Implementación en PST |
|---------------------------|----------------------|
| "Los pesos son componentes de conexión `ω_μ`" | Gates complejos `(amp_k, phase_k)` por frecuencia |
| "La atención como métrica dinámica `g_μν`" | Gate FFT: métrica aprendida en dominio frecuencial |
| "Aprender es curvar el espacio" | Aprender `phase_k` redistribuye qué posiciones temporales influyen |
| "PE es una muleta" | Confirmado: `ComplexFFT_noPE ≈ ComplexFFT_PE` (delta=0.0002) |
| "Flujo O(N log N)" | FFT Mixer: exactamente O(N log N) |

---

*"La atención es O(N²) porque intenta comparar cada par de tokens. La FFT dice: no necesitas comparar tokens directamente, necesitas encontrar qué frecuencias del texto resuenan entre sí. Y eso cuesta O(N log N)."*
