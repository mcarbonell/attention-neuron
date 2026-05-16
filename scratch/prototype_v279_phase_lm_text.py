"""
prototype_v279_phase_lm_text.py
================================
V279: Complex Phase Mixer en texto real.

Hipótesis (de V278): ComplexFFT codifica posición via fases analíticas.
Pregunta de V279: ¿Puede un LM espectral basado en ComplexFFT prescindir
del Positional Encoding explícito en texto real, mientras que RealWalsh no?

Task: Character-level language modeling (next-char prediction).
      Dataset: "The Tiny Prince" generado sintéticamente o texto local.
      Usa el clásico "copy task" si no hay datos externos: predecir el carácter
      en posición t+k dada la ventana [t-T, t]. El orden importa crucialmente.

Modelos comparados (todos sin PE salvo variantes _PE):
  A_Walsh_noPE   : RealWalsh mixer, sin positional encoding
  B_Walsh_PE     : RealWalsh mixer, con PE (referencia de V260)
  C_ComplexFFT_noPE: ComplexFFT mixer, sin PE  <- CANDIDATO
  D_ComplexFFT_PE  : ComplexFFT mixer, con PE  <- cota superior

Si C_ComplexFFT_noPE ≈ B_Walsh_PE, la hipótesis se confirma:
las fases complejas sustituyen al PE.

Dataset: descarga tiny-shakespeare si existe en scratch/data/, si no lo genera.
"""

import torch, torch.nn as nn, torch.nn.functional as F
import math, time, os, urllib.request

# ── Config ────────────────────────────────────────────────────────────
CFG = dict(
    # Data
    seq_len=128, batch_size=64,
    # Architecture
    d_model=64, n_layers=3, n_heads=1,   # n_heads for future use
    # Training
    epochs=20, lr=3e-3, seed=42,
    # Paths
    data_url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
    data_path='scratch/data/tiny_shakespeare.txt',
)
torch.manual_seed(CFG['seed'])
T = CFG['seq_len']

# ── Dataset ───────────────────────────────────────────────────────────
def load_data(cfg):
    path = cfg['data_path']
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading tiny-shakespeare...")
        urllib.request.urlretrieve(cfg['data_url'], path)
    text = open(path, encoding='utf-8').read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    c2i = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([c2i[c] for c in text], dtype=torch.long)
    print(f"  Dataset: {len(text):,} chars | vocab={vocab_size}")
    return data, vocab_size

def get_batch(data, seq_len, batch_size, device='cpu'):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)

# ── FWHT (fixed butterfly) ────────────────────────────────────────────
def fwht(x: torch.Tensor) -> torch.Tensor:
    N = x.shape[-1]
    h = 1
    while h < N:
        x = x.reshape(*x.shape[:-1], N // (2 * h), 2 * h)
        a, b = x[..., :h], x[..., h:]
        x = torch.cat([a + b, a - b], dim=-1)
        x = x.reshape(*x.shape[:-2], N)
        h *= 2
    return x

# ── Positional Encoding ───────────────────────────────────────────────
class SinCosPE(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):   # x: (B, T, D)
        return x + self.pe[:, :x.shape[1], :]

# ── Spectral Mixers (causal-ish via residual; non-causal mixing) ──────
class RealWalshMixer(nn.Module):
    """Real FWHT gates. Position-aware via ±1 sign patterns."""
    def __init__(self, T, D):
        super().__init__()
        assert (T & (T-1)) == 0, "T must be power of 2"
        self.T = T
        self.log_amp = nn.Parameter(torch.zeros(T))
        self.ff = nn.Sequential(nn.Linear(D, D*2), nn.GELU(), nn.Linear(D*2, D))
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)

    def forward(self, x):
        # Spectral mix
        r = x
        xt = x.permute(0, 2, 1)
        X = fwht(xt)
        X = X * torch.exp(self.log_amp)
        out = (fwht(X) / self.T).permute(0, 2, 1)
        x = self.norm1(out + r)
        # FFN
        return self.norm2(self.ff(x) + x)


class ComplexFFTMixer(nn.Module):
    """
    Complex FFT gates: A_k * exp(i*phi_k). Phase encodes position analytically.
    Non-causal (sees full context) — same as Walsh. Both are used as
    bidirectional sequence mixers feeding a causal LM head.
    """
    def __init__(self, T, D):
        super().__init__()
        self.T = T
        self.n_freq = T // 2 + 1
        self.log_amp = nn.Parameter(torch.zeros(self.n_freq))
        self.phase   = nn.Parameter(torch.zeros(self.n_freq))
        self.ff = nn.Sequential(nn.Linear(D, D*2), nn.GELU(), nn.Linear(D*2, D))
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)

    def forward(self, x):
        r = x
        xt = x.permute(0, 2, 1)
        X = torch.fft.rfft(xt, dim=-1)
        gate = torch.exp(self.log_amp) * torch.exp(1j * self.phase)
        out = torch.fft.irfft(X * gate, n=self.T, dim=-1).permute(0, 2, 1)
        x = self.norm1(out + r)
        return self.norm2(self.ff(x) + x)


# ── Full LM ───────────────────────────────────────────────────────────
class SpectralLM(nn.Module):
    def __init__(self, mixer_cls, vocab_size, T, D, n_layers, use_pe=False):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, D)
        self.pe      = SinCosPE(D) if use_pe else None
        self.mixers  = nn.ModuleList([mixer_cls(T, D) for _ in range(n_layers)])
        self.head    = nn.Linear(D, vocab_size, bias=False)
        self.T       = T

    def forward(self, x):       # x: (B, T) token ids
        h = self.embed(x)       # (B, T, D)
        if self.pe is not None:
            h = self.pe(h)
        for m in self.mixers:
            h = m(h)
        return self.head(h)     # (B, T, vocab_size) — predict next char at each pos

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Training ──────────────────────────────────────────────────────────
def run_model(label, model, data, vocab_size, epochs, lr, batch_size, seq_len):
    print(f"\n{'='*55}\n  {label}\n  params={model.n_params():,}\n{'='*55}")
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_loss = float('inf')
    conv_epoch = None
    t0 = time.time()
    n_batches_per_epoch = 200   # fixed steps per epoch

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for b in range(n_batches_per_epoch):
            xb, yb = get_batch(data, seq_len, batch_size)
            logits = model(xb)          # (B, T, V)
            loss   = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()

            # Fast feedback: first 5 batches of epoch 1
            if ep == 1 and b < 5:
                print(f"  [Ep1 B{b+1}] loss={loss.item():.4f}")

        sched.step()
        avg_loss = ep_loss / n_batches_per_epoch
        if avg_loss < best_loss:
            best_loss = avg_loss
        if conv_epoch is None and avg_loss < 2.5:
            conv_epoch = ep

        print(f"  Ep {ep:02d} | loss={avg_loss:.4f}")

    elapsed = time.time() - t0
    pei = (1 / best_loss) / math.log10(model.n_params() + 1)   # lower loss = higher efficiency
    conv_str = f"Ep{conv_epoch}" if conv_epoch else "Never"
    print(f"  BEST_LOSS={best_loss:.4f} | PEI(inv_loss)={pei:.4f} | Conv={conv_str} | {elapsed:.1f}s")
    return dict(label=label, loss=best_loss, params=model.n_params(), pei=pei,
                conv=conv_epoch, time=elapsed)


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    data, vocab_size = load_data(CFG)

    # Split: 90% train
    n_train = int(len(data) * 0.9)
    train_data = data[:n_train]
    val_data   = data[n_train:]

    D, NL, T = CFG['d_model'], CFG['n_layers'], CFG['seq_len']

    print(f"\nV279: Complex Phase LM on real text")
    print(f"Key question: Does ComplexFFT_noPE ≈ Walsh_PE in perplexity?")
    print(f"Seq={T} | d={D} | layers={NL} | epochs={CFG['epochs']}")

    experiments = [
        # Proposed first (regla de oro)
        ('C_ComplexFFT_noPE [PROPOSED]', ComplexFFTMixer, False),
        ('D_ComplexFFT_PE   [upper]',    ComplexFFTMixer, True),
        ('A_Walsh_noPE      [baseline]', RealWalshMixer,  False),
        ('B_Walsh_PE        [ref V260]', RealWalshMixer,  True),
    ]

    results = []
    for label, mixer_cls, use_pe in experiments:
        torch.manual_seed(CFG['seed'])
        model = SpectralLM(mixer_cls, vocab_size, T, D, NL, use_pe=use_pe)
        r = run_model(label, model, train_data, vocab_size,
                      CFG['epochs'], CFG['lr'], CFG['batch_size'], T)
        results.append(r)

    # Summary
    print(f"\n\n{'='*65}")
    print(f"  V279 SUMMARY — Phase LM on Tiny Shakespeare")
    print(f"{'='*65}")
    print(f"  {'Model':<35} {'Loss':>6} {'Params':>8} {'Conv':>6} {'Time':>8}")
    print(f"  {'-'*63}")
    for r in sorted(results, key=lambda x: x['loss']):
        c = f"Ep{r['conv']}" if r['conv'] else "Never"
        print(f"  {r['label']:<35} {r['loss']:>6.4f} {r['params']:>8,} {c:>6} {r['time']:>7.1f}s")
    print(f"{'='*65}")

    c_no_pe = next(r for r in results if 'ComplexFFT_noPE' in r['label'])
    w_pe    = next(r for r in results if 'Walsh_PE'        in r['label'])
    w_no_pe = next(r for r in results if 'Walsh_noPE'      in r['label'])

    print(f"\n  Key comparisons:")
    d1 = c_no_pe['loss'] - w_pe['loss']
    d2 = c_no_pe['loss'] - w_no_pe['loss']
    print(f"  ComplexFFT_noPE vs Walsh_PE    : {d1:+.4f}  (< 0.1 = hypothesis confirmed)")
    print(f"  ComplexFFT_noPE vs Walsh_noPE  : {d2:+.4f}  (< 0 = phase helps vs Walsh)")
    if d1 < 0.1:
        print("  -> CONFIRMED: ComplexFFT phases replace explicit PE.")
    elif d1 < 0.3:
        print("  -> PARTIAL: ComplexFFT_noPE is competitive but PE still helps.")
    else:
        print("  -> INCONCLUSIVE: PE remains necessary for both mixers.")
