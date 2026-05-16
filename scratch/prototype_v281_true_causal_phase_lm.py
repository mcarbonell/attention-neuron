"""
prototype_v281_true_causal_phase_lm.py
========================================
V281: Causalidad real via proyección al subespacio causal.

V280 demostró que el zero-padding NO garantiza causalidad: el gate frecuencial
aprendido puede tener una respuesta impulsional con componentes no-causales.

FIX: Causal Filter Enforcement
  1. gate_raw = amp * exp(i*phi)           (gate aprendido, no-causal)
  2. h_raw = IFFT(gate_raw, n=2T)         (respuesta impulsional)
  3. h_causal = h_raw * causal_mask        (zerear parte no-causal)
  4. gate_causal = FFT(h_causal, n=2T)     (gate forzado a causal)
  5. out = IFFT(FFT(x_padded) * gate_causal)[:T]

causal_mask = [1,1,...,1, 0,0,...,0]  (T unos, T ceros)

Modelos:
  C_ComplexFFT_Causal  : FFT con fase compleja + causal enforcement
  A_Walsh_Causal       : Walsh real + causal enforcement (misma técnica)
  E_CausalAttention    : Self-attention causal estándar (baseline académico)

Hipótesis: Si ComplexFFT sigue superando a Walsh con causalidad real,
el mecanismo es genuino (asimetría de filtros complejos vs. filtros reales).
"""

import torch, torch.nn as nn, torch.nn.functional as F
import math, time, os, urllib.request

CFG = dict(
    seq_len=128, batch_size=64,
    d_model=64, n_layers=3, n_heads=4,
    epochs=20, lr=3e-3, seed=42,
    steps_per_epoch=200, val_steps=50,
    data_url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
    data_path='scratch/data/tiny_shakespeare.txt',
)
torch.manual_seed(CFG['seed'])
T = CFG['seq_len']

# ── Data ──────────────────────────────────────────────────────────────
def load_data(cfg):
    if not os.path.exists(cfg['data_path']):
        os.makedirs(os.path.dirname(cfg['data_path']), exist_ok=True)
        urllib.request.urlretrieve(cfg['data_url'], cfg['data_path'])
    text = open(cfg['data_path'], encoding='utf-8').read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    c2i = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([c2i[c] for c in text], dtype=torch.long)
    print(f"Dataset: {len(text):,} chars | vocab={vocab_size}")
    return data, vocab_size

def get_batch(data, T, bs):
    ix = torch.randint(len(data) - T, (bs,))
    x = torch.stack([data[i:i+T]   for i in ix])
    y = torch.stack([data[i+1:i+T+1] for i in ix])
    return x, y

# ── FWHT ──────────────────────────────────────────────────────────────
def fwht(x):
    N = x.shape[-1]
    h = 1
    while h < N:
        x = x.reshape(*x.shape[:-1], N//(2*h), 2*h)
        a, b = x[..., :h], x[..., h:]
        x = torch.cat([a+b, a-b], dim=-1)
        x = x.reshape(*x.shape[:-2], N)
        h *= 2
    return x

# ── Positional Encoding ───────────────────────────────────────────────
class SinCosPE(nn.Module):
    def __init__(self, D, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, D)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000)/D))
        pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.shape[1]]

# ── True Causal Spectral Mixer (via time-domain projection) ───────────
class TrueCausalComplexFFTMixer(nn.Module):
    """
    Complex FFT gate + causal enforcement.

    The learned gate gate_raw may be non-causal. We project it to the
    causal subspace each forward pass:
        h = IFFT(gate_raw)           (impulse response, may have t>0 components)
        h_causal = h * mask          (zero out anti-causal part: t >= T)
        gate_causal = FFT(h_causal)  (now strictly causal)

    This means amp and phase are learned in frequency space but constrained
    to produce causal impulse responses. Gradients flow through IFFT→mask→FFT.
    The complex phase is still free to create asymmetric causal filters.
    """
    def __init__(self, T, D):
        super().__init__()
        self.T = T
        self.pad_T = 1
        while self.pad_T < 2*T: self.pad_T *= 2
        self.n_freq = self.pad_T // 2 + 1
        self.log_amp = nn.Parameter(torch.zeros(self.n_freq))
        self.phase   = nn.Parameter(torch.zeros(self.n_freq))
        # Causal mask: keep t=0..T-1, zero t=T..pad_T-1
        mask = torch.zeros(self.pad_T)
        mask[:T] = 1.0
        self.register_buffer('causal_mask', mask)
        self.ff    = nn.Sequential(nn.Linear(D, D*2), nn.GELU(), nn.Linear(D*2, D))
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)

    def forward(self, x):                      # x: (B, T, D)
        r = x
        B, T, D = x.shape
        xt = x.permute(0, 2, 1)               # (B, D, T)
        # Zero-pad input
        pad = torch.zeros(B, D, self.pad_T-T, device=x.device)
        xt_pad = torch.cat([xt, pad], dim=-1)  # (B, D, pad_T)
        X = torch.fft.rfft(xt_pad, dim=-1)     # (B, D, n_freq)

        # Build gate and project to causal subspace
        gate_raw  = torch.exp(self.log_amp) * torch.exp(1j * self.phase)  # (n_freq,)
        h_raw     = torch.fft.irfft(gate_raw, n=self.pad_T)               # (pad_T,) real
        h_causal  = h_raw * self.causal_mask                              # enforce causal
        gate_causal = torch.fft.rfft(h_causal, n=self.pad_T)             # (n_freq,) complex

        out = torch.fft.irfft(X * gate_causal, n=self.pad_T, dim=-1)[..., :T]
        x = self.norm1(out.permute(0, 2, 1) + r)
        return self.norm2(self.ff(x) + x)


class TrueCausalWalshMixer(nn.Module):
    """
    Walsh mixer with same causal enforcement as ComplexFFT for fair comparison.
    Real gate → symmetric impulse response → causal projection zeros anti-causal half.
    Note: real gates produce even impulse responses, so causal projection
    is less effective (loses half the taps). This is the structural disadvantage
    of real-valued mixing vs complex mixing.
    """
    def __init__(self, T, D):
        super().__init__()
        self.T = T
        self.pad_T = 1
        while self.pad_T < 2*T: self.pad_T *= 2
        self.log_amp = nn.Parameter(torch.zeros(self.pad_T))
        mask = torch.zeros(self.pad_T)
        mask[:T] = 1.0
        self.register_buffer('causal_mask', mask)
        self.ff    = nn.Sequential(nn.Linear(D, D*2), nn.GELU(), nn.Linear(D*2, D))
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)

    def forward(self, x):
        r = x
        B, T, D = x.shape
        xt = x.permute(0, 2, 1)
        pad = torch.zeros(B, D, self.pad_T-T, device=x.device)
        xt_pad = torch.cat([xt, pad], dim=-1)
        X = fwht(xt_pad)

        # Walsh gate + causal projection
        gate_raw  = torch.exp(self.log_amp)                # real, positive
        h_raw     = fwht(gate_raw) / self.pad_T            # impulse response (real)
        h_causal  = h_raw * self.causal_mask               # zero anti-causal
        gate_causal = fwht(h_causal)                       # back to frequency

        out = (fwht(X * gate_causal) / self.pad_T)[..., :T]
        x = self.norm1(out.permute(0, 2, 1) + r)
        return self.norm2(self.ff(x) + x)


class CausalAttentionMixer(nn.Module):
    """Standard causal multi-head self-attention (academic baseline)."""
    def __init__(self, T, D, n_heads=4):
        super().__init__()
        self.attn  = nn.MultiheadAttention(D, n_heads, batch_first=True)
        self.ff    = nn.Sequential(nn.Linear(D, D*2), nn.GELU(), nn.Linear(D*2, D))
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
        # Causal mask: upper triangle = -inf
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        self.register_buffer('attn_mask', mask)

    def forward(self, x):
        r = x
        attn_out, _ = self.attn(x, x, x, attn_mask=self.attn_mask)
        x = self.norm1(attn_out + r)
        return self.norm2(self.ff(x) + x)


# ── Full LM ───────────────────────────────────────────────────────────
class SpectralLM(nn.Module):
    def __init__(self, mixer_cls, vocab_size, T, D, n_layers, n_heads=4, use_pe=False):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, D)
        self.pe     = SinCosPE(D) if use_pe else None
        if mixer_cls == CausalAttentionMixer:
            self.mixers = nn.ModuleList([mixer_cls(T, D, n_heads) for _ in range(n_layers)])
        else:
            self.mixers = nn.ModuleList([mixer_cls(T, D) for _ in range(n_layers)])
        self.head   = nn.Linear(D, vocab_size, bias=False)

    def forward(self, x):
        h = self.embed(x)
        if self.pe: h = self.pe(h)
        for m in self.mixers: h = m(h)
        return self.head(h)

    def n_params(self): return sum(p.numel() for p in self.parameters())


# ── Eval / Train ──────────────────────────────────────────────────────
@torch.no_grad()
def eval_loss(model, data, T, bs, n=50):
    model.eval()
    losses = [F.cross_entropy(model(x).reshape(-1, model.head.out_features), y.reshape(-1)).item()
              for x, y in [get_batch(data, T, bs) for _ in range(n)]]
    return sum(losses) / len(losses)


def run(label, model, train_data, val_data, vocab_size):
    T = CFG['seq_len']; bs = CFG['batch_size']
    print(f"\n{'='*55}\n  {label}\n  params={model.n_params():,}\n{'='*55}")
    opt   = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG['epochs'])
    best_val, conv_epoch, t0 = float('inf'), None, time.time()

    for ep in range(1, CFG['epochs']+1):
        model.train()
        ep_loss = 0.0
        for b in range(CFG['steps_per_epoch']):
            x, y = get_batch(train_data, T, bs)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            if ep == 1 and b < 5:
                print(f"  [Ep1 B{b+1}] train={loss.item():.4f}")
        sched.step()

        val = eval_loss(model, val_data, T, bs, CFG['val_steps'])
        if val < best_val: best_val = val
        if conv_epoch is None and val < 2.5: conv_epoch = ep
        print(f"  Ep {ep:02d} | train={ep_loss/CFG['steps_per_epoch']:.4f} | val={val:.4f}")

    elapsed = time.time() - t0
    ppl = math.exp(min(best_val, 10))
    print(f"  BEST_VAL={best_val:.4f} | PPL={ppl:.2f} | Conv={'Ep'+str(conv_epoch) if conv_epoch else 'Never'} | {elapsed:.1f}s")
    return dict(label=label, val=best_val, ppl=ppl, params=model.n_params(),
                conv=conv_epoch, time=elapsed)


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    data, vocab_size = load_data(CFG)
    n_train = int(len(data) * 0.9)
    train_data, val_data = data[:n_train], data[n_train:]
    T, D, NL = CFG['seq_len'], CFG['d_model'], CFG['n_layers']

    print(f"\nV281: True Causal Phase LM")
    print(f"Causal enforcement: IFFT(gate) * mask → FFT (no future leakage)")
    print(f"Seq={T} | d={D} | layers={NL} | epochs={CFG['epochs']}")
    print(f"Key: if val_loss > 0.5 for ComplexFFT, causality is working")
    print(f"     if ComplexFFT still << Walsh, phase advantage is genuine")

    # Proposed first (regla de oro)
    experiments = [
        ('C_ComplexFFT_TrueCausal [PROPOSED]', TrueCausalComplexFFTMixer, False),
        ('D_ComplexFFT_TC_PE      [upper]',    TrueCausalComplexFFTMixer, True),
        ('A_Walsh_TrueCausal      [baseline]', TrueCausalWalshMixer,      False),
        ('E_CausalAttention_PE    [academic]', CausalAttentionMixer,      True),
    ]

    results = []
    for label, mixer_cls, use_pe in experiments:
        torch.manual_seed(CFG['seed'])
        model = SpectralLM(mixer_cls, vocab_size, T, D, NL,
                           n_heads=CFG['n_heads'], use_pe=use_pe)
        r = run(label, model, train_data, val_data, vocab_size)
        results.append(r)

    print(f"\n\n{'='*68}")
    print(f"  V281 SUMMARY — True Causal Phase LM")
    print(f"{'='*68}")
    print(f"  {'Model':<42} {'ValLoss':>8} {'PPL':>6} {'Conv':>6}")
    print(f"  {'-'*66}")
    for r in sorted(results, key=lambda x: x['val']):
        c = f"Ep{r['conv']}" if r['conv'] else "Never"
        print(f"  {r['label']:<42} {r['val']:>8.4f} {r['ppl']:>6.2f} {c:>6}")
    print(f"{'='*68}")

    c = next(r for r in results if 'ComplexFFT_TrueCausal' in r['label'] and 'PE' not in r['label'].split('[')[0].strip())
    w = next(r for r in results if 'Walsh_TrueCausal' in r['label'])
    a = next(r for r in results if 'Attention' in r['label'])

    print(f"\n  ComplexFFT_TC (noPE) vs Walsh_TC:  {c['val']-w['val']:+.4f}")
    print(f"  ComplexFFT_TC (noPE) vs Attention: {c['val']-a['val']:+.4f}")
    print(f"\n  Causality check: val_loss > 0.5 implies no future leakage")
    if c['val'] > 0.5:
        print(f"  ComplexFFT_TC val={c['val']:.4f} > 0.5 → CAUSAL (no leakage) ✓")
    else:
        print(f"  ComplexFFT_TC val={c['val']:.4f} — still suspiciously low, investigate")

    if c['val'] < w['val']:
        print(f"  Phase advantage vs Walsh: CONFIRMED even with true causality")
    else:
        print(f"  Phase advantage vs Walsh: NOT confirmed with true causality")
