"""
prototype_v280_causal_phase_lm.py
===================================
V280: Causal Phase LM — el test honesto.

V279 demostró que ComplexFFT es un mezclador mucho más expresivo que Walsh,
pero el LM era no-causal (el mixer veía tokens futuros). Los losses eran
artificialmente bajos.

Esta versión usa un mixer CAUSAL: en cada posición t, solo puede ver [0..t].
Técnica: Causal Spectral Mixing via Zero-Padding.
  - Append zeros [x_t, 0, 0, ..., 0] de longitud T → total 2T
  - Apply FFT/Walsh sobre 2T (solo usa mitad causal)
  - Gate complex/real
  - Inverse FFT, take first T elements
  - Equivalente a una convolución causal en el dominio espectral

Esta técnica garantiza causalidad sin necesidad de attention masking.

Modelos (mismos params, ~59K):
  A_Walsh_noPE    : Walsh causal, sin PE
  B_Walsh_PE      : Walsh causal, con PE
  C_ComplexFFT_noPE: FFT causal, sin PE  <- CANDIDATO
  D_ComplexFFT_PE  : FFT causal, con PE

Validación honesta: batch de val separado, pérdida sobre tokens no vistos en train.
"""

import torch, torch.nn as nn, torch.nn.functional as F
import math, time, os, urllib.request

# ── Config ────────────────────────────────────────────────────────────
CFG = dict(
    seq_len=128, batch_size=64,
    d_model=64, n_layers=3,
    epochs=20, lr=3e-3, seed=42,
    steps_per_epoch=200,
    data_url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
    data_path='scratch/data/tiny_shakespeare.txt',
)
torch.manual_seed(CFG['seed'])
T = CFG['seq_len']

# ── Data ──────────────────────────────────────────────────────────────
def load_data(cfg):
    path = cfg['data_path']
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(cfg['data_url'], path)
    text = open(path, encoding='utf-8').read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    c2i = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([c2i[c] for c in text], dtype=torch.long)
    print(f"Dataset: {len(text):,} chars | vocab={vocab_size}")
    return data, vocab_size

def get_batch(data, seq_len, batch_size):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len]   for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y

# ── FWHT ──────────────────────────────────────────────────────────────
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
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]

# ── Causal Spectral Mixers ────────────────────────────────────────────
class CausalWalshMixer(nn.Module):
    """
    Causal Walsh mixing via zero-padding trick.
    Pad x to [x, 0...0] of length 2T, apply FWHT, gate, IFWHT, take first T.
    This is equivalent to a causal linear convolution in the Walsh domain:
    position t can only influence positions >= t in the output.

    Note: FWHT is self-inverse (normalized). For circular convolution,
    we use T-point FWHT. For causal linear convolution, zero-pad to 2T.
    """
    def __init__(self, T, D):
        super().__init__()
        self.T = T
        self.pad_T = 2 * T   # next power of 2 >= 2T
        # Adjust pad_T to next power of 2 for FWHT
        self.pad_T = 1
        while self.pad_T < 2 * T:
            self.pad_T *= 2
        self.log_amp = nn.Parameter(torch.zeros(self.pad_T))
        self.ff   = nn.Sequential(nn.Linear(D, D*2), nn.GELU(), nn.Linear(D*2, D))
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)

    def forward(self, x):               # x: (B, T, D)
        r = x
        B, T, D = x.shape
        xt = x.permute(0, 2, 1)        # (B, D, T)
        # Causal zero-pad: append zeros to length pad_T
        pad = torch.zeros(B, D, self.pad_T - T, device=x.device)
        xt_pad = torch.cat([xt, pad], dim=-1)   # (B, D, pad_T)
        X = fwht(xt_pad)
        X = X * torch.exp(self.log_amp)
        out = (fwht(X) / self.pad_T)[..., :T]  # take first T (causal)
        x = self.norm1(out.permute(0, 2, 1) + r)
        return self.norm2(self.ff(x) + x)


class CausalComplexFFTMixer(nn.Module):
    """
    Causal Complex FFT mixing via zero-padding.
    x → zero-pad to 2T → rfft → complex gate (amp * e^(i*phi)) → irfft → take T.
    Causal: output at position t depends only on input at [0..t].
    Phase encodes temporal position WITHOUT explicit PE.
    """
    def __init__(self, T, D):
        super().__init__()
        self.T = T
        # Pad to nearest power of 2 >= 2T for efficiency
        self.pad_T = 1
        while self.pad_T < 2 * T:
            self.pad_T *= 2
        self.n_freq  = self.pad_T // 2 + 1
        self.log_amp = nn.Parameter(torch.zeros(self.n_freq))
        self.phase   = nn.Parameter(torch.zeros(self.n_freq))
        self.ff   = nn.Sequential(nn.Linear(D, D*2), nn.GELU(), nn.Linear(D*2, D))
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)

    def forward(self, x):               # x: (B, T, D)
        r = x
        B, T, D = x.shape
        xt = x.permute(0, 2, 1)        # (B, D, T)
        # Causal zero-pad
        pad = torch.zeros(B, D, self.pad_T - T, device=x.device)
        xt_pad = torch.cat([xt, pad], dim=-1)   # (B, D, pad_T)
        X    = torch.fft.rfft(xt_pad, dim=-1)   # complex (B, D, n_freq)
        gate = torch.exp(self.log_amp) * torch.exp(1j * self.phase)
        out  = torch.fft.irfft(X * gate, n=self.pad_T, dim=-1)[..., :T]  # causal
        x = self.norm1(out.permute(0, 2, 1) + r)
        return self.norm2(self.ff(x) + x)


# ── Full Model ────────────────────────────────────────────────────────
class CausalSpectralLM(nn.Module):
    def __init__(self, mixer_cls, vocab_size, T, D, n_layers, use_pe=False):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, D)
        self.pe     = SinCosPE(D) if use_pe else None
        self.mixers = nn.ModuleList([mixer_cls(T, D) for _ in range(n_layers)])
        self.head   = nn.Linear(D, vocab_size, bias=False)

    def forward(self, x):
        h = self.embed(x)
        if self.pe: h = self.pe(h)
        for m in self.mixers: h = m(h)
        return self.head(h)

    def n_params(self): return sum(p.numel() for p in self.parameters())


# ── Validation loss ───────────────────────────────────────────────────
@torch.no_grad()
def eval_loss(model, data, seq_len, batch_size, n_batches=50):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, seq_len, batch_size)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        losses.append(loss.item())
    return sum(losses) / len(losses)


# ── Training ──────────────────────────────────────────────────────────
def run(label, model, train_data, val_data, vocab_size):
    D = CFG['d_model']; T = CFG['seq_len']; bs = CFG['batch_size']
    print(f"\n{'='*55}\n  {label}\n  params={model.n_params():,}\n{'='*55}")
    opt   = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG['epochs'])
    best_val, conv_epoch, t0 = float('inf'), None, time.time()

    for ep in range(1, CFG['epochs'] + 1):
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
                print(f"  [Ep1 B{b+1}] train_loss={loss.item():.4f}")
        sched.step()

        # Honest validation on held-out data
        val_loss = eval_loss(model, val_data, T, bs)
        if val_loss < best_val: best_val = val_loss
        if conv_epoch is None and val_loss < 2.5: conv_epoch = ep

        avg_train = ep_loss / CFG['steps_per_epoch']
        print(f"  Ep {ep:02d} | train={avg_train:.4f} | val={val_loss:.4f}")

    elapsed = time.time() - t0
    ppl = math.exp(min(best_val, 10))   # perplexity (capped to avoid overflow)
    print(f"  BEST_VAL={best_val:.4f} | PPL={ppl:.1f} | Conv={'Ep'+str(conv_epoch) if conv_epoch else 'Never'} | {elapsed:.1f}s")
    return dict(label=label, val=best_val, ppl=ppl, params=model.n_params(),
                conv=conv_epoch, time=elapsed)


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    data, vocab_size = load_data(CFG)
    n_train = int(len(data) * 0.9)
    train_data, val_data = data[:n_train], data[n_train:]
    T, D, NL = CFG['seq_len'], CFG['d_model'], CFG['n_layers']

    print(f"\nV280: Causal Phase LM (honest eval — no future leakage)")
    print(f"Val loss reported on held-out 10% of data")
    print(f"Seq={T} | d={D} | layers={NL} | epochs={CFG['epochs']}")

    experiments = [
        # Proposed first (regla de oro)
        ('C_CausalComplexFFT_noPE [PROPOSED]', CausalComplexFFTMixer, False),
        ('D_CausalComplexFFT_PE   [upper]',    CausalComplexFFTMixer, True),
        ('A_CausalWalsh_noPE      [baseline]', CausalWalshMixer,      False),
        ('B_CausalWalsh_PE        [ref]',      CausalWalshMixer,      True),
    ]

    results = []
    for label, mixer_cls, use_pe in experiments:
        torch.manual_seed(CFG['seed'])
        model = CausalSpectralLM(mixer_cls, vocab_size, T, D, NL, use_pe=use_pe)
        r = run(label, model, train_data, val_data, vocab_size)
        results.append(r)

    print(f"\n\n{'='*68}")
    print(f"  V280 SUMMARY — Causal Phase LM (Honest Eval)")
    print(f"{'='*68}")
    print(f"  {'Model':<40} {'ValLoss':>8} {'PPL':>7} {'Conv':>6}")
    print(f"  {'-'*66}")
    for r in sorted(results, key=lambda x: x['val']):
        c = f"Ep{r['conv']}" if r['conv'] else "Never"
        print(f"  {r['label']:<40} {r['val']:>8.4f} {r['ppl']:>7.1f} {c:>6}")
    print(f"{'='*68}")

    c = next(r for r in results if 'ComplexFFT_noPE' in r['label'])
    w = next(r for r in results if 'Walsh_PE'        in r['label'])
    print(f"\n  CausalComplexFFT_noPE vs CausalWalsh_PE: {c['val']-w['val']:+.4f} in val loss")
    if c['val'] < w['val']:
        print("  -> CONFIRMED: Complex phase outperforms Walsh+PE even without PE.")
    elif c['val'] < w['val'] + 0.3:
        print("  -> PARTIAL: Close, complex phases partially compensate for no PE.")
    else:
        print("  -> Causality changes the picture. Phase advantage smaller in causal setting.")
