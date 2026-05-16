"""
prototype_v278_phase_spectral_lm.py  (v2 — bugs fixed)
=======================================================
Hypothesis: Complex FFT phases encode temporal position. Real-valued gates
(Walsh, FFT magnitude-only) are provably blind to position.

Mathematical proof in the task:
  DFT of a spike at position t: X[k] = exp(-i*2pi*k*t/N)
  -> |X[k]| = 1/N for ALL k  (magnitude is FLAT, position-blind)
  -> phase(X[k]) = -2pi*k*t/N  (phase ENCODES position uniquely)

  Walsh of a spike at position t: W[k] = ±1 for all k
  -> Same magnitude structure, real-valued, position-blind

Task: "Single Spike Half Detection"
  - Sequence of T=64 zeros, with exactly ONE position set to 1.
  - Class 0: spike in positions [0, T//2)
  - Class 1: spike in positions [T//2, T)
  - Provably: RealWalsh = ~50%, RealFFT (no phase) = ~50%, ComplexFFT = >90%

Key architecture fix (v1 bug): Use CLS pooling (position 0) not mean pooling.
  mean(IFFT(gate * FFT(x))) = gate[0] * mean(x) = same for both classes!
  But h[:, 0, :] after ComplexFFT mixing carries phase-encoded position info.
"""

import torch, torch.nn as nn, torch.nn.functional as F
import math, time

# ── Config ────────────────────────────────────────────────────────────
CFG = dict(T=64, d_model=64, n_layers=3, n_train=8000, n_val=2000,
           batch_size=128, epochs=25, lr=3e-3, seed=42)
torch.manual_seed(CFG['seed'])

# ── Dataset ───────────────────────────────────────────────────────────
def make_dataset(n, T, seed_offset=0):
    """Single spike at random position. Label = spike in second half."""
    rng = torch.Generator(); rng.manual_seed(seed_offset)
    X = torch.zeros(n, T)
    positions = torch.randint(0, T, (n,), generator=rng)
    X[torch.arange(n), positions] = 1.0
    y = (positions >= T // 2).long()
    return X, y

# ── FWHT (fixed butterfly) ────────────────────────────────────────────
def fwht(x: torch.Tensor) -> torch.Tensor:
    """Walsh-Hadamard Transform along last dim. N must be power of 2."""
    N = x.shape[-1]
    h = 1
    while h < N:
        # group into chunks of 2h: (..., N//(2h), 2h)
        x = x.reshape(*x.shape[:-1], N // (2 * h), 2 * h)
        a, b = x[..., :h], x[..., h:]           # split each chunk in half
        x = torch.cat([a + b, a - b], dim=-1)   # butterfly
        x = x.reshape(*x.shape[:-2], N)         # flatten back
        h *= 2
    return x

# ── Spectral Mixers ───────────────────────────────────────────────────
class RealWalshMixer(nn.Module):
    """Baseline: real FWHT gates. Position-blind (Walsh magnitudes are flat)."""
    def __init__(self, T, D):
        super().__init__()
        self.T = T
        self.log_amp = nn.Parameter(torch.zeros(T))
        self.norm = nn.LayerNorm(D)

    def forward(self, x):          # x: (B, T, D)
        r = x
        xt = x.permute(0, 2, 1)                    # (B, D, T)
        X  = fwht(xt)
        X  = X * torch.exp(self.log_amp)            # real gate
        out = fwht(X) / self.T                      # inverse WHT
        return self.norm(out.permute(0, 2, 1) + r)


class RealFFTMixer(nn.Module):
    """Ablation: FFT amplitude only, phase forcibly zeroed. Still position-blind."""
    def __init__(self, T, D):
        super().__init__()
        self.T = T
        self.n_freq = T // 2 + 1
        self.log_amp = nn.Parameter(torch.zeros(self.n_freq))
        self.norm = nn.LayerNorm(D)

    def forward(self, x):
        r = x
        xt = x.permute(0, 2, 1)                    # (B, D, T)
        X  = torch.fft.rfft(xt, dim=-1)            # complex
        amp = torch.exp(self.log_amp)
        # Kill phase: reconstruct from magnitude only (phase = 0)
        X_noPhase = X.abs() * amp + 0j
        out = torch.fft.irfft(X_noPhase, n=self.T, dim=-1)
        return self.norm(out.permute(0, 2, 1) + r)


class ComplexFFTMixer(nn.Module):
    """
    PROPOSED: FFT with learned amplitude AND phase.
    Phase phi_k creates a matched filter: gate = A_k * exp(i*phi_k)
    filtered[t] = IFFT(gate * FFT(x))[t]
               = sum_k A_k * exp(i*(phi_k - 2pi*k*t/N)) * X[k]
    -> Constructive interference when phi_k = 2pi*k*t_target/N
    -> Model learns to 'tune in' to specific positions via phase alignment.
    """
    def __init__(self, T, D):
        super().__init__()
        self.T = T
        self.n_freq = T // 2 + 1
        self.log_amp = nn.Parameter(torch.zeros(self.n_freq))
        self.phase   = nn.Parameter(torch.zeros(self.n_freq))
        self.norm = nn.LayerNorm(D)

    def forward(self, x):
        r = x
        xt = x.permute(0, 2, 1)                    # (B, D, T)
        X  = torch.fft.rfft(xt, dim=-1)            # complex (B, D, n_freq)
        gate = torch.exp(self.log_amp) * torch.exp(1j * self.phase)
        out  = torch.fft.irfft(X * gate, n=self.T, dim=-1)
        return self.norm(out.permute(0, 2, 1) + r)


class DenseMixer(nn.Module):
    """Reference: dense linear mix over sequence. Trivially position-aware."""
    def __init__(self, T, D):
        super().__init__()
        self.W    = nn.Linear(T, T)
        self.norm = nn.LayerNorm(D)

    def forward(self, x):
        r  = x
        out = self.W(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.norm(out + r)


# ── Full model ────────────────────────────────────────────────────────
class SpectralClassifier(nn.Module):
    """
    Embed → N×Mixer → CLS pool (position 0) → Linear head.

    CLS pooling (h[:,0,:]) instead of mean pooling is CRITICAL:
      mean(IFFT(gate*FFT(x))) = gate[0]*mean(x)  -- same for all spike positions!
      But h[:,0,:] after mixing carries phase-encoded positional information.
    """
    def __init__(self, mixer_cls, T, D, n_layers):
        super().__init__()
        self.embed  = nn.Embedding(2, D)
        self.mixers = nn.ModuleList([mixer_cls(T, D) for _ in range(n_layers)])
        self.head   = nn.Linear(D, 2)

    def forward(self, x_float):
        h = self.embed(x_float.long())      # (B, T, D)
        for m in self.mixers:
            h = m(h)
        cls = h[:, 0, :]                    # CLS token: position 0 after mixing
        return self.head(cls)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Training ──────────────────────────────────────────────────────────
def run(name, mixer_cls, X_tr, y_tr, X_va, y_va):
    T, D, NL = CFG['T'], CFG['d_model'], CFG['n_layers']
    model = SpectralClassifier(mixer_cls, T, D, NL)
    torch.manual_seed(CFG['seed'])

    opt   = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG['epochs'])
    n     = X_tr.shape[0]; bs = CFG['batch_size']

    print(f"\n{'='*52}\n  {name}  |  params={model.n_params():,}\n{'='*52}")
    best, conv_epoch, t0 = 0.0, None, time.time()

    for ep in range(1, CFG['epochs'] + 1):
        model.train()
        perm = torch.randperm(n); X_tr, y_tr = X_tr[perm], y_tr[perm]
        ep_loss = 0.0
        for i in range(0, n, bs):
            xb, yb = X_tr[i:i+bs], y_tr[i:i+bs]
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
            if ep == 1 and i // bs < 5:
                print(f"  [Ep1 B{i//bs+1}] loss={loss.item():.4f}")
        sched.step()

        model.eval()
        with torch.no_grad():
            acc = (model(X_va).argmax(1) == y_va).float().mean().item()
        if acc > best: best = acc
        if conv_epoch is None and acc >= 0.75: conv_epoch = ep
        print(f"  Ep {ep:02d} | loss={ep_loss/(n//bs):.4f} | val={acc:.3f}")

    pei = best / math.log10(model.n_params() + 1)
    print(f"  BEST={best:.4f} | PEI={pei:.4f} | Conv={'Ep'+str(conv_epoch) if conv_epoch else 'Never'} | {time.time()-t0:.1f}s")
    return dict(name=name, acc=best, params=model.n_params(), pei=pei, conv=conv_epoch)


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    T = CFG['T']
    X_tr, y_tr = make_dataset(CFG['n_train'], T, 0)
    X_va, y_va = make_dataset(CFG['n_val'],   T, 9999)
    print(f"Task: Single spike, predict half. Random baseline = 50%")
    print(f"Class balance: {(y_tr==0).sum()}/{(y_tr==1).sum()} train")

    # Proposed first (regla de oro)
    experiments = [
        ('C_ComplexFFT [PROPOSED]', ComplexFFTMixer),
        ('A_RealWalsh  [baseline]', RealWalshMixer),
        ('B_RealFFT    [ablation]', RealFFTMixer),
        ('D_Dense      [reference]', DenseMixer),
    ]
    results = [run(n, m, X_tr, y_tr, X_va, y_va) for n, m in experiments]

    print(f"\n{'='*60}")
    print(f"  V278 SUMMARY — Complex Phase Spectral Mixer")
    print(f"{'='*60}")
    print(f"  {'Model':<30} {'Acc':>6} {'Params':>8} {'PEI':>7} {'Conv':>6}")
    print(f"  {'-'*58}")
    for r in sorted(results, key=lambda x: -x['acc']):
        c = f"Ep{r['conv']}" if r['conv'] else "Never"
        print(f"  {r['name']:<30} {r['acc']:>6.3f} {r['params']:>8,} {r['pei']:>7.4f} {c:>6}")
    print(f"{'='*60}")

    c  = next(r for r in results if 'Complex' in r['name'])
    w  = next(r for r in results if 'Walsh'   in r['name'])
    d  = c['acc'] - w['acc']
    print(f"\n  ComplexFFT vs RealWalsh: {d:+.3f} ({d*100:+.1f}%)")
    verdict = "CONFIRMED" if d > 0.15 else "PARTIAL" if d > 0.05 else "INCONCLUSIVE"
    print(f"  Verdict: {verdict}")
