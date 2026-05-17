"""
prototype_v284_spherical_loss.py
================================
V284: Spherical Loss & Phase Continuity Regularization

Prueba dos innovaciones teóricas sobre el motor Matrix-Free V283:
1. Spherical Loss: Substituir la salida lineal por una Similitud Coseno con Temperatura aprendible.
2. Phase Continuity: Regularizador para forzar transiciones de fase suaves en el espectro de la FFT.
"""

import torch, torch.nn as nn, torch.nn.functional as F
import math, time, os, urllib.request

CFG = dict(
    seq_len=128, batch_size=64,
    d_model=128, n_layers=3, k_walsh=32, # Mismo baseline que Matrix-Free k32
    epochs=40, lr=3e-2,
    seed=42, steps_per_epoch=200, val_steps=50,
    lambda_phase=0.01,
    data_url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
    data_path='scratch/data/tiny_shakespeare.txt',
)

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
    return data, vocab_size

def get_batch(data, T, bs):
    ix = torch.randint(len(data) - T, (bs,))
    x = torch.stack([data[i:i+T]   for i in ix])
    y = torch.stack([data[i+1:i+T+1] for i in ix])
    return x, y

# ══════════════════════════════════════════════════════════════════════
# nGPT & SPHERICAL UTILS
# ══════════════════════════════════════════════════════════════════════
def norm_sphere(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

class SphericalHead(nn.Module):
    """
    Head final basado en similitud coseno con temperatura aprendible.
    """
    def __init__(self, in_features, out_features, init_tau=10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        self.tau = nn.Parameter(torch.tensor(init_tau))

    def forward(self, x):
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        return F.linear(x_norm, w_norm) * self.tau

# ══════════════════════════════════════════════════════════════════════
# WALSH/HADAMARD UTILS
# ══════════════════════════════════════════════════════════════════════
def get_walsh_matrix_1d(dim):
    if dim == 1:
        return torch.tensor([[1.]])
    H = get_walsh_matrix_1d(dim // 2)
    return torch.cat([
        torch.cat([H, H], dim=1),
        torch.cat([H, -H], dim=1)
    ], dim=0) / math.sqrt(2)

class WalshLinear(nn.Module):
    def __init__(self, in_features, out_features, k, normalized=True):
        super().__init__()
        self.k = k
        self.in_features = in_features
        self.out_features = out_features
        self.normalized = normalized
        self.core = nn.Parameter(torch.randn(k, k) / math.sqrt(k))
        self.scale = nn.Parameter(torch.ones(1)) if normalized else None
        self.register_buffer('H_in', get_walsh_matrix_1d(in_features))
        self.register_buffer('H_out', get_walsh_matrix_1d(out_features))

    def forward(self, x):
        W_synthesized = self.H_out[:, :self.k] @ self.core @ self.H_in[:self.k, :] 
        if self.normalized:
            w = F.normalize(W_synthesized, dim=-1)
            return F.linear(x, w) * self.scale
        else:
            return F.linear(x, W_synthesized)

# ══════════════════════════════════════════════════════════════════════
# MIXER & FFN (Matrix-Free)
# ══════════════════════════════════════════════════════════════════════
class CausalComplexFFTMixer(nn.Module):
    def __init__(self, T, D, k_walsh):
        super().__init__()
        self.T = T
        self.pad_T = 1
        while self.pad_T < 2*T: self.pad_T *= 2
        self.n_freq = self.pad_T // 2 + 1
        
        self.log_amp = nn.Parameter(torch.zeros(self.n_freq))
        self.phase   = nn.Parameter(torch.zeros(self.n_freq))
        
        mask = torch.zeros(self.pad_T)
        mask[:T] = 1.0
        self.register_buffer('causal_mask', mask)
        self.out_proj = WalshLinear(D, D, k_walsh, normalized=True)

    def forward(self, x):
        B, T, D = x.shape
        xt = x.permute(0, 2, 1)
        pad = torch.zeros(B, D, self.pad_T-T, device=x.device)
        xt_pad = torch.cat([xt, pad], dim=-1)
        X = torch.fft.rfft(xt_pad, dim=-1)

        gate_raw  = torch.exp(self.log_amp) * torch.exp(1j * self.phase)
        h_raw     = torch.fft.irfft(gate_raw, n=self.pad_T)
        h_causal  = h_raw * self.causal_mask
        gate_causal = torch.fft.rfft(h_causal, n=self.pad_T)

        out = torch.fft.irfft(X * gate_causal, n=self.pad_T, dim=-1)[..., :T]
        out = out.permute(0, 2, 1)
        return self.out_proj(out)

    def get_phase_loss(self):
        """Calcula la pérdida de continuidad (suavidad) en el espectro de la fase."""
        diffs = self.phase[1:] - self.phase[:-1]
        return torch.mean(torch.abs(diffs))

class NarrowFFN(nn.Module):
    def __init__(self, D, k_walsh):
        super().__init__()
        self.proj = WalshLinear(D, D, k_walsh, normalized=True)
            
    def forward(self, x):
        return F.gelu(self.proj(x))

# ══════════════════════════════════════════════════════════════════════
# MODELO
# ══════════════════════════════════════════════════════════════════════
class nGPTBlock(nn.Module):
    def __init__(self, D, T, k_walsh, alpha_init=0.05):
        super().__init__()
        self.mixer = CausalComplexFFTMixer(T, D, k_walsh)
        self.ffn = NarrowFFN(D, k_walsh)
        self.alpha_mixer = nn.Parameter(torch.full((D,), alpha_init))
        self.alpha_ffn   = nn.Parameter(torch.full((D,), alpha_init))

    def forward(self, x):
        m = norm_sphere(self.mixer(x))
        alpha = self.alpha_mixer.abs().unsqueeze(0).unsqueeze(0)
        x = norm_sphere(x + alpha * m)
        f = norm_sphere(self.ffn(x))
        alpha = self.alpha_ffn.abs().unsqueeze(0).unsqueeze(0)
        x = norm_sphere(x + alpha * f)
        return x

class LM(nn.Module):
    def __init__(self, vocab_size, T, D, L, k_walsh, use_spherical_head=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, D)
        self.blocks = nn.ModuleList([nGPTBlock(D, T, k_walsh) for _ in range(L)])
        
        if use_spherical_head:
            self.head = SphericalHead(D, vocab_size)
        else:
            self.head = nn.Linear(D, vocab_size, bias=False)

    def forward(self, x):
        h = norm_sphere(self.embed(x))
        for b in self.blocks: h = b(h)
        return self.head(h)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

# ══════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_loss(model, data, T, bs, vocab_size, n=50):
    model.eval()
    losses = []
    for _ in range(n):
        x, y = get_batch(data, T, bs)
        loss = F.cross_entropy(model(x).reshape(-1, vocab_size), y.reshape(-1))
        losses.append(loss.item())
    return sum(losses) / len(losses)

def run(label, model, use_phase_reg, train_data, val_data, vocab_size, lr, epochs):
    T = CFG['seq_len']; bs = CFG['batch_size']
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  params={model.n_params():,} | lr={lr} | epochs={epochs}")
    print(f"{'='*65}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_val = float('inf'); conv_epoch = None; t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train(); ep_loss = 0.0
        for b in range(CFG['steps_per_epoch']):
            x, y = get_batch(train_data, T, bs)
            logits = model(x)
            loss_ce = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            
            # Phase Regularization
            loss_phase = 0.0
            if use_phase_reg:
                for m in model.modules():
                    if isinstance(m, CausalComplexFFTMixer):
                        loss_phase += m.get_phase_loss()
                loss = loss_ce + CFG['lambda_phase'] * loss_phase
            else:
                loss = loss_ce

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ep_loss += loss_ce.item()
            
            if ep == 1 and b < 5:
                # Verificamos que tau se está aprendiendo si existe
                tau_val = f" | tau={model.head.tau.item():.2f}" if hasattr(model.head, 'tau') else ""
                print(f"  [Ep1 B{b+1}] train={loss_ce.item():.4f}{tau_val}")

        sched.step()

        val = eval_loss(model, val_data, T, bs, vocab_size, CFG['val_steps'])
        if val < best_val: best_val = val
        if conv_epoch is None and val < 2.0: conv_epoch = ep
        
        tau_val = f" | tau={model.head.tau.item():.2f}" if hasattr(model.head, 'tau') else ""
        print(f"  Ep {ep:02d} | train={ep_loss/CFG['steps_per_epoch']:.4f} | val={val:.4f}{tau_val}")

    elapsed = time.time() - t0
    ppl = math.exp(min(best_val, 10))

    print(f"  BEST_VAL={best_val:.4f} | PPL={ppl:.2f} | "
          f"Conv={'Ep'+str(conv_epoch) if conv_epoch else 'Never'} | {elapsed:.1f}s")
    return dict(label=label, val=best_val, ppl=ppl, params=model.n_params(),
                conv=conv_epoch, time=elapsed)

if __name__ == '__main__':
    data, vocab_size = load_data(CFG)
    n_train = int(len(data) * 0.9)
    train_data, val_data = data[:n_train], data[n_train:]
    T, D, L, K = CFG['seq_len'], CFG['d_model'], CFG['n_layers'], CFG['k_walsh']

    print(f"\nV284: Spherical Loss & Phase Continuity (Matrix-Free k={K})")

    experiments = [
        ("A_V283_Baseline         ", False, False),
        ("B_SphericalLoss         ", True, False),
        ("C_Spherical_and_PhaseReg", True, True),
    ]

    results = []
    for label, use_spherical, use_phase in experiments:
        torch.manual_seed(CFG['seed'])
        model = LM(vocab_size, T, D, L, K, use_spherical_head=use_spherical)
        r = run(label, model, use_phase, train_data, val_data, vocab_size, CFG['lr'], CFG['epochs'])
        results.append(r)

    print(f"\n\n{'='*75}")
    print(f"  V284 SUMMARY — Spherical Loss Benchmark")
    print(f"{'='*75}")
    print(f"  {'Model':<26} {'Params':>8} {'ValLoss':>8} {'PPL':>6} {'Conv':>6} {'Time':>6}")
    print(f"  {'-'*73}")
    for r in sorted(results, key=lambda x: x['val']):
        c = f"Ep{r['conv']}" if r['conv'] else "Never"
        print(f"  {r['label']:<26} {r['params']:>8,} {r['val']:>8.4f} {r['ppl']:>6.2f} {c:>6} {r['time']:>5.1f}s")
    print(f"{'='*75}")
