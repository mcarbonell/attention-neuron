"""
prototype_v283_matrix_free_lm.py
================================
V283: The Matrix-Free Phase-nGPT Model

Elimina las últimas matrices densas O(d^2) de la arquitectura (el out_proj del mixer causal y el NarrowFFN)
sustituyéndolas por la capa `WalshLinear` (núcleo denso sub-dimensional + transformadas de Walsh ortogonales).

Arquitecturas evaluadas:
A_Ultimate_Phase_nGPT: CausalPhase + NarrowFFN (matrices dxd, el campeón de V282)
B_MatrixFree_k64:      Versión Matrix-Free con núcleo k=64
C_MatrixFree_k32:      Versión Matrix-Free con núcleo k=32
D_MatrixFree_k16:      Versión Matrix-Free con compresión extrema k=16
"""

import torch, torch.nn as nn, torch.nn.functional as F
import math, time, os, urllib.request

CFG = dict(
    seq_len=128, batch_size=64,
    d_model=128, n_layers=3, 
    epochs=40, lr=3e-2,
    seed=42, steps_per_epoch=200, val_steps=50,
    data_url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
    data_path='scratch/data/tiny_shakespeare.txt',
)
torch.manual_seed(CFG['seed'])

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
# nGPT UTILS
# ══════════════════════════════════════════════════════════════════════
def norm_sphere(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

class NormalizedLinear(nn.Linear):
    def forward(self, x):
        w = F.normalize(self.weight, dim=-1)
        return F.linear(x, w, self.bias)

# ══════════════════════════════════════════════════════════════════════
# WALSH/HADAMARD UTILS
# ══════════════════════════════════════════════════════════════════════
def get_walsh_matrix_1d(dim):
    """Genera matriz de Walsh-Hadamard recursivamente (ortogonal, normalizada)."""
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
        
        # Núcleo de baja dimensionalidad
        self.core = nn.Parameter(torch.randn(k, k) / math.sqrt(k))
        # Escala para recuperar magnitud tras normalización
        self.scale = nn.Parameter(torch.ones(1)) if normalized else None
        
        self.register_buffer('H_in', get_walsh_matrix_1d(in_features))
        self.register_buffer('H_out', get_walsh_matrix_1d(out_features))

    def forward(self, x):
        # Síntesis on-the-fly de la matriz densa DxD usando el núcleo KxK
        # W_synthesized = H_out[:, :k] @ core @ H_in[:k, :]
        H_out_k = self.H_out[:, :self.k]   # (D, k)
        H_in_k = self.H_in[:self.k, :]     # (k, D)
        
        W_synthesized = H_out_k @ self.core @ H_in_k # (D, k) @ (k, k) @ (k, D) -> (D, D)
        
        if self.normalized:
            w = F.normalize(W_synthesized, dim=-1)
            return F.linear(x, w) * self.scale
        else:
            return F.linear(x, W_synthesized)

# ══════════════════════════════════════════════════════════════════════
# MIXERS
# ══════════════════════════════════════════════════════════════════════
class CausalComplexFFTMixer(nn.Module):
    def __init__(self, T, D, normalized=True, k_walsh=None):
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
        
        # Reemplazo Matrix-Free si k_walsh está definido
        if k_walsh is None:
            self.out_proj = NormalizedLinear(D, D, bias=False) if normalized else nn.Linear(D, D, bias=False)
        else:
            self.out_proj = WalshLinear(D, D, k_walsh, normalized=normalized)

    def forward(self, x):
        B, T, D = x.shape
        xt = x.permute(0, 2, 1) # (B, D, T)
        pad = torch.zeros(B, D, self.pad_T-T, device=x.device)
        xt_pad = torch.cat([xt, pad], dim=-1)
        X = torch.fft.rfft(xt_pad, dim=-1)

        # Gate causal
        gate_raw  = torch.exp(self.log_amp) * torch.exp(1j * self.phase)
        h_raw     = torch.fft.irfft(gate_raw, n=self.pad_T)
        h_causal  = h_raw * self.causal_mask
        gate_causal = torch.fft.rfft(h_causal, n=self.pad_T)

        out = torch.fft.irfft(X * gate_causal, n=self.pad_T, dim=-1)[..., :T]
        out = out.permute(0, 2, 1) # (B, T, D)
        return self.out_proj(out)

# ══════════════════════════════════════════════════════════════════════
# FFNs
# ══════════════════════════════════════════════════════════════════════
class NarrowFFN(nn.Module):
    def __init__(self, D, normalized=True, k_walsh=None):
        super().__init__()
        if k_walsh is None:
            Linear = NormalizedLinear if normalized else nn.Linear
            self.proj = Linear(D, D)
        else:
            self.proj = WalshLinear(D, D, k_walsh, normalized=normalized)
            
    def forward(self, x):
        return F.gelu(self.proj(x))

# ══════════════════════════════════════════════════════════════════════
# BLOCKS
# ══════════════════════════════════════════════════════════════════════
class nGPTBlock(nn.Module):
    def __init__(self, mixer, ffn, D, alpha_init=0.05):
        super().__init__()
        self.mixer = mixer; self.ffn = ffn
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
    def __init__(self, vocab_size, T, D, blocks):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, D)
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(D, vocab_size, bias=False)

    def forward(self, x):
        h = norm_sphere(self.embed(x))
        for b in self.blocks: h = b(h)
        return self.head(h)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

def make_model(vocab_size, T, D, L, k_walsh=None):
    blocks = []
    for _ in range(L):
        mixer = CausalComplexFFTMixer(T, D, normalized=True, k_walsh=k_walsh)
        ffn = NarrowFFN(D, normalized=True, k_walsh=k_walsh)
        blocks.append(nGPTBlock(mixer, ffn, D))
    return LM(vocab_size, T, D, blocks)

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

def run(label, model, train_data, val_data, vocab_size, lr, epochs):
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
            loss = F.cross_entropy(model(x).reshape(-1, vocab_size), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ep_loss += loss.item()
            if ep == 1 and b < 5:
                print(f"  [Ep1 B{b+1}] train={loss.item():.4f}")
        sched.step()

        val = eval_loss(model, val_data, T, bs, vocab_size, CFG['val_steps'])
        if val < best_val: best_val = val
        if conv_epoch is None and val < 2.0: conv_epoch = ep
        print(f"  Ep {ep:02d} | train={ep_loss/CFG['steps_per_epoch']:.4f} | val={val:.4f}")

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
    T, D, L = CFG['seq_len'], CFG['d_model'], CFG['n_layers']

    print(f"\nV283: The Matrix-Free Phase-nGPT Model")
    print(f"Seq={T} | d={D} | L={L} | Vocab={vocab_size}")

    experiments = [
        ("A_Ultimate_Phase_nGPT [dense reference]", None),
        ("B_MatrixFree_k64      [1/4 core area]  ", 64),
        ("C_MatrixFree_k32      [1/16 core area] ", 32),
        ("D_MatrixFree_k16      [1/64 core area] ", 16),
    ]

    results = []
    for label, k in experiments:
        torch.manual_seed(CFG['seed'])
        model = make_model(vocab_size, T, D, L, k_walsh=k)
        r = run(label, model, train_data, val_data, vocab_size, CFG['lr'], CFG['epochs'])
        results.append(r)

    print(f"\n\n{'='*75}")
    print(f"  V283 SUMMARY — Matrix-Free Compression Benchmark")
    print(f"{'='*75}")
    print(f"  {'Model':<42} {'Params':>8} {'ValLoss':>8} {'PPL':>6} {'Conv':>6} {'Time':>6}")
    print(f"  {'-'*73}")
    for r in sorted(results, key=lambda x: x['val']):
        c = f"Ep{r['conv']}" if r['conv'] else "Never"
        print(f"  {r['label']:<42} {r['params']:>8,} {r['val']:>8.4f} {r['ppl']:>6.2f} {c:>6} {r['time']:>5.1f}s")
    print(f"{'='*75}")
