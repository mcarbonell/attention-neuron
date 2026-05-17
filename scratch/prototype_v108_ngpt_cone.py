"""
prototype_v108_ngpt_cone.py
===========================
V108: nGPT + ConeAttn + NarrowFFN

nGPT (NVIDIA 2024): all hidden states live on the unit hypersphere S^(d-1).
Updates: h <- normalize(h + alpha * f(h))
Eigen learning rates alpha ∈ R^d control step size per dimension.

Key hypotheses:
  1. nGPT alone converges faster than standard Transformer
  2. nGPT + ConeAttn maintains the convergence speed gain
  3. nGPT rehabilitates DimGate: normalize(x + alpha*DimGate(f(x)))
     is no longer collapsible because normalization is the nonlinearity
  4. nGPT + ConeAttn + DimGate ≈ nGPT's eigen learning rates paper claim

Configs (all d=128, L=3 for fair comparison):
  A) Standard Transformer         [baseline]
  B) nGPT Transformer             [ngpt baseline]
  C) nGPT + ConeAttn + Dense      [proposed: best of V103 + nGPT]
  D) nGPT + ConeAttn + Narrow     [proposed: compressed + nGPT]
  E) nGPT + Attn + DimGate        [DimGate rehabilitated?]
  F) nGPT + ConeAttn + DimGate    [most compressed + nGPT]
"""

import torch, torch.nn as nn, torch.nn.functional as F
import math, time, os, urllib.request

CFG = dict(
    seq_len=128, batch_size=64,
    d_model=128, n_layers=3, n_heads=4,
    n_cones_attn=32,
    epochs=20, lr=3e-3, seed=42,
    steps_per_epoch=200, val_steps=50,
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
    print(f"Dataset: {len(text):,} chars | vocab={vocab_size}")
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
    """Project x onto unit hypersphere: ||x|| = 1 per token."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)

class NormalizedLinear(nn.Linear):
    """Linear layer with weight rows normalized to unit norm."""
    def forward(self, x):
        w = F.normalize(self.weight, dim=-1)
        return F.linear(x, w, self.bias)


# ══════════════════════════════════════════════════════════════════════
# MIXERS
# ══════════════════════════════════════════════════════════════════════
class CausalSelfAttention(nn.Module):
    def __init__(self, T, D, n_heads, normalized=False):
        super().__init__()
        Linear = NormalizedLinear if normalized else nn.Linear
        self.qkv = Linear(D, 3 * D, bias=False)
        self.out  = Linear(D, D,     bias=False)
        self.n_heads = n_heads
        self.head_dim = D // n_heads
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # normalize q, k to unit sphere per head (nGPT style)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        scale = math.sqrt(dh)
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(self.mask[:T, :T].unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class Cone1DTemporalMixer(nn.Module):
    def __init__(self, T, D, n_cones, normalized=False):
        super().__init__()
        Linear = NormalizedLinear if normalized else nn.Linear
        self.offset = nn.Parameter(torch.linspace(0, T//2, n_cones))
        self.radius = nn.Parameter(torch.ones(n_cones) * (T / n_cones * 2))
        self.amplitude = nn.Parameter(torch.empty(n_cones).uniform_(-0.5, 0.5))
        self.v_proj   = Linear(D, n_cones, bias=False)
        self.out_proj = Linear(n_cones, D, bias=False)
        pos = torch.arange(T).unsqueeze(1) - torch.arange(T).unsqueeze(0)
        self.register_buffer('rel_pos', pos.float())
        self.register_buffer('causal_mask', torch.tril(torch.ones(T, T)))

    def forward(self, x):
        B, T, D = x.shape
        V = self.v_proj(x)
        radius = F.softplus(self.radius) + 1e-4
        offset = F.softplus(self.offset)
        dist = torch.abs(self.rel_pos[:T, :T].unsqueeze(-1) - offset.unsqueeze(0).unsqueeze(0))
        weights = F.relu(1.0 - dist / radius.unsqueeze(0).unsqueeze(0))
        weights = weights * self.amplitude.unsqueeze(0).unsqueeze(0)
        weights = weights * self.causal_mask[:T, :T].unsqueeze(-1)
        weight_sum = weights.abs().sum(dim=1, keepdim=True) + 1e-8
        weights = weights / weight_sum
        out = torch.einsum('tjc,bjc->btc', weights, V)
        return self.out_proj(out)


# ══════════════════════════════════════════════════════════════════════
# FFNs
# ══════════════════════════════════════════════════════════════════════
class DenseFFN(nn.Module):
    def __init__(self, D, normalized=False):
        super().__init__()
        Linear = NormalizedLinear if normalized else nn.Linear
        self.up   = Linear(D, 4*D)
        self.down = Linear(4*D, D)
    def forward(self, x):
        return self.down(F.gelu(self.up(x)))

class NarrowFFN(nn.Module):
    def __init__(self, D, normalized=False):
        super().__init__()
        Linear = NormalizedLinear if normalized else nn.Linear
        self.net = nn.Sequential(Linear(D, D), nn.GELU())
    def forward(self, x): return self.net(x)

class DimGateFFN(nn.Module):
    """The rehabilitated DimGate: within nGPT update, no longer collapsible."""
    def __init__(self, D):
        super().__init__()
        self.gate  = nn.Parameter(torch.zeros(D))
        self.scale = nn.Parameter(torch.ones(D))
    def forward(self, x):
        return x * (self.scale * torch.sigmoid(self.gate))


# ══════════════════════════════════════════════════════════════════════
# BLOCKS
# ══════════════════════════════════════════════════════════════════════
class StandardBlock(nn.Module):
    """Standard Transformer block with LayerNorm."""
    def __init__(self, mixer, ffn, D):
        super().__init__()
        self.mixer = mixer; self.ffn = ffn
        self.norm1 = nn.LayerNorm(D); self.norm2 = nn.LayerNorm(D)
    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class nGPTBlock(nn.Module):
    """nGPT block: updates keep x on the unit hypersphere.
    h <- normalize(h + alpha * f(h))
    alpha ∈ R^d are the 'eigen learning rates' (learnable per-dim step sizes).
    """
    def __init__(self, mixer, ffn, D, alpha_init=0.05):
        super().__init__()
        self.mixer = mixer; self.ffn = ffn
        # Eigen learning rates: per-dimension step sizes on the hypersphere
        self.alpha_mixer = nn.Parameter(torch.full((D,), alpha_init))
        self.alpha_ffn   = nn.Parameter(torch.full((D,), alpha_init))

    def forward(self, x):
        # x lives on S^(d-1): ||x_i|| = 1 for each token i
        # Mixer update
        m = norm_sphere(self.mixer(x))                    # normalize output to sphere
        alpha = self.alpha_mixer.abs().unsqueeze(0).unsqueeze(0)  # (1,1,D)
        x = norm_sphere(x + alpha * m)                   # slerp step + renormalize

        # FFN update
        f = norm_sphere(self.ffn(x))
        alpha = self.alpha_ffn.abs().unsqueeze(0).unsqueeze(0)
        x = norm_sphere(x + alpha * f)
        return x


# ══════════════════════════════════════════════════════════════════════
# LM
# ══════════════════════════════════════════════════════════════════════
class LM(nn.Module):
    def __init__(self, vocab_size, T, D, blocks, use_ngpt=False):
        super().__init__()
        self.use_ngpt = use_ngpt
        self.embed = nn.Embedding(vocab_size, D)
        if not use_ngpt:
            self.pe = self._make_pe(T, D)
        self.blocks = nn.ModuleList(blocks)
        self.ln_final = nn.LayerNorm(D) if not use_ngpt else None
        self.head = nn.Linear(D, vocab_size, bias=False)

    def _make_pe(self, T, D):
        pe = torch.zeros(T, D)
        pos = torch.arange(T).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000)/D))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x):
        h = self.embed(x)
        if self.use_ngpt:
            h = norm_sphere(h)   # project embeddings onto sphere
        else:
            h = h + self.pe[:, :h.shape[1]]
        for b in self.blocks:
            h = b(h)
        if self.ln_final is not None:
            h = self.ln_final(h)
        return self.head(h)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════
def make_model(vocab_size, T, D, L, n_heads, mixer_type, ffn_type, use_ngpt, n_cones=32):
    Block = nGPTBlock if use_ngpt else StandardBlock
    blocks = []
    for _ in range(L):
        if mixer_type == 'attention':
            mixer = CausalSelfAttention(T, D, n_heads, normalized=use_ngpt)
        else:
            mixer = Cone1DTemporalMixer(T, D, n_cones, normalized=use_ngpt)

        if ffn_type == 'dense':
            ffn = DenseFFN(D, normalized=use_ngpt)
        elif ffn_type == 'narrow':
            ffn = NarrowFFN(D, normalized=use_ngpt)
        elif ffn_type == 'dimgate':
            ffn = DimGateFFN(D)
        blocks.append(Block(mixer, ffn, D))
    return LM(vocab_size, T, D, blocks, use_ngpt=use_ngpt)


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

def run(label, model, train_data, val_data, vocab_size):
    T = CFG['seq_len']; bs = CFG['batch_size']
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  params={model.n_params():,}")
    print(f"{'='*65}")

    opt = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG['epochs'])
    best_val = float('inf'); conv_epoch = None; t0 = time.time()

    for ep in range(1, CFG['epochs'] + 1):
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

    # Report alpha stats for nGPT models
    for i, blk in enumerate(model.blocks):
        if hasattr(blk, 'alpha_mixer'):
            am = blk.alpha_mixer.abs().detach()
            af = blk.alpha_ffn.abs().detach()
            print(f"  L{i} alpha_mixer=[{am.min():.3f},{am.mean():.3f},{am.max():.3f}] "
                  f"alpha_ffn=[{af.min():.3f},{af.mean():.3f},{af.max():.3f}]")

    print(f"  BEST_VAL={best_val:.4f} | PPL={ppl:.2f} | "
          f"Conv={'Ep'+str(conv_epoch) if conv_epoch else 'Never'} | {elapsed:.1f}s")
    return dict(label=label, val=best_val, ppl=ppl, params=model.n_params(),
                conv=conv_epoch, time=elapsed)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    data, vocab_size = load_data(CFG)
    n_train = int(len(data) * 0.9)
    train_data, val_data = data[:n_train], data[n_train:]
    T, D, L, H = CFG['seq_len'], CFG['d_model'], CFG['n_layers'], CFG['n_heads']
    C = CFG['n_cones_attn']

    print(f"\nV108: nGPT + ConeAttn + NarrowFFN")
    print(f"Seq={T} | d={D} | L={L} | epochs={CFG['epochs']}")
    print(f"Hypothesis: nGPT normalization rehabilitates DimGate & boosts ConeAttn")

    # label, mixer, ffn, use_ngpt
    experiments = [
        # PROPOSED FIRST
        ("C_nGPT+Cone+Dense   [proposed]", 'cone',      'dense',   True),
        ("D_nGPT+Cone+Narrow  [proposed]", 'cone',      'narrow',  True),
        ("E_nGPT+Attn+DimGate [DG rehab]", 'attention', 'dimgate', True),
        ("F_nGPT+Cone+DimGate [DG rehab]", 'cone',      'dimgate', True),
        # Baselines
        ("A_Standard Transformer[baseline]",'attention','dense',   False),
        ("B_nGPT Transformer    [ngpt ref]",'attention','dense',   True),
    ]

    results = []
    for label, mixer, ffn, use_ngpt in experiments:
        torch.manual_seed(CFG['seed'])
        model = make_model(vocab_size, T, D, L, H, mixer, ffn, use_ngpt, C)
        r = run(label, model, train_data, val_data, vocab_size)
        results.append(r)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n\n{'='*72}")
    print(f"  V108 SUMMARY — nGPT + Cone Architectures")
    print(f"{'='*72}")
    print(f"  {'Model':<42} {'Params':>8} {'ValLoss':>8} {'PPL':>6} {'Conv':>6} {'Time':>6}")
    print(f"  {'-'*70}")

    for r in sorted(results, key=lambda x: x['val']):
        c = f"Ep{r['conv']}" if r['conv'] else "Never"
        print(f"  {r['label']:<42} {r['params']:>8,} {r['val']:>8.4f} {r['ppl']:>6.2f} {c:>6} {r['time']:>5.1f}s")

    print(f"{'='*72}")

    # ── Key comparisons ───────────────────────────────────────────────
    std   = next(r for r in results if 'Standard' in r['label'])
    ngpt  = next(r for r in results if 'nGPT Transformer' in r['label'])
    dg_r  = next((r for r in results if 'DG rehab' in r['label'] and 'Attn' in r['label']), None)
    cone_n = next((r for r in results if 'Cone+Narrow' in r['label']), None)

    print(f"\n  KEY RESULTS:")
    print(f"  nGPT convergence gain: {ngpt['val']:.4f} vs {std['val']:.4f} "
          f"({'faster' if ngpt['conv'] and std['conv'] and ngpt['conv'] < std['conv'] else 'same'} conv)")
    if dg_r:
        print(f"  DimGate rehabilitated: {dg_r['val']:.4f} vs V105 baseline 1.6298 "
              f"({'BETTER' if dg_r['val'] < 1.6298 else 'worse'})")
    if cone_n:
        delta = cone_n['val'] - std['val']
        print(f"  nGPT+Cone+Narrow:     {cone_n['val']:.4f} vs std {std['val']:.4f} ({delta:+.4f})")
