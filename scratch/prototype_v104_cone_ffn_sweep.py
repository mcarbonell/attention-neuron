"""
prototype_v104_cone_ffn_sweep.py
================================
V104: ConeFFN Radius Collapse Investigation

V103 showed ConeFFN radii collapsed to ~1 dimension (each neuron reads
a single dim). User hypothesis: "this is a FEATURE, not a bug — the FFN
is massively overparameterized".

This experiment tests 3 axes:
  1. Forced minimum radius (floor) — does forcing wider cones help or hurt?
  2. Gaussian vs triangular cone shape — does smoothness prevent collapse?
  3. Scaled d_model (64 → 128 → 256) — does more dims allow topology?

All configs use standard causal self-attention (from V103 baseline).
Only the FFN varies. This isolates the ConeFFN behavior.

Key diagnostic: if forcing wider cones HURTS performance, then the collapse
IS the optimal solution and dense FFN is indeed overparameterized.
"""

import torch, torch.nn as nn, torch.nn.functional as F
import math, time, os, urllib.request

CFG = dict(
    seq_len=128, batch_size=64,
    n_layers=3, n_heads=4,
    n_cones_ffn=256,
    ffn_mult=4,
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

# ── Positional Encoding ──────────────────────────────────────────────
class SinCosPE(nn.Module):
    def __init__(self, D, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, D)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000)/D))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]


# ══════════════════════════════════════════════════════════════════════
# CONE 1D FFN VARIANTS
# ══════════════════════════════════════════════════════════════════════

class Cone1DFFN(nn.Module):
    """
    ConeFFN with configurable shape and minimum radius.
    
    shape: 'triangular' or 'gaussian'
    min_radius: minimum radius in dimensions (0 = no floor)
    """
    def __init__(self, D, n_neurons, shape='triangular', min_radius=0.0):
        super().__init__()
        self.D = D
        self.n_neurons = n_neurons
        self.shape = shape
        self.min_radius_val = min_radius
        
        # 4 params per neuron
        self.center = nn.Parameter(torch.linspace(0, D-1, n_neurons))
        self.raw_radius = nn.Parameter(torch.ones(n_neurons) * (D / n_neurons * 2))
        self.amplitude = nn.Parameter(torch.empty(n_neurons).uniform_(-1.0, 1.0))
        self.bias = nn.Parameter(torch.zeros(n_neurons))
        
        # Output projection
        self.out_proj = nn.Linear(n_neurons, D, bias=False)
        
        # Dim positions
        self.register_buffer('dim_positions', torch.arange(D).float())
    
    def get_radius(self):
        """Apply softplus + optional floor."""
        r = F.softplus(self.raw_radius) + 1e-4
        if self.min_radius_val > 0:
            r = r + self.min_radius_val
        return r
    
    def forward(self, x):
        center = self.center.unsqueeze(1)       # (N, 1)
        radius = self.get_radius().unsqueeze(1)  # (N, 1)
        amp = self.amplitude.unsqueeze(1)        # (N, 1)
        dim_pos = self.dim_positions.unsqueeze(0) # (1, D)
        
        dist = torch.abs(dim_pos - center)  # (N, D)
        
        if self.shape == 'gaussian':
            # Gaussian: exp(-0.5 * (dist/sigma)^2) * amplitude
            cone_weights = torch.exp(-0.5 * (dist / radius) ** 2) * amp
        else:
            # Triangular: max(0, 1 - dist/radius) * amplitude
            cone_weights = F.relu(1.0 - dist / radius) * amp
        
        hidden = F.linear(x, cone_weights, self.bias)
        hidden = F.relu(hidden)
        return self.out_proj(hidden)


class DenseFFN(nn.Module):
    def __init__(self, D, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, D * mult), nn.GELU(), nn.Linear(D * mult, D),
        )
    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════
# STANDARD ATTENTION + BLOCK
# ══════════════════════════════════════════════════════════════════════
class CausalSelfAttention(nn.Module):
    def __init__(self, T, D, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(D, n_heads, batch_first=True)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        self.register_buffer('attn_mask', mask)
    def forward(self, x):
        T = x.shape[1]
        out, _ = self.attn(x, x, x, attn_mask=self.attn_mask[:T, :T])
        return out


class TransformerBlock(nn.Module):
    def __init__(self, mixer, ffn, D):
        super().__init__()
        self.mixer = mixer
        self.ffn = ffn
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
    def forward(self, x):
        x = self.norm1(x + self.mixer(x))
        x = self.norm2(x + self.ffn(x))
        return x


class LM(nn.Module):
    def __init__(self, vocab_size, T, D, blocks):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, D)
        self.pe = SinCosPE(D)
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(D, vocab_size, bias=False)
    def forward(self, x):
        h = self.pe(self.embed(x))
        for b in self.blocks: h = b(h)
        return self.head(h)
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ══════════════════════════════════════════════════════════════════════
def make_model(vocab_size, T, D, n_layers, n_heads, ffn_type, n_cones=256,
               cone_shape='triangular', min_radius=0.0, ffn_mult=4):
    blocks = []
    for _ in range(n_layers):
        mixer = CausalSelfAttention(T, D, n_heads)
        if ffn_type == 'dense':
            ffn = DenseFFN(D, ffn_mult)
        elif ffn_type == 'cone':
            ffn = Cone1DFFN(D, n_cones, shape=cone_shape, min_radius=min_radius)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")
        blocks.append(TransformerBlock(mixer, ffn, D))
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


def run(label, model, train_data, val_data, vocab_size):
    T = CFG['seq_len']; bs = CFG['batch_size']
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  params={model.n_params():,}")
    print(f"{'='*65}")
    
    opt = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG['epochs'])
    best_val = float('inf')
    conv_epoch = None
    t0 = time.time()
    
    for ep in range(1, CFG['epochs'] + 1):
        model.train()
        ep_loss = 0.0
        for b in range(CFG['steps_per_epoch']):
            x, y = get_batch(train_data, T, bs)
            loss = F.cross_entropy(model(x).reshape(-1, vocab_size), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            if ep == 1 and b < 5:
                print(f"  [Ep1 B{b+1}] train={loss.item():.4f}")
        sched.step()
        
        val = eval_loss(model, val_data, T, bs, vocab_size, CFG['val_steps'])
        if val < best_val: best_val = val
        if conv_epoch is None and val < 2.0: conv_epoch = ep
        print(f"  Ep {ep:02d} | train={ep_loss/CFG['steps_per_epoch']:.4f} | val={val:.4f}")
    
    elapsed = time.time() - t0
    ppl = math.exp(min(best_val, 10))
    
    # Cone diagnostics
    for i, block in enumerate(model.blocks):
        if hasattr(block.ffn, 'center'):
            r = block.ffn.get_radius().detach()
            amps = block.ffn.amplitude.detach()
            n_ex = (amps > 0).sum().item()
            n_in = (amps < 0).sum().item()
            print(f"  L{i} FFN: r=[{r.min():.2f}, {r.median():.2f}, {r.max():.2f}] "
                  f"excit={n_ex} inhib={n_in}")
    
    print(f"  BEST_VAL={best_val:.4f} | PPL={ppl:.2f} | "
          f"Conv={'Ep'+str(conv_epoch) if conv_epoch else 'Never'} | {elapsed:.1f}s")
    return dict(label=label, val=best_val, ppl=ppl, params=model.n_params(),
                conv=conv_epoch, time=elapsed)


# ══════════════════════════════════════════════════════════════════════
# MAIN — Sweep
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    data, vocab_size = load_data(CFG)
    n_train = int(len(data) * 0.9)
    train_data, val_data = data[:n_train], data[n_train:]
    T, NL = CFG['seq_len'], CFG['n_layers']
    
    print(f"\nV104: ConeFFN Radius Collapse Investigation")
    print(f"Seq={T} | layers={NL} | epochs={CFG['epochs']}")
    print(f"Hypothesis: radius collapse = FFN overparameterization signal")
    
    # Define sweep configurations
    # Format: (label, d_model, ffn_type, n_cones, shape, min_radius)
    experiments = [
        # ── PROPOSED FIRST (candidatos) ──
        # 1. Original collapse (reproduce V103)
        ("ConeFFN_tri_nofloor_d64",     64,  'cone', 256, 'triangular', 0.0),
        # 2. Forced minimum radius
        ("ConeFFN_tri_floor4_d64",      64,  'cone', 256, 'triangular', 4.0),
        ("ConeFFN_tri_floor8_d64",      64,  'cone', 256, 'triangular', 8.0),
        # 3. Gaussian shape (smoother)
        ("ConeFFN_gauss_nofloor_d64",   64,  'cone', 256, 'gaussian',   0.0),
        ("ConeFFN_gauss_floor4_d64",    64,  'cone', 256, 'gaussian',   4.0),
        # 4. Scaled d_model (more room for topology)
        ("ConeFFN_tri_nofloor_d128",    128, 'cone', 256, 'triangular', 0.0),
        ("ConeFFN_gauss_nofloor_d128",  128, 'cone', 256, 'gaussian',   0.0),
        # ── BASELINES (control) ──
        ("Dense_FFN_d64  [baseline]",   64,  'dense', 0,  '',           0.0),
        ("Dense_FFN_d128 [baseline]",   128, 'dense', 0,  '',           0.0),
    ]
    
    results = []
    for label, d_model, ffn_type, n_cones, shape, min_r in experiments:
        torch.manual_seed(CFG['seed'])
        model = make_model(
            vocab_size, T, d_model, NL, CFG['n_heads'],
            ffn_type=ffn_type, n_cones=n_cones,
            cone_shape=shape, min_radius=min_r,
            ffn_mult=CFG['ffn_mult'],
        )
        r = run(label, model, train_data, val_data, vocab_size)
        results.append(r)
    
    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n\n{'='*75}")
    print(f"  V104 SUMMARY — ConeFFN Radius Collapse Investigation")
    print(f"{'='*75}")
    print(f"  {'Model':<38} {'Params':>8} {'ValLoss':>8} {'PPL':>6} {'Conv':>6} {'Time':>6}")
    print(f"  {'-'*73}")
    
    # Group by d_model
    for d in [64, 128]:
        group = [r for r in results if f'd{d}' in r['label'] or f'd{d} ' in r['label']]
        if not group: continue
        print(f"  --- d_model={d} ---")
        for r in sorted(group, key=lambda x: x['val']):
            c = f"Ep{r['conv']}" if r['conv'] else "Never"
            print(f"  {r['label']:<38} {r['params']:>8,} {r['val']:>8.4f} {r['ppl']:>6.2f} {c:>6} {r['time']:>5.1f}s")
    
    print(f"{'='*75}")
    
    # ── Key comparisons ───────────────────────────────────────────────
    baseline_d64 = next((r for r in results if 'Dense' in r['label'] and 'd64' in r['label']), None)
    baseline_d128 = next((r for r in results if 'Dense' in r['label'] and 'd128' in r['label']), None)
    
    print(f"\n  KEY QUESTIONS:")
    
    # Q1: Does forcing wider radius help or hurt?
    nofloor = next((r for r in results if 'tri_nofloor_d64' in r['label']), None)
    floor4  = next((r for r in results if 'tri_floor4_d64' in r['label']), None)
    floor8  = next((r for r in results if 'tri_floor8_d64' in r['label']), None)
    if nofloor and floor4 and floor8:
        print(f"\n  Q1: Does forcing wider cones help?")
        print(f"    no_floor: {nofloor['val']:.4f}")
        print(f"    floor=4:  {floor4['val']:.4f} ({floor4['val']-nofloor['val']:+.4f})")
        print(f"    floor=8:  {floor8['val']:.4f} ({floor8['val']-nofloor['val']:+.4f})")
        if floor4['val'] > nofloor['val'] and floor8['val'] > nofloor['val']:
            print(f"    → WIDER CONES HURT. Collapse IS optimal. FFN overparameterization confirmed.")
        elif floor4['val'] < nofloor['val']:
            print(f"    → WIDER CONES HELP. Collapse was suboptimal. Topology may emerge.")
    
    # Q2: Gaussian vs triangular
    tri = nofloor
    gauss = next((r for r in results if 'gauss_nofloor_d64' in r['label']), None)
    if tri and gauss:
        print(f"\n  Q2: Gaussian vs triangular (d=64)?")
        print(f"    triangular: {tri['val']:.4f}")
        print(f"    gaussian:   {gauss['val']:.4f} ({gauss['val']-tri['val']:+.4f})")
    
    # Q3: Does d=128 change the collapse behavior?
    d128_tri = next((r for r in results if 'tri_nofloor_d128' in r['label']), None)
    if d128_tri and baseline_d128:
        print(f"\n  Q3: ConeFFN at d=128?")
        print(f"    ConeFFN d128: {d128_tri['val']:.4f} (params={d128_tri['params']:,})")
        print(f"    Dense   d128: {baseline_d128['val']:.4f} (params={baseline_d128['params']:,})")
        ratio = d128_tri['params'] / baseline_d128['params']
        delta = d128_tri['val'] - baseline_d128['val']
        print(f"    delta={delta:+.4f} | param_ratio={ratio:.2f}x")
