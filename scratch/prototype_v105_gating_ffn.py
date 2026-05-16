"""
prototype_v105_gating_ffn.py
============================
V105: Is FFN just a dimension gate?

V104 proved ConeFFN radii collapse to ~1 dimension = each neuron reads
a single feature. This suggests FFN is a sparse dimension selector.

Test: replace FFN with increasingly simple gating mechanisms:

A) DimGate:      output = x * sigmoid(g),  g ∈ R^d           → d params
B) DimGateScale: output = x * (g_scale * sigmoid(g_gate)),    → 2d params  
C) DimGateBias:  output = x * sigmoid(g) + bias               → 2d params
D) NarrowFFN:    Linear(d→d) → GELU → no down-proj            → d² + d params
E) BottleneckFFN: Linear(d→k) → GELU → Linear(k→d), k=d//4   → d*k*2 params
F) ConeFFN:      from V104 (reference, 256 neurons, no floor) → ~34K params
G) DenseFFN:     standard d→4d→d                              → d*4d*2 params

If DimGate (d params) ≈ ConeFFN (~34K) ≈ Dense (~131K):
→ FFN is literally an on/off mask per dimension. Revolutionary.

If DimGate fails but BottleneckFFN works:
→ FFN needs SOME recombination, but much less than 4d.
"""

import torch, torch.nn as nn, torch.nn.functional as F
import math, time, os, urllib.request

CFG = dict(
    seq_len=128, batch_size=64,
    d_model=128, n_layers=3, n_heads=4,
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
# FFN VARIANTS — from simplest to most complex
# ══════════════════════════════════════════════════════════════════════

class DimGateFFN(nn.Module):
    """A) Simplest possible: learned sigmoid mask per dimension.
    output = x * sigmoid(g)
    Params: d
    """
    def __init__(self, D):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(D))  # init at sigmoid(0) = 0.5
    
    def forward(self, x):
        return x * torch.sigmoid(self.gate)


class DimGateScaleFFN(nn.Module):
    """B) Gate + learned scale per dimension.
    output = x * (scale * sigmoid(gate))
    Params: 2d
    """
    def __init__(self, D):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(D))
        self.scale = nn.Parameter(torch.ones(D))
    
    def forward(self, x):
        return x * (self.scale * torch.sigmoid(self.gate))


class DimGateBiasFFN(nn.Module):
    """C) Gate + bias.
    output = x * sigmoid(gate) + bias
    Params: 2d
    """
    def __init__(self, D):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(D))
        self.bias = nn.Parameter(torch.zeros(D))
    
    def forward(self, x):
        return x * torch.sigmoid(self.gate) + self.bias


class NarrowFFN(nn.Module):
    """D) Square linear: d→d with GELU. No expansion.
    Params: d² + d
    """
    def __init__(self, D):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, D), nn.GELU())
    
    def forward(self, x):
        return self.net(x)


class BottleneckFFN(nn.Module):
    """E) Bottleneck: d→k→d with k=d//4.
    Tests if FFN needs recombination but less than 4d.
    Params: 2*d*k + k + d
    """
    def __init__(self, D, k=None):
        super().__init__()
        if k is None: k = max(D // 4, 16)
        self.net = nn.Sequential(
            nn.Linear(D, k), nn.GELU(), nn.Linear(k, D),
        )
    
    def forward(self, x):
        return self.net(x)


class Cone1DFFN(nn.Module):
    """F) ConeFFN from V104 (reference). Triangular, no floor."""
    def __init__(self, D, n_neurons=256):
        super().__init__()
        self.center = nn.Parameter(torch.linspace(0, D-1, n_neurons))
        self.raw_radius = nn.Parameter(torch.ones(n_neurons) * (D / n_neurons * 2))
        self.amplitude = nn.Parameter(torch.empty(n_neurons).uniform_(-1.0, 1.0))
        self.bias = nn.Parameter(torch.zeros(n_neurons))
        self.out_proj = nn.Linear(n_neurons, D, bias=False)
        self.register_buffer('dim_positions', torch.arange(D).float())
    
    def forward(self, x):
        center = self.center.unsqueeze(1)
        radius = (F.softplus(self.raw_radius) + 1e-4).unsqueeze(1)
        amp = self.amplitude.unsqueeze(1)
        dist = torch.abs(self.dim_positions.unsqueeze(0) - center)
        cone_weights = F.relu(1.0 - dist / radius) * amp
        hidden = F.relu(F.linear(x, cone_weights, self.bias))
        return self.out_proj(hidden)


class DenseFFN(nn.Module):
    """G) Standard dense FFN: d→4d→d."""
    def __init__(self, D, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, D * mult), nn.GELU(), nn.Linear(D * mult, D),
        )
    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════
# STANDARD ATTENTION + BLOCK + LM
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

class Block(nn.Module):
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
FFN_CLASSES = {
    'dim_gate':       lambda D: DimGateFFN(D),
    'dim_gate_scale': lambda D: DimGateScaleFFN(D),
    'dim_gate_bias':  lambda D: DimGateBiasFFN(D),
    'narrow':         lambda D: NarrowFFN(D),
    'bottleneck':     lambda D: BottleneckFFN(D),
    'cone':           lambda D: Cone1DFFN(D, n_neurons=256),
    'dense':          lambda D: DenseFFN(D, mult=4),
}

def make_model(vocab_size, T, D, n_layers, n_heads, ffn_type):
    blocks = []
    for _ in range(n_layers):
        mixer = CausalSelfAttention(T, D, n_heads)
        ffn = FFN_CLASSES[ffn_type](D)
        blocks.append(Block(mixer, ffn, D))
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
    T, D, NL = CFG['seq_len'], CFG['d_model'], CFG['n_layers']
    
    print(f"\nV105: Is FFN just a dimension gate?")
    print(f"Seq={T} | d={D} | layers={NL} | epochs={CFG['epochs']}")
    print(f"All configs use standard causal self-attention. Only FFN varies.")
    
    # Proposed first (simplest → most complex)
    experiments = [
        ('A_DimGate         [PROPOSED]',  'dim_gate'),
        ('B_DimGateScale    [proposed]',  'dim_gate_scale'),
        ('C_DimGateBias     [proposed]',  'dim_gate_bias'),
        ('D_NarrowFFN_dxd   [proposed]',  'narrow'),
        ('E_BottleneckFFN   [proposed]',  'bottleneck'),
        ('F_ConeFFN         [reference]', 'cone'),
        ('G_DenseFFN        [baseline]',  'dense'),
    ]
    
    results = []
    for label, ffn_type in experiments:
        torch.manual_seed(CFG['seed'])
        model = make_model(vocab_size, T, D, NL, CFG['n_heads'], ffn_type)
        r = run(label, model, train_data, val_data, vocab_size)
        results.append(r)
    
    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n\n{'='*75}")
    print(f"  V105 SUMMARY — Is FFN just a dimension gate?")
    print(f"{'='*75}")
    print(f"  {'Model':<38} {'Params':>8} {'FFN_p':>8} {'ValLoss':>8} {'PPL':>6} {'Conv':>6}")
    print(f"  {'-'*73}")
    
    # Calculate FFN-only params (total - attention - embeddings - head)
    # Attention per layer: 4*D*D + 4*D (Q,K,V,O with bias)
    # Embeddings: vocab*D, Head: D*vocab, PE: seq*D, LN: 2*D*layers*2
    attn_per_layer = 4 * D * D + 4 * D
    shared = vocab_size * D + D * vocab_size  # embed + head (or tied)
    
    for r in sorted(results, key=lambda x: x['val']):
        # Rough FFN params estimate
        total = r['params']
        non_ffn = shared + NL * (attn_per_layer + 4 * D) + D * CFG['seq_len']  # attn+LN+PE
        ffn_est = total - non_ffn
        c = f"Ep{r['conv']}" if r['conv'] else "Never"
        print(f"  {r['label']:<38} {total:>8,} {ffn_est:>8,} {r['val']:>8.4f} {r['ppl']:>6.2f} {c:>6}")
    
    print(f"{'='*75}")
    
    # ── Key comparisons ───────────────────────────────────────────────
    baseline = next(r for r in results if 'Dense' in r['label'])
    cone = next(r for r in results if 'Cone' in r['label'])
    dimgate = next(r for r in results if 'A_Dim' in r['label'])
    
    print(f"\n  KEY QUESTION: Is FFN just a dimension gate?")
    print(f"    DimGate  ({dimgate['params']:>7,}p): val={dimgate['val']:.4f}")
    print(f"    ConeFFN  ({cone['params']:>7,}p): val={cone['val']:.4f}")
    print(f"    DenseFFN ({baseline['params']:>7,}p): val={baseline['val']:.4f}")
    
    gap_gate_vs_cone = dimgate['val'] - cone['val']
    gap_cone_vs_dense = cone['val'] - baseline['val']
    gap_gate_vs_dense = dimgate['val'] - baseline['val']
    
    print(f"\n    DimGate vs ConeFFN:  {gap_gate_vs_cone:+.4f}")
    print(f"    ConeFFN vs Dense:    {gap_cone_vs_dense:+.4f}")
    print(f"    DimGate vs Dense:    {gap_gate_vs_dense:+.4f}")
    
    if abs(gap_gate_vs_cone) < 0.03:
        print(f"\n    >>> DimGate ~ ConeFFN: YES, FFN is literally a dimension gate!")
        print(f"        The cone's center/radius/amplitude machinery was discovering")
        print(f"        what a simple sigmoid mask already knows.")
    elif gap_gate_vs_dense > 0.1:
        print(f"\n    >>> DimGate too weak: FFN needs some recombination, not just gating.")
    
    # Find the simplest model that's within 5% of baseline
    for r in sorted(results, key=lambda x: x['params']):
        delta_pct = (r['val'] - baseline['val']) / baseline['val'] * 100
        if delta_pct < 5.0:
            print(f"\n    Cheapest model within 5% of baseline: {r['label'].split('[')[0].strip()}")
            print(f"      params={r['params']:,} ({r['params']/baseline['params']:.2%} of baseline)")
            print(f"      val={r['val']:.4f} (delta={delta_pct:+.1f}%)")
            break
