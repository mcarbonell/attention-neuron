"""
prototype_v107_iso_budget.py
============================
V107: Iso-Budget Comparisons — Fair Fight

V105 showed DimGate at +5.9% vs Dense, same architecture (d=128, L=3).
BUT DimGate has O(d) params/layer vs O(d²) for Dense. So for the SAME
param budget, DimGate can afford:
  - MANY more layers (depth compensates?)
  - Much wider d_model (capacity compensates?)

This experiment compares architectures at FIXED budgets:
  Budget A: ~150K params (iso-param with Transformer d=64, L=3)
  Budget B: ~600K params (iso-param with Transformer d=128, L=3)

For each budget: what's the best way to spend those params?

Key insight: if DimGate at d=256, L=30 matches Dense d=128, L=3
with the same param count, then DimGate IS the right architecture
at scale — you just need to redistribute the param budget to depth.
"""

import torch, torch.nn as nn, torch.nn.functional as F
import math, time, os, urllib.request

CFG = dict(
    seq_len=128, batch_size=64,
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
    def forward(self, x): return x + self.pe[:, :x.shape[1]]


# ══════════════════════════════════════════════════════════════════════
# MIXERS
# ══════════════════════════════════════════════════════════════════════
class CausalSelfAttention(nn.Module):
    def __init__(self, T, D, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(D, n_heads, batch_first=True)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        self.register_buffer('attn_mask', mask)
    def forward(self, x):
        out, _ = self.attn(x, x, x, attn_mask=self.attn_mask[:x.shape[1], :x.shape[1]])
        return out

class Cone1DTemporalMixer(nn.Module):
    def __init__(self, T, D, n_cones):
        super().__init__()
        self.n_cones = n_cones
        self.offset = nn.Parameter(torch.linspace(0, T//2, n_cones))
        self.radius = nn.Parameter(torch.ones(n_cones) * (T / n_cones * 2))
        self.amplitude = nn.Parameter(torch.empty(n_cones).uniform_(-0.5, 0.5))
        self.v_proj = nn.Linear(D, n_cones, bias=False)
        self.out_proj = nn.Linear(n_cones, D, bias=False)
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
class DimGateScaleFFN(nn.Module):
    """O(d) params: x * (scale * sigmoid(gate))"""
    def __init__(self, D):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(D))
        self.scale = nn.Parameter(torch.ones(D))
    def forward(self, x):
        return x * (self.scale * torch.sigmoid(self.gate))

class NarrowFFN(nn.Module):
    """O(d²) params: d→d + GELU"""
    def __init__(self, D):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, D), nn.GELU())
    def forward(self, x): return self.net(x)

class DenseFFN(nn.Module):
    """O(d²) params: d→4d→d"""
    def __init__(self, D, mult=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, D*mult), nn.GELU(), nn.Linear(D*mult, D))
    def forward(self, x): return self.net(x)


# ══════════════════════════════════════════════════════════════════════
# BLOCK + LM
# ══════════════════════════════════════════════════════════════════════
class Block(nn.Module):
    def __init__(self, mixer, ffn, D):
        super().__init__()
        self.mixer = mixer; self.ffn = ffn
        self.norm1 = nn.LayerNorm(D); self.norm2 = nn.LayerNorm(D)
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
        self.ln_final = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab_size, bias=False)
    def forward(self, x):
        h = self.pe(self.embed(x))
        for b in self.blocks: h = b(h)
        return self.head(self.ln_final(h))
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════
def make_model(vocab_size, T, d, L, mixer_type, ffn_type, n_heads=4, n_cones=32):
    # Adjust n_heads to be compatible with d
    while d % n_heads != 0 and n_heads > 1:
        n_heads -= 1
    blocks = []
    for _ in range(L):
        if mixer_type == 'attention':
            mixer = CausalSelfAttention(T, d, n_heads)
        else:
            mixer = Cone1DTemporalMixer(T, d, n_cones)
        if ffn_type == 'dense':
            ffn = DenseFFN(d)
        elif ffn_type == 'narrow':
            ffn = NarrowFFN(d)
        elif ffn_type == 'dimgate':
            ffn = DimGateScaleFFN(d)
        else:
            raise ValueError(ffn_type)
        blocks.append(Block(mixer, ffn, d))
    return LM(vocab_size, T, d, blocks)


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
    n_params = model.n_params()
    n_layers = len(model.blocks)
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  params={n_params:,} | layers={n_layers}")
    print(f"{'='*70}")
    
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
    print(f"  BEST_VAL={best_val:.4f} | PPL={ppl:.2f} | "
          f"Conv={'Ep'+str(conv_epoch) if conv_epoch else 'Never'} | {elapsed:.1f}s")
    return dict(label=label, val=best_val, ppl=ppl, params=n_params,
                conv=conv_epoch, time=elapsed, layers=n_layers)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    data, vocab_size = load_data(CFG)
    n_train = int(len(data) * 0.9)
    train_data, val_data = data[:n_train], data[n_train:]
    T = CFG['seq_len']
    
    print(f"\nV107: Iso-Budget Comparisons — Fair Fight")
    print(f"Seq={T} | epochs={CFG['epochs']}")
    print(f"Question: at SAME param budget, does DimGate + more depth win?")
    
    target_A = 158_000   # ~ Transformer d=64, L=3
    target_B = 611_000   # ~ Transformer d=128, L=3
    # Budget B target: ~612K (Transformer d=128, L=3 = 611,712)
    
    experiments = [
        # === BUDGET A: ~158K params ===
        # PROPOSED: DimGate reinvests budget into depth/width
        ("DG_d160_L12 [isoA]",  160, 12, 'cone',      'dimgate'),   # 157K deep
        ("DG_d192_L6  [isoA]",  192,  6, 'cone',      'dimgate'),   # 107K wide (underspend)
        ("DG_d96_L20  [isoA]",   96, 20, 'cone',      'dimgate'),   # 149K very deep
        # References
        ("NarrowFFN_d96_L9   [isoA]",  96, 9, 'cone',  'narrow'),   # 156K
        ("Baseline_d64_L3    [isoA]",  64,  3, 'attention', 'dense'), # 158K
        
        # === BUDGET B: ~612K params ===
        # PROPOSED: DimGate at large scale
        ("DG_d256_L30 [isoB]",  256, 30, 'cone',      'dimgate'),   # 574K deep+wide
        ("DG_d256_L12 [isoB]",  256, 12, 'cone',      'dimgate'),   # 250K (underspend)
        ("DG_d128_L30 [isoB]",  128, 30, 'cone',      'dimgate'),   # 289K (underspend)
        # References
        ("NarrowFFN_d192_L12 [isoB]", 192, 12, 'cone', 'narrow'),   # 628K
        ("Attn+Narrow_d192_L3 [isoB]", 192, 3, 'attention', 'narrow'), # 583K
        ("Baseline_d128_L3   [isoB]", 128,  3, 'attention', 'dense'), # 612K
    ]
    
    results = []
    for label, d, L, mixer, ffn in experiments:
        torch.manual_seed(CFG['seed'])
        model = make_model(vocab_size, T, d, L, mixer, ffn)
        r = run(label, model, train_data, val_data, vocab_size)
        results.append(r)
    
    # ── Summary by budget ─────────────────────────────────────────────
    print(f"\n\n{'='*78}")
    print(f"  V107 SUMMARY — Iso-Budget Comparisons")
    print(f"{'='*78}")
    print(f"  {'Model':<35} {'Params':>8} {'Layers':>6} {'ValLoss':>8} {'PPL':>6} {'Time':>6}")
    print(f"  {'-'*76}")
    
    for budget_label, target in [("Budget A (~150K)", target_A), ("Budget B (~600K)", target_B)]:
        tag = budget_label.split('(')[1].split(')')[0].replace('~','')
        group = [r for r in results if tag.replace('K','').strip() in r['label'].split('_')[-1].split('[')[0] 
                 or f'iso{budget_label[7]}' in r['label']]
        # Simpler: just match by tag
        tag = 'isoA' if 'A' in budget_label else 'isoB'
        group = [r for r in results if tag in r['label']]
        
        if not group: continue
        print(f"  --- {budget_label} ---")
        for r in sorted(group, key=lambda x: x['val']):
            print(f"  {r['label']:<35} {r['params']:>8,} {r['layers']:>6} {r['val']:>8.4f} {r['ppl']:>6.2f} {r['time']:>5.1f}s")
    
    print(f"{'='*78}")
    
    # ── Key analysis ──────────────────────────────────────────────────
    print(f"\n  ISO-PARAM ANALYSIS:")
    for tag, target in [('isoA', target_A), ('isoB', target_B)]:
        group = [r for r in results if tag in r['label']]
        if not group: continue
        baseline = next((r for r in group if 'Baseline' in r['label']), None)
        dimgates = [r for r in group if 'DimGate' in r['label']]
        if baseline and dimgates:
            best_dg = min(dimgates, key=lambda x: x['val'])
            delta = best_dg['val'] - baseline['val']
            print(f"\n  [{tag}] Baseline: val={baseline['val']:.4f} ({baseline['params']:,}p, L={baseline['layers']})")
            print(f"  [{tag}] Best DimGate: val={best_dg['val']:.4f} ({best_dg['params']:,}p, L={best_dg['layers']})")
            print(f"  [{tag}] Delta: {delta:+.4f} ({delta/baseline['val']*100:+.1f}%)")
            if delta < 0:
                print(f"  [{tag}] >>> DimGate WINS at iso-param! Depth compensates for weak FFN.")
            elif delta < baseline['val'] * 0.05:
                print(f"  [{tag}] >>> DimGate COMPETITIVE at iso-param (<5% gap)")
