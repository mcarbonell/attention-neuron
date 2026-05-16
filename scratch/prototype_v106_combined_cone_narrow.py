"""
prototype_v106_combined_cone_narrow.py
======================================
V106: ConeAttn + NarrowFFN — The Combined Architecture

V103: ConeAttn replaces attention with +4% loss, 24% fewer params
V105: NarrowFFN (d→d) replaces FFN with +1% loss, 11.5× fewer FFN params

Question: Do both savings STACK? Or does combining two weaker components
create a model that's worse than the sum of its parts?

Configs (all at d=128 and d=64):
  A) ConeAttn + NarrowFFN     ← THE PROPOSED ARCHITECTURE
  B) ConeAttn + BottleneckFFN ← Even more compressed
  C) ConeAttn + DenseFFN      ← V103 reference
  D) StdAttn + NarrowFFN      ← V105 reference
  E) StdAttn + DenseFFN       ← Baseline Transformer

If A ≈ E with 60%+ fewer params → we have a publishable architecture.
"""

import torch, torch.nn as nn, torch.nn.functional as F
import math, time, os, urllib.request

CFG = dict(
    seq_len=128, batch_size=64,
    n_layers=3, n_heads=4,
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
# MIXERS
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


class Cone1DTemporalMixer(nn.Module):
    """From V103: relative positional cones with causal masking."""
    def __init__(self, T, D, n_cones):
        super().__init__()
        self.T = T; self.D = D; self.n_cones = n_cones
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
class NarrowFFN(nn.Module):
    """From V105: d→d + GELU. The +1% solution."""
    def __init__(self, D):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, D), nn.GELU())
    def forward(self, x):
        return self.net(x)

class BottleneckFFN(nn.Module):
    """From V105: d→d/4→d. The +2.7% solution."""
    def __init__(self, D, k=None):
        super().__init__()
        if k is None: k = max(D // 4, 16)
        self.net = nn.Sequential(nn.Linear(D, k), nn.GELU(), nn.Linear(k, D))
    def forward(self, x):
        return self.net(x)

class DenseFFN(nn.Module):
    """Standard d→4d→d."""
    def __init__(self, D, mult=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, D*mult), nn.GELU(), nn.Linear(D*mult, D))
    def forward(self, x):
        return self.net(x)


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
        self.head = nn.Linear(D, vocab_size, bias=False)
    def forward(self, x):
        h = self.pe(self.embed(x))
        for b in self.blocks: h = b(h)
        return self.head(h)
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════
def make_model(vocab_size, T, D, n_layers, n_heads, mixer_type, ffn_type, n_cones=32):
    blocks = []
    for _ in range(n_layers):
        if mixer_type == 'attention':
            mixer = CausalSelfAttention(T, D, n_heads)
        elif mixer_type == 'cone':
            mixer = Cone1DTemporalMixer(T, D, n_cones)
        else:
            raise ValueError(mixer_type)

        if ffn_type == 'dense':
            ffn = DenseFFN(D)
        elif ffn_type == 'narrow':
            ffn = NarrowFFN(D)
        elif ffn_type == 'bottleneck':
            ffn = BottleneckFFN(D)
        else:
            raise ValueError(ffn_type)

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

    # Cone diagnostics
    for i, block in enumerate(model.blocks):
        if hasattr(block.mixer, 'offset'):
            r = F.softplus(block.mixer.radius).detach()
            amps = block.mixer.amplitude.detach()
            print(f"  L{i} Mixer: r=[{r.min():.1f},{r.max():.1f}] "
                  f"excit={(amps>0).sum().item()} inhib={(amps<0).sum().item()}")

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
    T, NL = CFG['seq_len'], CFG['n_layers']
    n_cones = CFG['n_cones_attn']

    print(f"\nV106: ConeAttn + NarrowFFN — Combined Architecture")
    print(f"Seq={T} | layers={NL} | epochs={CFG['epochs']}")

    # (label, d_model, mixer, ffn)
    experiments = [
        # PROPOSED FIRST
        ("A_Cone+Narrow_d128   [PROPOSED]",  128, 'cone',      'narrow'),
        ("B_Cone+Bottleneck_d128 [compact]", 128, 'cone',      'bottleneck'),
        ("C_Cone+Narrow_d64    [micro]",      64, 'cone',      'narrow'),
        # References
        ("D_Cone+Dense_d128    [V103 ref]",  128, 'cone',      'dense'),
        ("E_Attn+Narrow_d128   [V105 ref]",  128, 'attention', 'narrow'),
        # Baselines
        ("F_Attn+Dense_d128    [baseline]",  128, 'attention', 'dense'),
        ("G_Attn+Dense_d64     [baseline]",   64, 'attention', 'dense'),
    ]

    results = []
    for label, D, mixer, ffn in experiments:
        torch.manual_seed(CFG['seed'])
        model = make_model(vocab_size, T, D, NL, CFG['n_heads'], mixer, ffn, n_cones)
        r = run(label, model, train_data, val_data, vocab_size)
        results.append(r)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n\n{'='*78}")
    print(f"  V106 SUMMARY — ConeAttn + NarrowFFN Combined")
    print(f"{'='*78}")
    print(f"  {'Model':<40} {'Params':>8} {'ValLoss':>8} {'PPL':>6} {'Conv':>6} {'Time':>6}")
    print(f"  {'-'*76}")

    for d_group in [128, 64]:
        group = [r for r in results if f'd{d_group}' in r['label']]
        if not group: continue
        print(f"  --- d_model={d_group} ---")
        for r in sorted(group, key=lambda x: x['val']):
            c = f"Ep{r['conv']}" if r['conv'] else "Never"
            print(f"  {r['label']:<40} {r['params']:>8,} {r['val']:>8.4f} {r['ppl']:>6.2f} {c:>6} {r['time']:>5.1f}s")

    print(f"{'='*78}")

    # ── Key comparison ────────────────────────────────────────────────
    baseline_128 = next((r for r in results if 'F_Attn+Dense_d128' in r['label']), None)
    proposed = next((r for r in results if 'A_Cone+Narrow_d128' in r['label']), None)

    if baseline_128 and proposed:
        delta = proposed['val'] - baseline_128['val']
        ratio = proposed['params'] / baseline_128['params']
        pei_p = (1/proposed['val']) / math.log10(proposed['params']+1)
        pei_b = (1/baseline_128['val']) / math.log10(baseline_128['params']+1)
        print(f"\n  === THE BIG QUESTION ===")
        print(f"  ConeAttn + NarrowFFN vs Full Transformer (d=128):")
        print(f"    val_loss: {proposed['val']:.4f} vs {baseline_128['val']:.4f} ({delta:+.4f}, {delta/baseline_128['val']*100:+.1f}%)")
        print(f"    params:   {proposed['params']:,} vs {baseline_128['params']:,} ({ratio:.2%} of baseline)")
        print(f"    PEI:      {pei_p:.4f} vs {pei_b:.4f} (higher=better)")
        print(f"    savings:  {(1-ratio)*100:.0f}% fewer parameters")

        if delta / baseline_128['val'] < 0.10:
            print(f"\n    >>> WITHIN 10%: The combined architecture works!")
            print(f"        {(1-ratio)*100:.0f}% parameter reduction for {delta/baseline_128['val']*100:.1f}% quality loss.")
        if pei_p > pei_b:
            print(f"    >>> PEI is HIGHER: more efficient per parameter!")
