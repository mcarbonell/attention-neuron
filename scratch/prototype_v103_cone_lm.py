"""
prototype_v103_cone_lm.py
=========================
V103: Cone Neurons for Language Modeling — char-level Shakespeare.

4 configurations compared:
  A) Baseline:  Standard causal self-attention + dense FFN (Transformer)
  B) Cone-Attn: Cone1D positional temporal mixer + dense FFN
  C) Cone-FFN:  Standard causal self-attention + Cone1D FFN
  D) Full-Cone: Cone1D positional temporal mixer + Cone1D FFN

Cone1DTemporalMixer:
  Each "head" learns 3 params: offset (where to look), radius (how wide),
  amplitude (excite/inhibit). Causal: only looks backward.
  weight(t, j) = amplitude * max(0, 1 - |t - j - offset| / radius) for j <= t

Cone1DFFN:
  Each neuron learns 3 params: center (which dimensions to read), radius,
  amplitude. Operates on the DIMENSION axis of the hidden state.
  neuron_k(x) = ReLU(amp_k * sum_i cone(i, center_k, radius_k) * x_i + bias_k)
  Output via dense projection back to d_model.

Hypothesis: Cone-based attention and FFN can match transformer quality
with significantly fewer parameters by exploiting learned locality.
"""

import torch, torch.nn as nn, torch.nn.functional as F
import math, time, os, urllib.request

CFG = dict(
    seq_len=128, batch_size=64,
    d_model=64, n_layers=3, n_heads=4,
    n_cones_attn=32,      # cones per layer for temporal mixing
    n_cones_ffn=256,      # cones per layer for FFN (replaces d_model*4 neurons)
    ffn_mult=4,           # FFN expansion factor for dense baseline
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
# CONE 1D TEMPORAL MIXER — replaces self-attention
# ══════════════════════════════════════════════════════════════════════
class Cone1DTemporalMixer(nn.Module):
    """
    Each cone defines a triangular receptive field over RELATIVE positions.
    
    For each sequence position t, cone c produces:
      w(t, j) = amp_c * max(0, 1 - |t - j - offset_c| / radius_c)
      Only j <= t (causal).
    
    The output per position t = sum_j w(t,j) * V(j) for each cone,
    then projected back to d_model.
    
    Params per cone: 3 (offset, radius, amplitude)
    Total temporal mixing params: n_cones * 3 + n_cones * d_model (V projection)
    """
    def __init__(self, T, D, n_cones):
        super().__init__()
        self.T = T
        self.D = D
        self.n_cones = n_cones
        
        # 3 params per cone
        # offset: how far back to look (positive = look backward)
        # Initialize spread across different offsets for coverage
        self.offset = nn.Parameter(torch.linspace(0, T//2, n_cones))
        # radius: how wide the cone is
        self.radius = nn.Parameter(torch.ones(n_cones) * (T / n_cones * 2))
        # amplitude: excitation (+) or inhibition (-)
        self.amplitude = nn.Parameter(torch.empty(n_cones).uniform_(-0.5, 0.5))
        
        # Value projection: each position -> n_cones features
        self.v_proj = nn.Linear(D, n_cones, bias=False)
        # Output projection: n_cones -> D
        self.out_proj = nn.Linear(n_cones, D, bias=False)
        
        # Pre-compute relative position indices for efficiency
        # positions[t, j] = t - j (how far back j is from t)
        pos = torch.arange(T).unsqueeze(1) - torch.arange(T).unsqueeze(0)  # (T, T)
        self.register_buffer('rel_pos', pos.float())
        
        # Causal mask: only j <= t
        causal = torch.tril(torch.ones(T, T))  # (T, T)
        self.register_buffer('causal_mask', causal)

    def forward(self, x):
        """x: (B, T, D) -> (B, T, D)"""
        B, T, D = x.shape
        
        # Value projection
        V = self.v_proj(x)  # (B, T, n_cones)
        
        # Compute cone weights: (T, T, n_cones)
        # rel_pos[t, j] = t - j; offset[c] = where cone c looks
        # distance = |rel_pos - offset_c|
        radius = F.softplus(self.radius) + 1e-4   # ensure positive, (n_cones,)
        offset = F.softplus(self.offset)            # ensure positive (look backward), (n_cones,)
        
        # (T, T, 1) - (1, 1, n_cones) -> (T, T, n_cones)
        dist = torch.abs(self.rel_pos[:T, :T].unsqueeze(-1) - offset.unsqueeze(0).unsqueeze(0))
        
        # Triangular cone: max(0, 1 - dist/radius)
        weights = F.relu(1.0 - dist / radius.unsqueeze(0).unsqueeze(0))  # (T, T, n_cones)
        
        # Apply amplitude (excitation/inhibition)
        weights = weights * self.amplitude.unsqueeze(0).unsqueeze(0)  # (T, T, n_cones)
        
        # Apply causal mask
        weights = weights * self.causal_mask[:T, :T].unsqueeze(-1)  # (T, T, n_cones)
        
        # Normalize weights per cone per query position (like softmax but simpler)
        # Sum of absolute weights to avoid division by zero with mixed signs
        weight_sum = weights.abs().sum(dim=1, keepdim=True) + 1e-8
        weights = weights / weight_sum
        
        # Apply: for each query position t, weighted sum over keys j
        # weights: (T, T, n_cones) -> need to apply to V: (B, T, n_cones)
        # out[b, t, c] = sum_j weights[t, j, c] * V[b, j, c]
        out = torch.einsum('tjc,bjc->btc', weights, V)  # (B, T, n_cones)
        
        # Project back to d_model
        return self.out_proj(out)  # (B, T, D)


# ══════════════════════════════════════════════════════════════════════
# CONE 1D FFN — replaces dense FFN
# ══════════════════════════════════════════════════════════════════════
class Cone1DFFN(nn.Module):
    """
    Each neuron is a cone over the DIMENSION axis of the hidden state.
    
    neuron_k(x) = amp_k * sum_i max(0, 1 - |i - center_k| / radius_k) * x_i + bias_k
    
    This forces the residual stream to self-organize topologically:
    nearby dimensions must encode related features.
    
    Params per neuron: 4 (center, radius, amplitude, bias)
    Total FFN params: n_neurons * 4 + n_neurons * d_model (output projection)
    
    vs Dense FFN: d_model * ffn_dim * 2 (up + down)
    """
    def __init__(self, D, n_neurons):
        super().__init__()
        self.D = D
        self.n_neurons = n_neurons
        
        # 4 params per neuron
        # center: where in the dimension axis this neuron looks
        self.center = nn.Parameter(torch.linspace(0, D-1, n_neurons))
        # radius: how many dimensions it covers
        self.radius = nn.Parameter(torch.ones(n_neurons) * (D / n_neurons * 2))
        # amplitude: excitation/inhibition
        self.amplitude = nn.Parameter(torch.empty(n_neurons).uniform_(-1.0, 1.0))
        # bias per neuron
        self.bias = nn.Parameter(torch.zeros(n_neurons))
        
        # Output projection: n_neurons -> D (dense, this is where most params are)
        self.out_proj = nn.Linear(n_neurons, D, bias=False)
        
        # Pre-compute dimension positions
        self.register_buffer('dim_positions', torch.arange(D).float())
    
    def forward(self, x):
        """x: (B, T, D) -> (B, T, D)"""
        # Compute cone weights over dimensions: (n_neurons, D)
        center = self.center.unsqueeze(1)  # (n_neurons, 1)
        radius = F.softplus(self.radius).unsqueeze(1) + 1e-4  # (n_neurons, 1)
        amp = self.amplitude.unsqueeze(1)  # (n_neurons, 1)
        
        dim_pos = self.dim_positions.unsqueeze(0)  # (1, D)
        dist = torch.abs(dim_pos - center)  # (n_neurons, D)
        
        # Triangular cone
        cone_weights = F.relu(1.0 - dist / radius) * amp  # (n_neurons, D)
        
        # Apply: each neuron reads a weighted region of the hidden state
        # x: (B, T, D), cone_weights: (n_neurons, D) -> (B, T, n_neurons)
        hidden = F.linear(x, cone_weights, self.bias)  # (B, T, n_neurons)
        hidden = F.relu(hidden)
        
        # Project back to d_model
        return self.out_proj(hidden)  # (B, T, D)


# ══════════════════════════════════════════════════════════════════════
# STANDARD COMPONENTS (for baseline and hybrid configs)
# ══════════════════════════════════════════════════════════════════════
class DenseFFN(nn.Module):
    def __init__(self, D, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, D * mult),
            nn.GELU(),
            nn.Linear(D * mult, D),
        )
    def forward(self, x):
        return self.net(x)


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention."""
    def __init__(self, T, D, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(D, n_heads, batch_first=True)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        self.register_buffer('attn_mask', mask)
    
    def forward(self, x):
        T = x.shape[1]
        out, _ = self.attn(x, x, x, attn_mask=self.attn_mask[:T, :T])
        return out


# ══════════════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK — composable with any mixer + any FFN
# ══════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════
# FULL LM
# ══════════════════════════════════════════════════════════════════════
class ConeLM(nn.Module):
    def __init__(self, vocab_size, T, D, blocks, use_pe=True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, D)
        self.pe = SinCosPE(D) if use_pe else None
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(D, vocab_size, bias=False)
    
    def forward(self, x):
        h = self.embed(x)
        if self.pe: h = self.pe(h)
        for block in self.blocks:
            h = block(h)
        return self.head(h)
    
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ══════════════════════════════════════════════════════════════════════
def make_model(config, vocab_size, T, D, n_layers, n_heads, n_cones_attn, n_cones_ffn, ffn_mult):
    blocks = []
    for _ in range(n_layers):
        if config == 'baseline':
            mixer = CausalSelfAttention(T, D, n_heads)
            ffn = DenseFFN(D, ffn_mult)
        elif config == 'cone_attn':
            mixer = Cone1DTemporalMixer(T, D, n_cones_attn)
            ffn = DenseFFN(D, ffn_mult)
        elif config == 'cone_ffn':
            mixer = CausalSelfAttention(T, D, n_heads)
            ffn = Cone1DFFN(D, n_cones_ffn)
        elif config == 'full_cone':
            mixer = Cone1DTemporalMixer(T, D, n_cones_attn)
            ffn = Cone1DFFN(D, n_cones_ffn)
        else:
            raise ValueError(f"Unknown config: {config}")
        blocks.append(TransformerBlock(mixer, ffn, D))
    
    return ConeLM(vocab_size, T, D, blocks, use_pe=True)


# ══════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_loss(model, data, T, bs, vocab_size, n=50):
    model.eval()
    losses = []
    for _ in range(n):
        x, y = get_batch(data, T, bs)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        losses.append(loss.item())
    return sum(losses) / len(losses)


def run(label, model, train_data, val_data, vocab_size):
    T = CFG['seq_len']; bs = CFG['batch_size']
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  params={model.n_params():,}")
    print(f"{'='*60}")
    
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
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            # Fast feedback: primeros 5 batches de epoch 1
            if ep == 1 and b < 5:
                print(f"  [Ep1 B{b+1}] train={loss.item():.4f}")
        sched.step()
        
        val = eval_loss(model, val_data, T, bs, vocab_size, CFG['val_steps'])
        if val < best_val:
            best_val = val
        if conv_epoch is None and val < 2.0:
            conv_epoch = ep
        print(f"  Ep {ep:02d} | train={ep_loss/CFG['steps_per_epoch']:.4f} | val={val:.4f}")
    
    elapsed = time.time() - t0
    ppl = math.exp(min(best_val, 10))
    print(f"  BEST_VAL={best_val:.4f} | PPL={ppl:.2f} | Conv={'Ep'+str(conv_epoch) if conv_epoch else 'Never'} | {elapsed:.1f}s")
    return dict(label=label, val=best_val, ppl=ppl, params=model.n_params(),
                conv=conv_epoch, time=elapsed)


# ══════════════════════════════════════════════════════════════════════
# CONE DIAGNOSTICS — visualize learned cone positions
# ══════════════════════════════════════════════════════════════════════
def print_cone_diagnostics(model, config_name):
    print(f"\n  --- Cone Diagnostics [{config_name}] ---")
    for i, block in enumerate(model.blocks):
        if hasattr(block.mixer, 'offset'):
            offsets = F.softplus(block.mixer.offset).detach()
            radii = F.softplus(block.mixer.radius).detach()
            amps = block.mixer.amplitude.detach()
            n_excit = (amps > 0).sum().item()
            n_inhib = (amps < 0).sum().item()
            print(f"  Layer {i} Mixer: offsets=[{offsets.min():.1f}, {offsets.max():.1f}] "
                  f"radii=[{radii.min():.1f}, {radii.max():.1f}] "
                  f"excit={n_excit} inhib={n_inhib}")
        if hasattr(block.ffn, 'center'):
            centers = block.ffn.center.detach()
            radii = F.softplus(block.ffn.radius).detach()
            amps = block.ffn.amplitude.detach()
            n_excit = (amps > 0).sum().item()
            n_inhib = (amps < 0).sum().item()
            print(f"  Layer {i} FFN:   centers=[{centers.min():.1f}, {centers.max():.1f}] "
                  f"radii=[{radii.min():.1f}, {radii.max():.1f}] "
                  f"excit={n_excit} inhib={n_inhib}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    data, vocab_size = load_data(CFG)
    n_train = int(len(data) * 0.9)
    train_data, val_data = data[:n_train], data[n_train:]
    T, D, NL = CFG['seq_len'], CFG['d_model'], CFG['n_layers']
    
    print(f"\nV103: Cone Neurons for Language Modeling")
    print(f"Seq={T} | d={D} | layers={NL} | epochs={CFG['epochs']}")
    print(f"Cones attn={CFG['n_cones_attn']} | Cones FFN={CFG['n_cones_ffn']}")
    print(f"Hypothesis: Cone-based layers can match transformer with fewer params")
    
    configs = [
        # PROPOSED FIRST (regla de oro)
        ('D_FullCone       [PROPOSED]', 'full_cone'),
        ('C_ConeFFN        [proposed]', 'cone_ffn'),
        ('B_ConeAttn       [proposed]', 'cone_attn'),
        ('A_Baseline       [control]',  'baseline'),
    ]
    
    results = []
    for label, config in configs:
        torch.manual_seed(CFG['seed'])
        model = make_model(
            config, vocab_size, T, D, NL,
            n_heads=CFG['n_heads'],
            n_cones_attn=CFG['n_cones_attn'],
            n_cones_ffn=CFG['n_cones_ffn'],
            ffn_mult=CFG['ffn_mult'],
        )
        r = run(label, model, train_data, val_data, vocab_size)
        
        # Diagnostics for cone models
        if 'cone' in config.lower():
            print_cone_diagnostics(model, config)
        
        results.append(r)
    
    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  V103 SUMMARY — Cone Neurons for Language Modeling")
    print(f"{'='*70}")
    print(f"  {'Model':<42} {'Params':>8} {'ValLoss':>8} {'PPL':>6} {'Conv':>6} {'Time':>6}")
    print(f"  {'-'*68}")
    for r in sorted(results, key=lambda x: x['val']):
        c = f"Ep{r['conv']}" if r['conv'] else "Never"
        print(f"  {r['label']:<42} {r['params']:>8,} {r['val']:>8.4f} {r['ppl']:>6.2f} {c:>6} {r['time']:>5.1f}s")
    print(f"{'='*70}")
    
    # ── Comparisons ───────────────────────────────────────────────────
    baseline = next(r for r in results if 'Baseline' in r['label'])
    for r in results:
        if r == baseline: continue
        delta = r['val'] - baseline['val']
        param_ratio = r['params'] / baseline['params']
        print(f"\n  {r['label'].split('[')[0].strip()} vs Baseline:")
        print(f"    val_loss: {delta:+.4f} ({'worse' if delta > 0 else 'BETTER'})")
        print(f"    params:   {param_ratio:.2f}x ({r['params']:,} vs {baseline['params']:,})")
        
        # Efficiency: quality per parameter
        if r['val'] < 10:
            pei_r = (1.0 / r['val']) / math.log10(r['params'] + 1)
            pei_b = (1.0 / baseline['val']) / math.log10(baseline['params'] + 1)
            print(f"    PEI:      {pei_r:.4f} vs {pei_b:.4f} (higher=better)")
