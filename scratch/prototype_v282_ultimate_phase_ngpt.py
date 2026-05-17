"""
prototype_v282_ultimate_phase_ngpt.py
=====================================
V282: The Ultimate Phase-nGPT Model

Fusiona lo mejor de los últimos tres avances estructurales:
1. TrueCausalComplexFFT Mixer (V281): Causalidad real con fase compleja, -45% params vs Attention.
2. NarrowFFN (V105): Recombinación lineal d->d que retiene el 99% de la calidad, reduciendo params FFN 11x.
3. nGPT Hypersphere Normalization (V108): Estabilidad extrema sin LayerNorms.

Hiperparámetros calibrados:
Para los modelos basados en nGPT, los pasos en la hiperesfera son pequeños.
Requerimos LR más alto (3e-2) y más épocas (40) frente al baseline clásico (3e-3, 40).
"""

import torch, torch.nn as nn, torch.nn.functional as F
import math, time, os, urllib.request

CFG = dict(
    seq_len=128, batch_size=64,
    d_model=128, n_layers=3, n_heads=4,
    epochs_standard=40, lr_standard=3e-3,
    epochs_ngpt=40, lr_ngpt=3e-2,
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
        self.out = Linear(D, D, bias=False)
        self.n_heads = n_heads
        self.head_dim = D // n_heads
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # In nGPT, Q and K should be normalized to unit sphere per head
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        scale = math.sqrt(dh)
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(self.mask[:T, :T].unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)

class TrueCausalComplexFFTMixer(nn.Module):
    """
    Complex FFT gate + causal enforcement (V281) + Linear projection.
    """
    def __init__(self, T, D, normalized=False):
        super().__init__()
        self.T = T
        self.pad_T = 1
        while self.pad_T < 2*T: self.pad_T *= 2
        self.n_freq = self.pad_T // 2 + 1
        
        # Usaremos phase y amp por cada dimension (más expresividad para d_model grande)
        # O podemos usar uno compartido como en V281 puro. Aquí, para d=128, compartido puede ser débil.
        # En V281 se usó un único filtro para todo el residual stream.
        # Sigamos el V281 puro:
        self.log_amp = nn.Parameter(torch.zeros(self.n_freq))
        self.phase   = nn.Parameter(torch.zeros(self.n_freq))
        
        mask = torch.zeros(self.pad_T)
        mask[:T] = 1.0
        self.register_buffer('causal_mask', mask)
        
        Linear = NormalizedLinear if normalized else nn.Linear
        # The out projection gives the model capability to mix across D
        self.out_proj = Linear(D, D, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        xt = x.permute(0, 2, 1) # (B, D, T)
        pad = torch.zeros(B, D, self.pad_T-T, device=x.device)
        xt_pad = torch.cat([xt, pad], dim=-1)
        X = torch.fft.rfft(xt_pad, dim=-1)

        # Gate computation
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
    def forward(self, x):
        return self.net(x)

# ══════════════════════════════════════════════════════════════════════
# BLOCKS
# ══════════════════════════════════════════════════════════════════════
class StandardBlock(nn.Module):
    def __init__(self, mixer, ffn, D):
        super().__init__()
        self.mixer = mixer; self.ffn = ffn
        self.norm1 = nn.LayerNorm(D); self.norm2 = nn.LayerNorm(D)
    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

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

# ══════════════════════════════════════════════════════════════════════
# LM
# ══════════════════════════════════════════════════════════════════════
class SinCosPE(nn.Module):
    def __init__(self, D, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, D)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000)/D))
        pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.shape[1]]

class LM(nn.Module):
    def __init__(self, vocab_size, T, D, blocks, use_ngpt=False, use_pe=True):
        super().__init__()
        self.use_ngpt = use_ngpt
        self.embed = nn.Embedding(vocab_size, D)
        self.pe = SinCosPE(D) if use_pe else None
        self.blocks = nn.ModuleList(blocks)
        self.ln_final = nn.LayerNorm(D) if not use_ngpt else None
        self.head = nn.Linear(D, vocab_size, bias=False)

    def forward(self, x):
        h = self.embed(x)
        if self.pe: h = self.pe(h)
        if self.use_ngpt: h = norm_sphere(h)
        for b in self.blocks: h = b(h)
        if self.ln_final is not None: h = self.ln_final(h)
        return self.head(h)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

def make_model(vocab_size, T, D, L, n_heads, mixer_type, ffn_type, use_ngpt):
    Block = nGPTBlock if use_ngpt else StandardBlock
    blocks = []
    use_pe = True
    for _ in range(L):
        if mixer_type == 'attention':
            mixer = CausalSelfAttention(T, D, n_heads, normalized=use_ngpt)
        else:
            mixer = TrueCausalComplexFFTMixer(T, D, normalized=use_ngpt)
            use_pe = False # Phase codifica la posición, no necesitamos PE explícito (V281)

        if ffn_type == 'dense':
            ffn = DenseFFN(D, normalized=use_ngpt)
        else:
            ffn = NarrowFFN(D, normalized=use_ngpt)
        blocks.append(Block(mixer, ffn, D))
        
    return LM(vocab_size, T, D, blocks, use_ngpt=use_ngpt, use_pe=use_pe)

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


if __name__ == '__main__':
    data, vocab_size = load_data(CFG)
    n_train = int(len(data) * 0.9)
    train_data, val_data = data[:n_train], data[n_train:]
    T, D, L, H = CFG['seq_len'], CFG['d_model'], CFG['n_layers'], CFG['n_heads']

    print(f"\nV282: The Ultimate Phase-nGPT Model")
    print(f"Seq={T} | d={D} | L={L}")
    print(f"Hypothesis: CausalComplexFFT + NarrowFFN + nGPT gives superior efficiency.")

    # (label, mixer, ffn, use_ngpt, lr, epochs)
    experiments = [
        ("A_Standard_Transformer    [baseline]", 'attention', 'dense', False, CFG['lr_standard'], CFG['epochs_standard']),
        ("B_nGPT_Transformer        [ngpt ref]", 'attention', 'dense', True,  CFG['lr_ngpt'],     CFG['epochs_ngpt']),
        ("C_CausalPhase_nGPT_Dense  [proposed]", 'causalphase','dense',True,  CFG['lr_ngpt'],     CFG['epochs_ngpt']),
        ("D_CausalPhase_nGPT_Narrow [ultimate]", 'causalphase','narrow',True, CFG['lr_ngpt'],     CFG['epochs_ngpt']),
    ]

    results = []
    for label, mixer, ffn, use_ngpt, lr, epochs in experiments:
        torch.manual_seed(CFG['seed'])
        model = make_model(vocab_size, T, D, L, H, mixer, ffn, use_ngpt)
        r = run(label, model, train_data, val_data, vocab_size, lr, epochs)
        results.append(r)

    print(f"\n\n{'='*75}")
    print(f"  V282 SUMMARY — Ultimate Phase-nGPT Benchmark")
    print(f"{'='*75}")
    print(f"  {'Model':<42} {'Params':>8} {'ValLoss':>8} {'PPL':>6} {'Conv':>6} {'Time':>6}")
    print(f"  {'-'*73}")
    for r in sorted(results, key=lambda x: x['val']):
        c = f"Ep{r['conv']}" if r['conv'] else "Never"
        print(f"  {r['label']:<42} {r['params']:>8,} {r['val']:>8.4f} {r['ppl']:>6.2f} {c:>6} {r['time']:>5.1f}s")
    print(f"{'='*75}")

    std = next(r for r in results if 'Standard' in r['label'])
    ult = next(r for r in results if 'ultimate' in r['label'])
    
    print(f"\n  ULTIMATE COMPARISON:")
    print(f"  Standard Transformer: {std['val']:.4f} ({std['params']:,} params)")
    print(f"  CausalPhase+Narrow:   {ult['val']:.4f} ({ult['params']:,} params)")
    param_ratio = ult['params'] / std['params']
    print(f"  -> The Ultimate Model uses {param_ratio:.1%} of the parameters.")
