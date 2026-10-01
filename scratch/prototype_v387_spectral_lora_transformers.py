"""
Prototype v387: Spectral-LoRA Global en Transformers (Modelado de Lenguaje Autorregresivo)
Hypothesis:
  H1: 2D-DCT Low-Frequency Spectral Adapters (dW = D_out^T @ C @ D_in) applied globally
      across both Attention and FFN layers in a frozen Transformer language model match or exceed
      the adaptation perplexity of standard LoRA, while requiring 5x - 18x fewer adapter parameters.
  H2: The low-frequency spectral bias acts as an implicit regularizer that prevents overfitting
      during language adaptation, stabilizing cross-domain style/task transfer.
  H3: Factored spectral projection eliminates optimizer state overhead for low-rank projection
      matrices, yielding higher Parametric Efficiency Index (PEI = Delta_PPL / log10(Params)).

Rigour Level: Level 1 (Exploratory multi-seed Transformer transfer benchmark, 3 seeds per condition)
"""

import sys
import os
import time
import math
import json
import platform
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Logging and Timing Contract
# ---------------------------------------------------------
START_TIME = time.time()

def log(msg):
    elapsed = time.time() - START_TIME
    mins = int(elapsed // 60)
    secs = elapsed % 60
    print(f"[+{mins:02d}:{secs:05.2f}] {msg}", flush=True)

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

# ---------------------------------------------------------
# Orthonormal DCT-II Basis Generator
# ---------------------------------------------------------
def get_dct_basis(N, device="cpu"):
    n = torch.arange(N, device=device).unsqueeze(1)
    k = torch.arange(N, device=device).unsqueeze(0)
    D = torch.cos(math.pi * (n + 0.5) * k / N)
    D[:, 0] *= 1.0 / math.sqrt(N)
    D[:, 1:] *= math.sqrt(2.0 / N)
    return D.T  # (N, N) where D[k, :] is k-th basis vector

def dirichlet_energy_2d(W):
    diff_i = W[1:, :] - W[:-1, :]
    diff_j = W[:, 1:] - W[:, :-1]
    return torch.sum(diff_i ** 2) + torch.sum(diff_j ** 2)

# ---------------------------------------------------------
# Adapters: Spectral-LoRA vs Standard LoRA
# ---------------------------------------------------------
class SpectralAdapter(nn.Module):
    """
    Spectral 2D-DCT Low-Frequency Adapter for 3D Batched Inputs (B, T, d_in).
    dW = D_out[:k_out, :]^T @ C @ D_in[:k_in, :]
    Parameters: exactly k_in * k_out trainable floats!
    """
    def __init__(self, in_features, out_features, k_in=16, k_out=16, alpha=1.0, device="cpu"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k_in = k_in
        self.k_out = k_out
        self.alpha = alpha

        Din = get_dct_basis(in_features, device=device)
        Dout = get_dct_basis(out_features, device=device)

        self.register_buffer("U_in", Din[:k_in, :].T.contiguous())   # (in_features, k_in)
        self.register_buffer("U_out", Dout[:k_out, :].contiguous())  # (k_out, out_features)
        self.C = nn.Parameter(torch.zeros(k_out, k_in, device=device))

    def forward(self, x):
        # x: (B, T, in_features)
        return ((x @ self.U_in) @ self.C.T) @ self.U_out * self.alpha

    def num_params(self):
        return self.k_in * self.k_out


class LoRAAdapter(nn.Module):
    """
    Standard Low-Rank Adapter (LoRA) for 3D Batched Inputs.
    dW = B @ A * (alpha / rank)
    Parameters: rank * (in_features + out_features)
    """
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, device="cpu"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.A = nn.Parameter(torch.randn(rank, in_features, device=device) / math.sqrt(rank))
        self.B = nn.Parameter(torch.zeros(out_features, rank, device=device))

    def forward(self, x):
        return (x @ self.A.T) @ self.B.T * self.scaling

    def num_params(self):
        return self.rank * (self.in_features + self.out_features)

# ---------------------------------------------------------
# Transformer with Pluggable Global Adapters
# ---------------------------------------------------------
class AdaptedLinear(nn.Module):
    """Linear layer with frozen base weight and optional adapter."""
    def __init__(self, base_weight, adapter_type="none", **adapter_kwargs):
        super().__init__()
        out_features, in_features = base_weight.shape
        self.weight = nn.Parameter(base_weight.clone().detach(), requires_grad=False)

        device = base_weight.device
        self.adapter_type = adapter_type
        if adapter_type == "spectral":
            k = adapter_kwargs.get("k", 16)
            self.adapter = SpectralAdapter(in_features, out_features, k_in=k, k_out=k, device=device)
        elif adapter_type == "lora":
            rank = adapter_kwargs.get("rank", 4)
            self.adapter = LoRAAdapter(in_features, out_features, rank=rank, device=device)
        else:
            self.adapter = None

    def forward(self, x):
        out = x @ self.weight.T
        if self.adapter is not None:
            out = out + self.adapter(x)
        return out

    def get_adapter_params(self):
        return self.adapter.num_params() if self.adapter is not None else 0


class AdaptedSelfAttention(nn.Module):
    def __init__(self, base_attn, max_len=128, adapter_type="none", **adapter_kwargs):
        super().__init__()
        d_model = base_attn.d_model
        self.d_model = d_model
        self.n_heads = base_attn.n_heads
        self.head_dim = base_attn.head_dim

        self.q = AdaptedLinear(base_attn.q.weight, adapter_type=adapter_type, **adapter_kwargs)
        self.k = AdaptedLinear(base_attn.k.weight, adapter_type=adapter_type, **adapter_kwargs)
        self.v = AdaptedLinear(base_attn.v.weight, adapter_type=adapter_type, **adapter_kwargs)
        self.o = AdaptedLinear(base_attn.o.weight, adapter_type=adapter_type, **adapter_kwargs)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1),
            persistent=False
        )

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o(y)

    def get_adapter_params(self):
        return self.q.get_adapter_params() + self.k.get_adapter_params() + self.v.get_adapter_params() + self.o.get_adapter_params()


class AdaptedFFN(nn.Module):
    def __init__(self, base_ffn, adapter_type="none", **adapter_kwargs):
        super().__init__()
        self.w_in = AdaptedLinear(base_ffn.w_in.weight, adapter_type=adapter_type, **adapter_kwargs)
        self.w_out = AdaptedLinear(base_ffn.w_out.weight, adapter_type=adapter_type, **adapter_kwargs)

    def forward(self, x):
        return self.w_out(F.gelu(self.w_in(x)))

    def get_adapter_params(self):
        return self.w_in.get_adapter_params() + self.w_out.get_adapter_params()


class AdaptedTransformerBlock(nn.Module):
    def __init__(self, base_block, max_len=128, adapter_type="none", **adapter_kwargs):
        super().__init__()
        self.ln1 = nn.LayerNorm(base_block.ln1.normalized_shape)
        self.ln1.weight = nn.Parameter(base_block.ln1.weight.clone().detach(), requires_grad=False)
        self.ln1.bias = nn.Parameter(base_block.ln1.bias.clone().detach(), requires_grad=False)

        self.attn = AdaptedSelfAttention(base_block.attn, max_len=max_len, adapter_type=adapter_type, **adapter_kwargs)

        self.ln2 = nn.LayerNorm(base_block.ln2.normalized_shape)
        self.ln2.weight = nn.Parameter(base_block.ln2.weight.clone().detach(), requires_grad=False)
        self.ln2.bias = nn.Parameter(base_block.ln2.bias.clone().detach(), requires_grad=False)

        self.ffn = AdaptedFFN(base_block.ffn, adapter_type=adapter_type, **adapter_kwargs)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

    def get_adapter_params(self):
        return self.attn.get_adapter_params() + self.ffn.get_adapter_params()


class AdaptedNanoLM(nn.Module):
    def __init__(self, base_model, adapter_type="none", **adapter_kwargs):
        super().__init__()
        self.vocab_size = base_model.vocab_size
        self.max_len = base_model.max_len

        self.tok_emb = nn.Embedding(self.vocab_size, base_model.tok_emb.embedding_dim)
        self.tok_emb.weight = nn.Parameter(base_model.tok_emb.weight.clone().detach(), requires_grad=False)

        self.pos_emb = nn.Parameter(base_model.pos_emb.clone().detach(), requires_grad=False)

        self.blocks = nn.ModuleList([
            AdaptedTransformerBlock(b, max_len=self.max_len, adapter_type=adapter_type, **adapter_kwargs)
            for b in base_model.blocks
        ])

        self.ln_f = nn.LayerNorm(base_model.ln_f.normalized_shape)
        self.ln_f.weight = nn.Parameter(base_model.ln_f.weight.clone().detach(), requires_grad=False)
        self.ln_f.bias = nn.Parameter(base_model.ln_f.bias.clone().detach(), requires_grad=False)

        self.d_model = base_model.tok_emb.embedding_dim
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def get_adapter_params(self):
        return sum(b.get_adapter_params() for b in self.blocks)

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ---------------------------------------------------------
# Base Model Architecture (Self-Contained)
# ---------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4, max_len=128):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1),
            persistent=False
        )

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o(y)

    def get_dirichlet_energy(self):
        return (dirichlet_energy_2d(self.q.weight) +
                dirichlet_energy_2d(self.k.weight) +
                dirichlet_energy_2d(self.v.weight) +
                dirichlet_energy_2d(self.o.weight))


class TopographicFFN(nn.Module):
    def __init__(self, d_model=128, ffn_dim=256):
        super().__init__()
        self.w_in = nn.Linear(d_model, ffn_dim, bias=False)
        self.w_out = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x):
        return self.w_out(F.gelu(self.w_in(x)))

    def get_dirichlet_energy(self):
        return dirichlet_energy_2d(self.w_in.weight) + dirichlet_energy_2d(self.w_out.weight)


class FullTransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, max_len=128, ffn_dim=256):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = TopographicFFN(d_model, ffn_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class FullTopographicLM(nn.Module):
    def __init__(self, vocab_size=65, d_model=128, n_heads=4, n_layers=2, max_len=128, ffn_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))

        self.blocks = nn.ModuleList([
            FullTransformerBlock(d_model, n_heads, max_len, ffn_dim)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.lm_head.weight = self.tok_emb.weight
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def get_attn_dirichlet_energy(self):
        return sum(block.attn.get_dirichlet_energy() for block in self.blocks)

    def get_ffn_dirichlet_energy(self):
        return sum(block.ffn.get_dirichlet_energy() for block in self.blocks)


# ---------------------------------------------------------
# Dataset Splitter: Domain Shift within Tiny Shakespeare
# ---------------------------------------------------------
class SplitShakespeareDataset:
    """
    Splits Tiny Shakespeare into two distinct linguistic corpora:
      - Base Corpus (first 75% of text): Historical tragedies & monarchic dramas.
      - Target Corpus (last 25% of text): Comedies & romance dialogues (Romeo & Juliet, etc.).
    """
    def __init__(self, data_path="data/tinyshakespeare.txt", seq_len=128):
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        split_idx = int(len(data) * 0.75)

        self.base_data = data[:split_idx]
        target_all = data[split_idx:]

        # Target train vs val
        n_target_train = int(len(target_all) * 0.8)
        self.target_train = target_all[:n_target_train]
        self.target_val = target_all[n_target_train:]
        self.seq_len = seq_len

    def get_batch(self, domain="base", split="train", batch_size=32, device="cpu"):
        if domain == "base":
            d = self.base_data
        else:
            d = self.target_train if split == "train" else self.target_val

        ix = torch.randint(len(d) - self.seq_len, (batch_size,))
        x = torch.stack([d[i:i + self.seq_len] for i in ix]).to(device)
        y = torch.stack([d[i + 1:i + 1 + self.seq_len] for i in ix]).to(device)
        return x, y

# ---------------------------------------------------------
# Evaluation Helpers (SE across Independent Sequences)
# ---------------------------------------------------------
def evaluate_language_model(model, dataset, num_batches=20, batch_size=32, device="cpu"):
    model.eval()
    all_seq_losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = dataset.get_batch(domain="target", split="val", batch_size=batch_size, device=device)
            logits = model(x)
            loss_token = F.cross_entropy(
                logits.view(-1, model.vocab_size), y.view(-1), reduction="none"
            ).view(x.shape[0], x.shape[1])
            seq_loss = loss_token.mean(dim=1)
            all_seq_losses.extend(seq_loss.cpu().tolist())

    seq_losses_arr = np.array(all_seq_losses)
    mean_loss = float(np.mean(seq_losses_arr))
    n_seq = len(seq_losses_arr)
    se_loss = float(np.std(seq_losses_arr) / math.sqrt(n_seq))
    ppl = float(math.exp(mean_loss))
    return mean_loss, se_loss, ppl, n_seq

# ---------------------------------------------------------
# Phase 1: Pretrain Base Models on Base Domain
# ---------------------------------------------------------
def pretrain_base_model(seed, dataset, max_steps=400, lr=2e-3, batch_size=32, eps=2e-3, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = FullTopographicLM(
        vocab_size=dataset.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_len=128,
        ffn_dim=256
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    log(f"--- Pretraining Full Topographic Base Model [Seed {seed}, Steps: {max_steps}, eps: {eps}] ---")
    t0 = time.time()
    for step in range(1, max_steps + 1):
        model.train()
        x, y = dataset.get_batch(domain="base", split="train", batch_size=batch_size, device=device)
        optimizer.zero_grad()
        logits = model(x)
        task_loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))
        reg_loss = eps * (model.get_attn_dirichlet_energy() + model.get_ffn_dirichlet_energy())
        total_loss = task_loss + reg_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 200 == 0 or step == max_steps:
            log(f"  Pretrain Step {step}/{max_steps} | Task Loss: {task_loss.item():.4f}")

    elapsed = time.time() - t0
    log(f"Pretraining Finished in {elapsed:.1f}s.")
    return model

# ---------------------------------------------------------
# Phase 2: Adaptation (Fine-Tuning on Target Domain)
# ---------------------------------------------------------
def adapt_target_model(base_model, cond_cfg, seed, dataset, max_steps=300, lr=2e-3, batch_size=32, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    adapter_type = cond_cfg["adapter_type"]
    adapter_kwargs = cond_cfg.get("kwargs", {})

    model = AdaptedNanoLM(base_model, adapter_type=adapter_type, **adapter_kwargs).to(device)

    adapter_params = model.get_adapter_params()
    trainable_params = model.get_trainable_params()

    log(f"  Adaptation Start [{cond_cfg['name']}, Seed {seed}]: AdapterParams={adapter_params:,}, Trainable={trainable_params:,}")

    # Zero-shot evaluation before training
    val_loss_0, val_se_0, val_ppl_0, _ = evaluate_language_model(model, dataset, num_batches=15, batch_size=batch_size, device=device)
    log(f"  Step 0 (Zero-Shot) | Target Val Loss: {val_loss_0:.4f} | Target Val PPL: {val_ppl_0:5.2f}")

    if adapter_params == 0:
        # Zero-shot baseline, no fine-tuning needed
        return {
            "final_val_loss": val_loss_0,
            "final_val_se": val_se_0,
            "final_val_ppl": val_ppl_0,
            "adapter_params": 0,
            "wall_time": 0.5,
            "step_history": [{"step": 0, "val_loss": val_loss_0, "val_ppl": val_ppl_0}]
        }

    trainable_tensors = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_tensors, lr=lr, weight_decay=1e-4)

    t0 = time.time()
    step_history = []

    for step in range(1, max_steps + 1):
        model.train()
        x, y = dataset.get_batch(domain="target", split="train", batch_size=batch_size, device=device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_tensors, max_norm=1.0)
        optimizer.step()

        # Fast feedback on first 5 steps of Run 1
        if step <= 5 and seed == 42 and "k16" in cond_cfg["name"]:
            log(f"  [Fast Feedback Step {step}/5] AdaptLoss = {loss.item():.4f}")

        if step % 100 == 0 or step == max_steps:
            val_loss, val_se, val_ppl, _ = evaluate_language_model(model, dataset, num_batches=15, batch_size=batch_size, device=device)
            st_s = step / (time.time() - t0)
            eta_s = (max_steps - step) / st_s if st_s > 0 else 0
            log(f"  Step {step:3d}/{max_steps} | Target Val Loss: {val_loss:.4f} (SE: {val_se:.4f}) | Val PPL: {val_ppl:5.2f} | {st_s:.1f} st/s (ETA: {eta_s:.1f}s)")
            step_history.append({"step": step, "val_loss": val_loss, "val_ppl": val_ppl})

    wall_time = time.time() - t0
    final_loss, final_se, final_ppl, _ = evaluate_language_model(model, dataset, num_batches=20, batch_size=batch_size, device=device)

    return {
        "final_val_loss": final_loss,
        "final_val_se": final_se,
        "final_val_ppl": final_ppl,
        "adapter_params": adapter_params,
        "wall_time": wall_time,
        "step_history": step_history
    }

# ---------------------------------------------------------
# Main Suite
# ---------------------------------------------------------
def main():
    log("=========================================================================================")
    log(" EXPERIMENT v387: Spectral-LoRA Global en Transformers (Language Adaptation Benchmark)")
    log("=========================================================================================")
    log(f"Commit: {get_git_commit()} | Python: {platform.python_version()} | PyTorch: {torch.__version__}")
    log(f"Platform: {platform.platform()} | Device: CPU")

    dataset = SplitShakespeareDataset(data_path="data/tinyshakespeare.txt", seq_len=128)
    log(f"Dataset Loaded: Tiny Shakespeare | Vocab Size: {dataset.vocab_size}")
    log(f"  Base Pretraining Domain: {len(dataset.base_data):,} characters (Tragedies / History)")
    log(f"  Target Adaptation Domain: {len(dataset.target_train) + len(dataset.target_val):,} characters (Romance / Comedy)")

    SEEDS = [42, 100, 2026]
    PRETRAIN_STEPS = 400
    ADAPT_STEPS = 300
    BATCH_SIZE = 32
    LR_ADAPT = 2e-3

    # -----------------------------------------------------
    # Conditions definition (Regla de Oro: Primero el Candidato)
    # -----------------------------------------------------
    conditions = [
        # Candidate 1: Spectral-LoRA k=16 (256 params x 12 matrices = 3,072 params)
        {
            "name": "Spectral_k16_FullLM",
            "adapter_type": "spectral",
            "kwargs": {"k": 16},
            "desc": "Spectral-LoRA (k=16, 3,072 adapter params across Attn+FFN)"
        },
        # Candidate 2: Spectral-LoRA k=8 (64 params x 12 matrices = 768 params)
        {
            "name": "Spectral_k8_FullLM",
            "adapter_type": "spectral",
            "kwargs": {"k": 8},
            "desc": "Spectral-LoRA (k=8, 768 adapter params across Attn+FFN)"
        },
        # Candidate 3: Spectral-LoRA k=24 (576 params x 12 matrices = 6,912 params)
        {
            "name": "Spectral_k24_FullLM",
            "adapter_type": "spectral",
            "kwargs": {"k": 24},
            "desc": "Spectral-LoRA (k=24, 6,912 adapter params across Attn+FFN)"
        },
        # Baseline 1: Standard LoRA rank 4 (14,336 params)
        {
            "name": "LoRA_r4_FullLM",
            "adapter_type": "lora",
            "kwargs": {"rank": 4},
            "desc": "Standard LoRA (r=4, 14,336 adapter params across Attn+FFN)"
        },
        # Baseline 2: Standard LoRA rank 2 (7,168 params, matched budget to k=24)
        {
            "name": "LoRA_r2_FullLM",
            "adapter_type": "lora",
            "kwargs": {"rank": 2},
            "desc": "Standard LoRA (r=2, 7,168 adapter params across Attn+FFN)"
        },
        # Baseline 3: Zero-shot unadapted frozen base model (0 params)
        {
            "name": "ZeroShot_FrozenBase",
            "adapter_type": "none",
            "kwargs": {},
            "desc": "Zero-shot Frozen Base Model (0 adapter params)"
        }
    ]

    log(f"Total Conditions: {len(conditions)} | Seeds: {len(SEEDS)}")
    log("Architecture: 2-layer NanoLM (d_model=128, n_heads=4, ffn_dim=256)")
    log("All linear projections adapted: W_q, W_k, W_v, W_o, W_in, W_out (12 matrices total)")

    # -----------------------------------------------------
    # Phase 1: Pretrain Base Models (Cached per seed)
    # -----------------------------------------------------
    log("\n=========================================================================================")
    log(" PHASE 1: PRE-TRAINING BASE FULL TOPOGRAPHIC MODELS (Base Domain)")
    log("=========================================================================================")

    pretrained_bases = {}
    for seed in SEEDS:
        base_model = pretrain_base_model(
            seed=seed,
            dataset=dataset,
            max_steps=PRETRAIN_STEPS,
            lr=2e-3,
            batch_size=BATCH_SIZE,
            eps=2e-3,
            device="cpu"
        )
        pretrained_bases[seed] = base_model

    # -----------------------------------------------------
    # Phase 2: Adaptation Benchmark on Target Domain
    # -----------------------------------------------------
    log("\n=========================================================================================")
    log(" PHASE 2: ADAPTATION BENCHMARK (Target Domain with Frozen Backbone)")
    log("=========================================================================================")

    all_results = []
    total_runs = len(conditions) * len(SEEDS)
    current_run = 0

    for cond in conditions:
        cond_name = cond["name"]
        log(f"\n>>> Running Condition: {cond_name} ({cond['desc']}) <<<")
        seed_runs = []

        for seed in SEEDS:
            current_run += 1
            progress_pct = (current_run / total_runs) * 100.0
            log(f"[Progress: {progress_pct:.1f}% | Run {current_run}/{total_runs}] Condition: {cond_name} | Seed: {seed}")

            base_model = pretrained_bases[seed]
            res = adapt_target_model(
                base_model=base_model,
                cond_cfg=cond,
                seed=seed,
                dataset=dataset,
                max_steps=ADAPT_STEPS,
                lr=LR_ADAPT,
                batch_size=BATCH_SIZE,
                device="cpu"
            )
            res["seed"] = seed
            res["cond_name"] = cond_name
            seed_runs.append(res)

        all_results.append({
            "cond_name": cond_name,
            "cond_cfg": cond,
            "runs": seed_runs
        })

    # -----------------------------------------------------
    # Phase 3: Consolidation & Table
    # -----------------------------------------------------
    log("\n=========================================================================================")
    log(" PHASE 3: CONSOLIDATED RESULTS & PARAMETRIC EFFICIENCY")
    log("=========================================================================================")

    summary_rows = []
    log(f"{'Condition':<26} | {'Adapter Params':<14} | {'Target Val PPL':<15} | {'Delta PPL vs 0-Shot':<20} | {'PEI (Delta/logP)':<16}")
    log("-" * 100)

    # Extract zero-shot baseline PPL for delta calculations
    zeroshot_res = next(c for c in all_results if c["cond_name"] == "ZeroShot_FrozenBase")
    zeroshot_ppl = float(np.mean([r["final_val_ppl"] for r in zeroshot_res["runs"]]))

    for cond_res in all_results:
        cname = cond_res["cond_name"]
        runs = cond_res["runs"]
        adapt_p = runs[0]["adapter_params"]
        ppls = [r["final_val_ppl"] for r in runs]
        losses = [r["final_val_loss"] for r in runs]
        ses = [r["final_val_se"] for r in runs]

        m_ppl = float(np.mean(ppls))
        sd_ppl = float(np.std(ppls))
        m_loss = float(np.mean(losses))
        sd_loss = float(np.std(losses))
        m_se = float(np.mean(ses))

        delta_ppl = zeroshot_ppl - m_ppl  # positive means improvement (reduction in PPL)
        pei = delta_ppl / math.log10(adapt_p + 1) if adapt_p > 0 else 0.0

        summary_rows.append({
            "cond_name": cname,
            "adapter_params": adapt_p,
            "mean_ppl": m_ppl,
            "std_ppl": sd_ppl,
            "mean_loss": m_loss,
            "std_loss": sd_loss,
            "mean_se": m_se,
            "delta_ppl": delta_ppl,
            "pei": pei,
            "runs": runs
        })

        log(f"{cname:<26} | {adapt_p:<14,d} | {m_ppl:5.2f} +/- {sd_ppl:4.2f}  | {delta_ppl:+6.2f} PPL           | {pei:6.3f}")

    # -----------------------------------------------------
    # Phase 4: Artifact Generation (Figures & JSON)
    # -----------------------------------------------------
    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    json_path = "results/raw/v387_spectral_lora_transformers.json"
    raw_payload = {
        "experiment_id": "v387",
        "date_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "commit_hash": get_git_commit(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "device": "cpu",
        "hyperparameters": {
            "pretrain_steps": PRETRAIN_STEPS,
            "adapt_steps": ADAPT_STEPS,
            "batch_size": BATCH_SIZE,
            "lr_adapt": LR_ADAPT,
            "d_model": 128,
            "ffn_dim": 256,
            "n_heads": 4,
            "n_layers": 2,
            "seq_len": 128,
            "seeds": SEEDS
        },
        "summary": [
            {k: v for k, v in row.items() if k != "runs"}
            for row in summary_rows
        ],
        "detailed_results": all_results
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_payload, f, indent=2)
    log(f"\nRaw results saved to {json_path}")

    # Generate Figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Pareto Frontier (Adapter Params vs Target Val PPL)
    params = [max(1, s["adapter_params"]) for s in summary_rows]
    mean_ppls = [s["mean_ppl"] for s in summary_rows]
    std_ppls = [s["std_ppl"] for s in summary_rows]
    names = [s["cond_name"] for s in summary_rows]

    colors = []
    for s in summary_rows:
        if "Spectral" in s["cond_name"]:
            colors.append("#2ecc71")
        elif "LoRA" in s["cond_name"]:
            colors.append("#e74c3c")
        else:
            colors.append("#95a5a6")

    ax1.errorbar(params, mean_ppls, yerr=std_ppls, fmt="o", capsize=4, color="#34495e", zorder=2)
    for p, ppl, name, c in zip(params, mean_ppls, names, colors):
        ax1.scatter([p], [ppl], color=c, s=130, zorder=3, edgecolors="black", linewidth=1.2)
        ax1.annotate(name.replace("_FullLM", ""), (p, ppl), textcoords="offset points",
                     xytext=(0, 10 if "LoRA_r2" not in name else -15), ha="center", fontsize=8, weight="bold")

    ax1.set_xscale("log")
    ax1.set_title("Pareto Frontier: Parameter Count vs Target Val PPL", fontsize=11, weight="bold")
    ax1.set_xlabel("Adapter Parameters across Attn+FFN (log scale)", fontsize=10)
    ax1.set_ylabel("Target Validation Perplexity (lower is better)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Parametric Efficiency Index (PEI = Delta_PPL / log10(P))
    peis = [s["pei"] for s in summary_rows if s["adapter_params"] > 0]
    pei_names = [s["cond_name"].replace("_FullLM", "") for s in summary_rows if s["adapter_params"] > 0]
    pei_colors = [colors[i] for i, s in enumerate(summary_rows) if s["adapter_params"] > 0]

    y_pos = np.arange(len(pei_names))
    ax2.barh(y_pos, peis, color=pei_colors, edgecolor="black", alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(pei_names, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("PEI (Delta PPL / log10(Params))", fontsize=10)
    ax2.set_title("Parametric Efficiency Index (Higher is Better)", fontsize=11, weight="bold")
    ax2.grid(True, axis="x", linestyle="--", alpha=0.5)

    for i, val in enumerate(peis):
        ax2.text(val + 0.05, i, f"{val:.2f}", va="center", fontsize=9, weight="bold")

    plt.tight_layout()
    fig_path = "results/figures/v387_spectral_lora_transformers_curves.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    log(f"Figure saved to {fig_path}")

    log("\n=========================================================================================")
    log(" EXPERIMENT v387 FINISHED SUCCESSFULLY")
    log("=========================================================================================")

if __name__ == "__main__":
    main()
