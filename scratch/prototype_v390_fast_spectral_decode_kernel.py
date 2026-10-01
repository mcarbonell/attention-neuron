"""
Prototype v390: Fast Inference Decompression via Block-DCT Decode Kernel
Hypothesis:
  H1 (Physical Storage Compression): A native binary serialization format (.specq)
      encoding radial 2D-DCT frequency bands (8-bit, 4-bit, 2-bit, 0-bit) achieves
      a true ~19.0x physical file size reduction on disk compared to standard PyTorch FP32
      checkpoints (.pt) for both 6-layer and 12-layer Transformers.
  H2 (Ultra-Fast Decompression Throughput): The Block-DCT Decode Kernel reconstructs dense
      FP32 projection matrices from packed bitstreams at >200 MB/s uncompressed throughput on CPU,
      unpacking the entire 12-layer model (1.6M parameters across 72 matrices) in under 35 milliseconds.
  H3 (Generation Latency Parity): Once decompressed at startup (AOT Decode), autoregressive token
      generation runs at 100% native dense GEMM speed with zero throughput degradation, while
      Layer-wise JIT Streaming enables execution with a 19x lower static parameter memory footprint.

Rigour Level: Level 1 (Systems & Algorithmic Decompression Benchmark)
"""

import sys
import os
import time
import math
import struct
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
    return D.T

# ---------------------------------------------------------
# Topographic Metrics
# ---------------------------------------------------------
def dirichlet_energy_2d(W):
    diff_i = W[1:, :] - W[:-1, :]
    diff_j = W[:, 1:] - W[:, :-1]
    return torch.sum(diff_i ** 2) + torch.sum(diff_j ** 2)

def adjacent_cosine_similarity_2d(W):
    r1, r2 = W[:-1, :], W[1:, :]
    cos_rows = F.cosine_similarity(r1, r2, dim=-1).mean().item()
    c1, c2 = W[:, :-1], W[:, 1:]
    cos_cols = F.cosine_similarity(c1, c2, dim=0).mean().item()
    return 0.5 * (cos_rows + cos_cols)

# ---------------------------------------------------------
# Transformer Architecture
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


class ScalableTransformerLM(nn.Module):
    def __init__(self, vocab_size=65, d_model=128, n_heads=4, n_layers=6, max_len=128, ffn_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))

        self.blocks = nn.ModuleList([
            FullTransformerBlock(d_model, n_heads, max_len, ffn_dim)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
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

    def generate(self, prompt_tokens, max_new_tokens=64, temperature=0.8, top_k=40):
        self.eval()
        curr_tokens = prompt_tokens.clone()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = curr_tokens[:, -self.max_len:]
                logits = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                curr_tokens = torch.cat((curr_tokens, idx_next), dim=1)
        return curr_tokens

# ---------------------------------------------------------
# Dataset Loader
# ---------------------------------------------------------
class TinyShakespeareDataset:
    def __init__(self, data_path="data/tinyshakespeare.txt", seq_len=128):
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n_train = int(len(data) * 0.9)
        self.train_data = data[:n_train]
        self.val_data = data[n_train:]
        self.seq_len = seq_len

    def get_batch(self, split="train", batch_size=32, device="cpu"):
        d = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(d) - self.seq_len, (batch_size,))
        x = torch.stack([d[i:i + self.seq_len] for i in ix]).to(device)
        y = torch.stack([d[i + 1:i + 1 + self.seq_len] for i in ix]).to(device)
        return x, y

    def encode(self, s):
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long).unsqueeze(0)

    def decode(self, tokens):
        return "".join([self.itos[i] for i in tokens[0].cpu().tolist()])

def evaluate_language_model(model, dataset, num_batches=20, batch_size=32, device="cpu"):
    model.eval()
    all_seq_losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = dataset.get_batch(split="val", batch_size=batch_size, device=device)
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
# Binary Bitstream Format (.specq) Encoder & Decoder
# ---------------------------------------------------------
def encode_matrix_to_bitstream(W, D_out, D_in):
    """
    Transforms W to 2D-DCT and packs into 3 radial frequency bands:
      Band 0 (rho <= 0.15): 8-bit (1 byte/coeff)
      Band 1 (0.15 < rho <= 0.35): 4-bit (2 coeffs/byte)
      Band 2 (0.35 < rho <= 0.60): 2-bit (4 coeffs/byte)
      Band 3 (rho > 0.60): 0-bit (omitted, exact 0 bytes!)
    """
    M, N = W.shape
    u = torch.arange(M, device=W.device, dtype=W.dtype).unsqueeze(1) / M
    v = torch.arange(N, device=W.device, dtype=W.dtype).unsqueeze(0) / N
    rho = torch.sqrt(u**2 + v**2) / math.sqrt(2.0)

    m0 = (rho <= 0.15)
    m1 = (rho > 0.15) & (rho <= 0.35)
    m2 = (rho > 0.35) & (rho <= 0.60)

    W_dct = D_out @ W @ D_in.T

    # Band 0: 8-bit
    x0 = W_dct[m0]
    s0 = float(x0.abs().max() + 1e-8)
    q0 = torch.clamp(torch.round((x0 / s0) * 127.0), -127, 127).to(torch.int16)
    b0 = (q0 + 128).to(torch.uint8).cpu().numpy()

    # Band 1: 4-bit packed
    x1 = W_dct[m1]
    s1 = float(x1.abs().max() + 1e-8)
    q1 = torch.clamp(torch.round((x1 / s1) * 7.0), -7, 7).to(torch.int16)
    v1 = (q1 + 7).to(torch.uint8).cpu().numpy()
    if len(v1) % 2 != 0:
        v1 = np.pad(v1, (0, 1))
    b1 = ((v1[0::2] << 4) | (v1[1::2] & 0x0F)).astype(np.uint8)

    # Band 2: 2-bit packed
    x2 = W_dct[m2]
    s2 = float(x2.abs().max() + 1e-8)
    q2 = torch.clamp(torch.round((x2 / s2) * 1.0), -1, 1).to(torch.int16)
    v2 = (q2 + 1).to(torch.uint8).cpu().numpy()
    pad_len = (4 - (len(v2) % 4)) % 4
    if pad_len > 0:
        v2 = np.pad(v2, (0, pad_len))
    b2 = ((v2[0::4] << 6) | (v2[1::4] << 4) | (v2[2::4] << 2) | (v2[3::4] & 0x03)).astype(np.uint8)

    return {
        "shape": (M, N),
        "scales": (s0, s1, s2),
        "n_coeffs": (int(m0.sum()), int(m1.sum()), int(m2.sum())),
        "b0": b0.tobytes(),
        "b1": b1.tobytes(),
        "b2": b2.tobytes()
    }


def decode_matrix_from_bitstream(record, D_out, D_in):
    """
    Block-DCT Decode Kernel:
    1. Fast unpack of 8-bit, 4-bit and 2-bit arrays
    2. De-quantization using per-band floating scales
    3. Scatter into sparse 2D-DCT buffer (Band 3 initialized to 0)
    4. Inverse 2D-DCT reconstruction: D_out.T @ W_dct @ D_in
    """
    M, N = record["shape"]
    u = torch.arange(M).unsqueeze(1) / M
    v = torch.arange(N).unsqueeze(0) / N
    rho = torch.sqrt(u**2 + v**2) / math.sqrt(2.0)

    m0 = (rho <= 0.15)
    m1 = (rho > 0.15) & (rho <= 0.35)
    m2 = (rho > 0.35) & (rho <= 0.60)

    n0, n1, n2 = record["n_coeffs"]
    s0, s1, s2 = record["scales"]

    # 1. Band 0: 8-bit
    b0_arr = np.frombuffer(record["b0"], dtype=np.uint8)
    q0 = (torch.from_numpy(b0_arr.astype(np.int16)) - 128).float()
    v0 = q0 * (s0 / 127.0)

    # 2. Band 1: 4-bit
    b1_arr = np.frombuffer(record["b1"], dtype=np.uint8)
    h1 = b1_arr >> 4
    l1 = b1_arr & 0x0F
    raw1 = np.empty(len(b1_arr) * 2, dtype=np.uint8)
    raw1[0::2] = h1
    raw1[1::2] = l1
    q1 = (torch.from_numpy(raw1[:n1].astype(np.int16)) - 7).float()
    v1 = q1 * (s1 / 7.0)

    # 3. Band 2: 2-bit
    b2_arr = np.frombuffer(record["b2"], dtype=np.uint8)
    raw2 = np.empty(len(b2_arr) * 4, dtype=np.uint8)
    raw2[0::4] = (b2_arr >> 6) & 0x03
    raw2[1::4] = (b2_arr >> 4) & 0x03
    raw2[2::4] = (b2_arr >> 2) & 0x03
    raw2[3::4] = b2_arr & 0x03
    q2 = (torch.from_numpy(raw2[:n2].astype(np.int16)) - 1).float()
    v2 = q2 * s2

    # 4. Scatter into DCT
    W_dct = torch.zeros((M, N), dtype=torch.float32)
    W_dct[m0] = v0
    W_dct[m1] = v1
    W_dct[m2] = v2

    # 5. Inverse 2D-DCT
    return D_out.T @ W_dct @ D_in


def save_specq_checkpoint(model, filepath, D_128, D_256):
    """
    Serializes a Topographic LM into native .specq binary format.
    Stores metadata + embedding weights + packed 1.68 bpp bitstreams for all linear projections.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    target_projections = [".q.weight", ".k.weight", ".v.weight", ".o.weight", ".w_in.weight", ".w_out.weight"]

    records = {}
    for name, param in model.named_parameters():
        if any(proj in name for proj in target_projections):
            W = param.data
            if W.shape == (128, 128):
                rec = encode_matrix_to_bitstream(W, D_128, D_128)
            elif W.shape == (256, 128):
                rec = encode_matrix_to_bitstream(W, D_256, D_128)
            elif W.shape == (128, 256):
                rec = encode_matrix_to_bitstream(W, D_128, D_256)
            else:
                raise ValueError(f"Unexpected shape: {W.shape}")
            records[name] = rec

    # Extract non-linear weights (embeddings, LayerNorms)
    aux_params = {
        name: param.data.half().cpu()
        for name, param in model.named_parameters()
        if not any(proj in name for proj in target_projections)
    }

    payload = {
        "format": "SPECQ_V1",
        "n_layers": model.n_layers,
        "d_model": model.d_model,
        "vocab_size": model.vocab_size,
        "records": records,
        "aux_params": aux_params
    }

    torch.save(payload, filepath)
    return os.path.getsize(filepath)


def load_specq_checkpoint_aot(filepath, dataset, D_128, D_256, device="cpu"):
    """
    Loads .specq binary file and runs the Block-DCT Decode Kernel Ahead-of-Time (AOT).
    Returns fully instantiated ScalableTransformerLM ready for native GEMM inference.
    """
    t0 = time.time()
    payload = torch.load(filepath, map_location="cpu", weights_only=False)
    io_time = time.time() - t0

    t_decode0 = time.time()
    n_layers = payload["n_layers"]
    model = ScalableTransformerLM(
        vocab_size=payload["vocab_size"],
        d_model=payload["d_model"],
        n_heads=4,
        n_layers=n_layers,
        max_len=128,
        ffn_dim=256
    ).to(device)

    # 1. Load aux weights
    for name, tensor in payload["aux_params"].items():
        dict(model.named_parameters())[name].data.copy_(tensor.float().to(device))

    # 2. Decode all linear matrices via Block-DCT Decode Kernel
    records = payload["records"]
    for name, rec in records.items():
        if rec["shape"] == (128, 128):
            d_out, d_in = D_128, D_128
        elif rec["shape"] == (256, 128):
            d_out, d_in = D_256, D_128
        elif rec["shape"] == (128, 256):
            d_out, d_in = D_128, D_256
        else:
            raise ValueError(f"Unexpected shape: {rec['shape']}")

        W_decoded = decode_matrix_from_bitstream(rec, d_out, d_in)
        dict(model.named_parameters())[name].data.copy_(W_decoded.to(device))

    decode_time = time.time() - t_decode0
    total_load_time = time.time() - t0

    return model, {
        "io_time": io_time,
        "decode_time": decode_time,
        "total_load_time": total_load_time,
        "n_matrices": len(records)
    }

# ---------------------------------------------------------
# Training Routine
# ---------------------------------------------------------
def train_topographic_model(depth, dataset, max_steps=350, lr=2e-3, batch_size=32, device="cpu"):
    torch.manual_seed(42)
    np.random.seed(42)

    model = ScalableTransformerLM(
        vocab_size=dataset.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=depth,
        max_len=128,
        ffn_dim=256
    ).to(device)

    eps = 2e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_params = sum(p.numel() for p in model.parameters())

    log(f"  Training Topographic Transformer L={depth} ({total_params:,} params) for {max_steps} steps...")
    t0 = time.time()
    for step in range(1, max_steps + 1):
        model.train()
        x, y = dataset.get_batch(split="train", batch_size=batch_size, device=device)
        optimizer.zero_grad()
        logits = model(x)
        task_loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))
        reg_loss = eps * (model.get_attn_dirichlet_energy() + model.get_ffn_dirichlet_energy())
        (task_loss + reg_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step <= 5 and depth == 6:
            log(f"  [Fast Feedback Step {step}/5] TaskLoss = {task_loss.item():.4f} | RegLoss = {float(reg_loss):.4f}")

    wall_time = time.time() - t0
    log(f"  Training L={depth} completed in {wall_time:.1f}s.")
    return model

# ---------------------------------------------------------
# Main Benchmark Suite
# ---------------------------------------------------------
def main():
    log("=========================================================================================")
    log(" EXPERIMENT v390: Fast Inference Decompression via Block-DCT Decode Kernel")
    log("=========================================================================================")
    log(f"Commit: {get_git_commit()} | Python: {platform.python_version()} | PyTorch: {torch.__version__}")
    log(f"Platform: {platform.platform()} | Device: CPU")

    dataset_path = "data/tinyshakespeare.txt"
    if not os.path.exists(dataset_path):
        log(f"ERROR: Dataset not found at {dataset_path}!")
        sys.exit(1)

    dataset = TinyShakespeareDataset(data_path=dataset_path, seq_len=128)
    log(f"Dataset Loaded: Tiny Shakespeare | Vocab Size: {dataset.vocab_size}")

    D_128 = get_dct_basis(128, device="cpu")
    D_256 = get_dct_basis(256, device="cpu")

    TEST_DEPTHS = [6, 12]
    bench_results = {}

    for L in TEST_DEPTHS:
        log(f"\n" + "=" * 90)
        log(f" EVALUATING ARCHITECTURE DEPTH L = {L} ({L*6} LINEAR PROJECTIONS)")
        log("=" * 90)

        # 1. Train or Instantiate Topographic Model
        model_trained = train_topographic_model(L, dataset, max_steps=350, lr=2e-3, batch_size=32)

        # Evaluate FP32 Validation PPL
        val_loss, val_se, val_ppl, _ = evaluate_language_model(model_trained, dataset, num_batches=20, batch_size=32)
        log(f"  Trained Model FP32 Baseline: Val Loss = {val_loss:.4f} (SE: {val_se:.4f}) | Val PPL = {val_ppl:.2f}")

        # 2. File Serialization Benchmark (.pt vs .specq)
        pt_path = f"results/raw/model_L{L}_fp32.pt"
        specq_path = f"results/raw/model_L{L}_jpeg1.68b.specq"

        torch.save(model_trained.state_dict(), pt_path)
        pt_bytes = os.path.getsize(pt_path)

        specq_bytes = save_specq_checkpoint(model_trained, specq_path, D_128, D_256)
        comp_ratio = pt_bytes / specq_bytes

        log(f"  Physical File Size on Disk:")
        log(f"    - Standard PyTorch FP32 (.pt)    : {pt_bytes/1024:.1f} KB ({pt_bytes/(1024*1024):.2f} MB)")
        log(f"    - Packed Spectral JPEG (.specq)  : {specq_bytes/1024:.1f} KB ({specq_bytes/(1024*1024):.2f} MB)")
        log(f"    - True Disk Footprint Compression: {comp_ratio:.2f}x ({100*(1 - specq_bytes/pt_bytes):.1f}% disk space saved)")

        # 3. Block-DCT Decode Kernel Benchmark
        log(f"  Benchmarking Block-DCT Decode Kernel...")
        model_decoded, decode_stats = load_specq_checkpoint_aot(specq_path, dataset, D_128, D_256)

        n_matrices = decode_stats["n_matrices"]
        decode_ms = decode_stats["decode_time"] * 1000.0
        us_per_matrix = (decode_stats["decode_time"] / n_matrices) * 1e6

        # Calculate uncompressed linear weight throughput
        linear_params = sum(p.numel() for n, p in model_decoded.named_parameters() if any(
            proj in n for proj in [".q.weight", ".k.weight", ".v.weight", ".o.weight", ".w_in.weight", ".w_out.weight"]
        ))
        uncompressed_mb = (linear_params * 4.0) / (1024.0 * 1024.0)
        throughput_mbs = uncompressed_mb / decode_stats["decode_time"]

        log(f"    - Total Matrices Decoded         : {n_matrices} linear projection matrices")
        log(f"    - Total Kernel Decode Time       : {decode_ms:.2f} ms")
        log(f"    - Latency per Matrix             : {us_per_matrix:.2f} microseconds/matrix")
        log(f"    - Uncompressed Weight Throughput : {throughput_mbs:.2f} MB/s uncompressed weights")

        # 4. Decoded Model Perplexity Validation
        q_loss, q_se, q_ppl, _ = evaluate_language_model(model_decoded, dataset, num_batches=20, batch_size=32)
        delta_ppl = q_ppl - val_ppl
        log(f"  Decoded Model Quality Check:")
        log(f"    - Decoded Val Loss               : {q_loss:.4f} (SE: {q_se:.4f})")
        log(f"    - Decoded Val PPL                : {q_ppl:.2f} (Delta vs FP32: {delta_ppl:+.2f} PPL)")

        # 5. Autoregressive Generation Latency Benchmark
        prompt_text = "ROMEO:\nIf I profane with my unworthiest hand\nThis holy shrine"
        prompt_tok = dataset.encode(prompt_text)

        log(f"  Benchmarking Autoregressive Generation Speed (64 tokens, temp=0.8)...")

        # Mode A: Dense FP32
        t0 = time.time()
        out_fp32 = model_trained.generate(prompt_tok, max_new_tokens=64, temperature=0.8, top_k=40)
        gen_time_fp32 = time.time() - t0
        tok_s_fp32 = 64.0 / gen_time_fp32
        ms_tok_fp32 = (gen_time_fp32 / 64.0) * 1000.0

        # Mode B: Spectral AOT Decoded FP32
        t0 = time.time()
        out_spec = model_decoded.generate(prompt_tok, max_new_tokens=64, temperature=0.8, top_k=40)
        gen_time_spec = time.time() - t0
        tok_s_spec = 64.0 / gen_time_spec
        ms_tok_spec = (gen_time_spec / 64.0) * 1000.0

        log(f"    [Dense FP32 Baseline     ] Gen Time: {gen_time_fp32:.2f}s | Speed: {tok_s_fp32:.1f} tok/s ({ms_tok_fp32:.1f} ms/tok)")
        log(f"    [Spectral AOT Decoded    ] Gen Time: {gen_time_spec:.2f}s | Speed: {tok_s_spec:.1f} tok/s ({ms_tok_spec:.1f} ms/tok)")
        log(f"    [Speed Retention Ratio   ] {(tok_s_spec / tok_s_fp32)*100:.1f}% of native FP32 generation throughput")

        sample_output = dataset.decode(out_spec)
        log(f"\n  --- Sample Generated Text from 1.68 bpp Model (L={L}) ---\n{sample_output[:250]}...\n  --------------------------------------------------------\n")

        bench_results[f"L{L}"] = {
            "depth": L,
            "linear_params": linear_params,
            "total_params": sum(p.numel() for p in model_trained.parameters()),
            "pt_bytes": pt_bytes,
            "specq_bytes": specq_bytes,
            "comp_ratio": comp_ratio,
            "decode_ms": decode_ms,
            "us_per_matrix": us_per_matrix,
            "throughput_mbs": throughput_mbs,
            "fp32_ppl": val_ppl,
            "decoded_ppl": q_ppl,
            "delta_ppl": delta_ppl,
            "tok_s_fp32": tok_s_fp32,
            "tok_s_spec": tok_s_spec,
            "ms_tok_spec": ms_tok_spec,
            "sample_output": sample_output[:300]
        }

    # ---------------------------------------------------------
    # Visualization & Figures
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Panel 1: Physical File Size on Disk (KB)
    ax1 = axes[0, 0]
    depths = [6, 12]
    pt_kbs = [bench_results[f"L{L}"]["pt_bytes"] / 1024.0 for L in depths]
    specq_kbs = [bench_results[f"L{L}"]["specq_bytes"] / 1024.0 for L in depths]

    x = np.arange(len(depths))
    w = 0.35
    b1 = ax1.bar(x - w/2, pt_kbs, w, label='PyTorch FP32 (.pt)', color='#d62728', alpha=0.85)
    b2 = ax1.bar(x + w/2, specq_kbs, w, label='Packed Spectral JPEG (.specq)', color='#1f77b4', alpha=0.85)

    for bar, kb in zip(b1, pt_kbs):
        ax1.text(bar.get_x() + bar.get_width()/2, kb + 50, f"{kb:.1f} KB\n({kb/1024:.2f} MB)",
                 ha='center', va='bottom', fontsize=8.5, fontweight="bold")
    for bar, kb, L in zip(b2, specq_kbs, depths):
        ratio = bench_results[f"L{L}"]["comp_ratio"]
        ax1.text(bar.get_x() + bar.get_width()/2, kb + 50, f"{kb:.1f} KB\n({ratio:.1f}x)",
                 ha='center', va='bottom', fontsize=8.5, fontweight="bold", color='#1f77b4')

    ax1.set_ylabel("File Size on Disk (KB)", fontsize=11, fontweight="bold")
    ax1.set_title("Physical Storage Footprint: .pt vs .specq", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"L={L} Layers" for L in depths])
    ax1.set_ylim(0, max(pt_kbs) * 1.25)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax1.legend(fontsize=10)

    # Panel 2: Decompression Latency & Throughput
    ax2 = axes[0, 1]
    decode_times = [bench_results[f"L{L}"]["decode_ms"] for L in depths]
    throughputs = [bench_results[f"L{L}"]["throughput_mbs"] for L in depths]

    ax2_r = ax2.twinx()
    p1 = ax2.bar(x, decode_times, width=0.4, color='#2ca02c', alpha=0.85, label='Decode Latency (ms)')
    p2 = ax2_r.plot(x, throughputs, 'o-', color='#ff7f0e', lw=2.5, markersize=8, label='Throughput (MB/s)')

    for bar, ms in zip(p1, decode_times):
        ax2.text(bar.get_x() + bar.get_width()/2, ms + 1.0, f"{ms:.2f} ms",
                 ha='center', va='bottom', fontsize=9, fontweight="bold")
    for xi, tp in zip(x, throughputs):
        ax2_r.text(xi, tp + 5, f"{tp:.1f} MB/s", ha='center', va='bottom', fontsize=9, fontweight="bold", color='#ff7f0e')

    ax2.set_ylabel("Decompression Time (Milliseconds)", fontsize=11, fontweight="bold", color='#2ca02c')
    ax2_r.set_ylabel("Uncompressed Throughput (MB/s)", fontsize=11, fontweight="bold", color='#ff7f0e')
    ax2.set_title("Block-DCT Decode Kernel Latency & Bandwidth", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"L={L} ({L*6} matrices)" for L in depths])
    ax2.set_ylim(0, max(decode_times) * 1.3)
    ax2_r.set_ylim(0, max(throughputs) * 1.3)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Panel 3: Generation Throughput (Tokens per Second)
    ax3 = axes[1, 0]
    tok_s_fp = [bench_results[f"L{L}"]["tok_s_fp32"] for L in depths]
    tok_s_sp = [bench_results[f"L{L}"]["tok_s_spec"] for L in depths]

    b3 = ax3.bar(x - w/2, tok_s_fp, w, label='Dense FP32 Baseline', color='#7f7f7f', alpha=0.85)
    b4 = ax3.bar(x + w/2, tok_s_sp, w, label='Spectral AOT Decoded', color='#1f77b4', alpha=0.85)

    for bar, sp in zip(b3, tok_s_fp):
        ax3.text(bar.get_x() + bar.get_width()/2, sp + 2.0, f"{sp:.1f} t/s",
                 ha='center', va='bottom', fontsize=9, fontweight="bold")
    for bar, sp, L in zip(b4, tok_s_sp, depths):
        ret = (sp / bench_results[f"L{L}"]["tok_s_fp32"]) * 100
        ax3.text(bar.get_x() + bar.get_width()/2, sp + 2.0, f"{sp:.1f} t/s\n({ret:.0f}%)",
                 ha='center', va='bottom', fontsize=8.5, fontweight="bold", color='#1f77b4')

    ax3.set_ylabel("Generation Speed (Tokens / Second)", fontsize=11, fontweight="bold")
    ax3.set_title("Inference Token Generation Throughput", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"L={L} Layers" for L in depths])
    ax3.set_ylim(0, max(tok_s_fp) * 1.25)
    ax3.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax3.legend(fontsize=10)

    # Panel 4: Perplexity Retention vs Compression
    ax4 = axes[1, 1]
    fp_ppl = [bench_results[f"L{L}"]["fp32_ppl"] for L in depths]
    sp_ppl = [bench_results[f"L{L}"]["decoded_ppl"] for L in depths]

    b5 = ax4.bar(x - w/2, fp_ppl, w, label='FP32 Original PPL', color='#2ca02c', alpha=0.85)
    b6 = ax4.bar(x + w/2, sp_ppl, w, label='Decoded 1.68 bpp PPL', color='#1f77b4', alpha=0.85)

    for bar, ppl in zip(b5, fp_ppl):
        ax4.text(bar.get_x() + bar.get_width()/2, ppl + 0.2, f"{ppl:.2f}",
                 ha='center', va='bottom', fontsize=9, fontweight="bold")
    for bar, ppl, L in zip(b6, sp_ppl, depths):
        delta = bench_results[f"L{L}"]["delta_ppl"]
        ax4.text(bar.get_x() + bar.get_width()/2, ppl + 0.2, f"{ppl:.2f}\n({delta:+.2f})",
                 ha='center', va='bottom', fontsize=8.5, fontweight="bold", color='#1f77b4')

    ax4.set_ylabel("Validation Perplexity (Tiny Shakespeare)", fontsize=11, fontweight="bold")
    ax4.set_title("Perplexity Preservation Under Native Binary Decode", fontsize=12, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"L={L} Layers" for L in depths])
    ax4.set_ylim(0, max(sp_ppl) * 1.3)
    ax4.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax4.legend(fontsize=10)

    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    fig_path = "results/figures/v390_fast_spectral_decode_kernel.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    log(f"Figure saved to: {fig_path}")

    # Save JSON Raw Results
    os.makedirs("results/raw", exist_ok=True)
    raw_path = "results/raw/v390_fast_spectral_decode_kernel.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(bench_results, f, indent=2)
    log(f"Raw results saved to: {raw_path}")
    log(f"Experiment v390 execution completed successfully in {time.time() - START_TIME:.2f}s total.")


if __name__ == "__main__":
    main()
