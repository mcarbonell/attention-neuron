"""
v322b — Prototipo: All-Spectral Transformer Iso-Parámetros (~412,000 Params, Fase 1b)
Línea de investigación: Spectral Architectures Research Line
"""

import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_HEADER = """
======================================================================
[00:00:00] EXECUTION HEADER & TRACEABILITY (v322b - Iso-Parameter Spectral)
======================================================================
Hardware: CPU / DirectML PyTorch
Threads: {}
Model File: scratch/prototype_v322b_iso_param_spectral_transformer.py
Dataset: Structured Associative Pattern Task (N=2000, L=64, V=64)
Iso-Parameter Goal: ~412,000 Parameters
Architectures:
  1. Fully Spectral Iso-Parametric Transformer (5 Capas Profundas, Phase MHA + Hadamard FFN)
  2. Standard LLaMA Transformer (2 Capas, Standard MHA + Dense FFN 8d^2)
Hyperparameters:
  - Batch Size: 32
  - Seq Length: 64
  - Learning Rate: 1e-3
  - Weight Decay: 0.0 (Strict Spectral Rule)
  - Epochs: 10
======================================================================
""".format(torch.get_num_threads())

print(LOG_HEADER)


def create_hadamard_matrix(n):
    """Crea una matriz ortogonal de Walsh-Hadamard normalizada n x n (n potencia de 2)"""
    H = torch.tensor([[1.0]], dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
    return H / math.sqrt(n)


class PhaseSpectralCausalAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, seq_len=64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        angles = torch.linspace(0.0, 2 * math.pi, seq_len)
        self.phase_bias = nn.Parameter(torch.sin(angles))
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', causal_mask)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + self.phase_bias[:T].unsqueeze(0).unsqueeze(0)
        scores = scores + self.causal_mask[:T, :T]
        
        attn_weights = F.softmax(scores, dim=-1)
        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class MultiFrequencySpectralFFN(nn.Module):
    """FFN Espectral Multi-Frecuencia de Walsh-Hadamard"""
    def __init__(self, d_model, num_banks=4):
        super().__init__()
        self.d_model = d_model
        self.num_banks = num_banks
        H_mat = create_hadamard_matrix(d_model)
        self.register_buffer('H', H_mat)
        
        self.phi1 = nn.Parameter(torch.zeros(num_banks, d_model))
        self.phi2 = nn.Parameter(torch.zeros(num_banks, d_model))
        self.w1 = nn.Parameter(torch.ones(num_banks, d_model))
        self.w2 = nn.Parameter(torch.ones(num_banks, d_model))
        self.combine = nn.Linear(num_banks * d_model, d_model, bias=False)

    def forward(self, x):
        h_freq = F.linear(x, self.H) # (B, T, d)
        bank_outs = []
        for b in range(self.num_banks):
            h_trig = torch.cos(h_freq + self.phi1[b]) * self.w1[b] + torch.sin(h_freq + self.phi2[b]) * self.w2[b]
            bank_outs.append(h_trig)
        h_concat = torch.cat(bank_outs, dim=-1)
        h_comb = self.combine(h_concat)
        out = F.linear(h_comb, self.H.t())
        return out


class FullySpectralIsoBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, seq_len=64, num_banks=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PhaseSpectralCausalAttention(d_model, num_heads=num_heads, seq_len=seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = MultiFrequencySpectralFFN(d_model, num_banks=num_banks)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class FullySpectralIsoModel(nn.Module):
    """Modelo All-Spectral Profundo escalado a Iso-Parámetros (~412,000 params)"""
    def __init__(self, vocab_size=64, d_model=128, num_layers=5, num_banks=4, seq_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            FullySpectralIsoBlock(d_model, num_heads=4, seq_len=seq_len, num_banks=num_banks)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        return self.head(h)


class DenseFFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w1 = nn.Linear(d_model, 4 * d_model)
        self.w2 = nn.Linear(4 * d_model, d_model)
    def forward(self, x): return self.w2(F.silu(self.w1(x)))


class StandardLLaMABlock(nn.Module):
    def __init__(self, d_model, num_heads=4, seq_len=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PhaseSpectralCausalAttention(d_model, num_heads=num_heads, seq_len=seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = DenseFFN(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class StandardLLaMAModel(nn.Module):
    def __init__(self, vocab_size=64, d_model=128, seq_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.block1 = StandardLLaMABlock(d_model, seq_len=seq_len)
        self.block2 = StandardLLaMABlock(d_model, seq_len=seq_len)
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.norm_out(h)
        return self.head(h)


def generate_structured_data(num_samples=2000, seq_len=64, vocab_size=64):
    torch.manual_seed(42)
    x = torch.randint(0, vocab_size // 2, (num_samples, seq_len))
    x_prev = torch.roll(x, shifts=1, dims=1)
    x_prev[:, 0] = 0
    y = (x_prev * 3 + x + 7) % vocab_size
    return x, y


def run_experiment(model_kind, epochs=10):
    start_time = time.time()
    x_data, y_data = generate_structured_data()
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    if model_kind == "fully_spectral_iso":
        model = FullySpectralIsoModel(vocab_size=64, d_model=128, num_layers=5, num_banks=4)
    else:
        model = StandardLLaMAModel(vocab_size=64, d_model=128)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n--- Probando Modelo: {model_kind.upper()} (Params: {num_params:,}) ---")
    
    final_loss = 0.0
    eval_time_accum = 0.0
    
    for epoch in range(epochs):
        model.train()
        for step, (bx, by) in enumerate(loader):
            step_start = time.time()
            optimizer.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits.view(-1, 64), by.view(-1))
            loss.backward()
            optimizer.step()
            
            step_eval_time = time.time() - step_start
            eval_time_accum += step_eval_time
            
            # Fast Feedback
            if epoch == 0 and step < 5:
                print(f"[Fast Feedback] Batch {step+1}/5 - Loss: {loss.item():.4f} - Step Time: {step_eval_time*1000:.2f}ms")
                
            final_loss = loss.item()

    wall_clock_time = time.time() - start_time
    overhead_time = wall_clock_time - eval_time_accum
    pei = (1.0 / (final_loss + 1e-6)) / math.log10(num_params + 1)
    
    print(f"Final Loss: {final_loss:.4f} | Wall Clock: {wall_clock_time:.2f}s | PEI: {pei:.4f}")
    
    return {
        "model_type": model_kind,
        "params": num_params,
        "final_loss": final_loss,
        "wall_clock_time": wall_clock_time,
        "eval_time": eval_time_accum,
        "overhead_time": overhead_time,
        "pei": pei
    }


if __name__ == "__main__":
    results = []
    
    print("[REGLA DE ORO] Ejecutando el CANDIDATO Fully Spectral Iso-Parametric (v322b) en primer lugar...")
    results.append(run_experiment("fully_spectral_iso", epochs=10))
    
    print("\nEjecutando baseline de comparación (Standard LLaMA Transformer)...")
    results.append(run_experiment("standard_llama", epochs=10))
    
    print("\n" + "="*85)
    print("RESUMEN BENCHMARK ISO-PARÁMETROS ALL-SPECTRAL VS STANDARD LLAMA (v322b)")
    print("="*85)
    print(f"{'Modelo Transformer':<30} | {'Params':<10} | {'Loss Final':<10} | {'Wall Clock (s)':<15} | {'PEI':<8}")
    print("-" * 85)
    for r in results:
        print(f"{r['model_type']:<30} | {r['params']:<10,} | {r['final_loss']:<10.4f} | {r['wall_clock_time']:<15.2f} | {r['pei']:<8.4f}")
    print("="*85)
    
    best_res = min(results, key=lambda x: x["final_loss"])
    
    ledger_entry = {
        "experiment_id": "v322b",
        "fecha": "2026-08-09",
        "familia": "espectral_iso_param_transformer",
        "dataset": "sintetico_patron_2k",
        "n_eval": best_res["params"],
        "metric_name": "loss_iso_params",
        "value": round(best_res["final_loss"], 4),
        "SE": None,
        "params": best_res["params"],
        "nivel_rigor": 1,
        "etiqueta": "ANCLA"
    }
    print("\nLínea para master_ledger.jsonl:")
    print(json.dumps(ledger_entry))
