"""
v321 — Prototipo: Capas Densas FFN vs Capas Espectrales (Walsh-Hadamard & Phase FFN, Fase 14)
Línea de investigación: Spectral Networks & Parametric Efficiency
"""

import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_HEADER = """
======================================================================
[00:00:00] EXECUTION HEADER & TRACEABILITY (v321 - Dense vs Spectral FFN)
======================================================================
Hardware: CPU / DirectML PyTorch
Threads: {}
Model File: scratch/prototype_v321_dense_vs_spectral_ffn.py
Dataset: Structured Associative Pattern Task (N=2000, L=64, V=64)
Comparison:
  1. Dense FFN (Standard 8d^2 params)
  2. Spectral Hadamard FFN (Walsh-Hadamard + Diagonal, O(d) params, 500x Compression)
  3. Spectral Phase FFN (Walsh-Hadamard + Trig Phase, O(d) params, 250x Compression)
  4. Hybrid Spectral FFN (Walsh-Hadamard + d x d Proj, O(d^2) params, 8x Compression)
Hyperparameters:
  - Batch Size: 32
  - Seq Length: 64
  - d_model: 128
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


class DenseFFN(nn.Module):
    """Capa Densa Tradicional FFN (8 d^2 params)"""
    def __init__(self, d_model):
        super().__init__()
        self.w1 = nn.Linear(d_model, 4 * d_model)
        self.w2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


class SpectralHadamardFFN(nn.Module):
    """Capa Espectral Walsh-Hadamard Pura (O(d) params, 500x compresión)"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        H_mat = create_hadamard_matrix(d_model)
        self.register_buffer('H', H_mat)
        self.w_spect = nn.Parameter(torch.ones(d_model))
        self.b_spect = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # 1. Proyección rápida al dominio espectral de Hadamard (vectorizada)
        h_freq = F.linear(x, self.H)
        # 2. Modulación de frecuencias no lineal
        h_mod = F.silu(h_freq * self.w_spect + self.b_spect)
        # 3. Transformada Inversa de Hadamard H^T
        out = F.linear(h_mod, self.H.t())
        return out


class SpectralPhaseFFN(nn.Module):
    """Capa Espectral de Fase Trigonométrica (O(d) params, 250x compresión)"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        H_mat = create_hadamard_matrix(d_model)
        self.register_buffer('H', H_mat)
        self.phi1 = nn.Parameter(torch.zeros(d_model))
        self.phi2 = nn.Parameter(torch.zeros(d_model))
        self.w1 = nn.Parameter(torch.ones(d_model))
        self.w2 = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        h_freq = F.linear(x, self.H)
        # Modulación de fase angular acotada en S^1
        h_trig = torch.cos(h_freq + self.phi1) * self.w1 + torch.sin(h_freq + self.phi2) * self.w2
        out = F.linear(h_trig, self.H.t())
        return out


class HybridSpectralFFN(nn.Module):
    """Capa Híbrida Espectral-Densa (O(d^2) params, 8x compresión)"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        H_mat = create_hadamard_matrix(d_model)
        self.register_buffer('H', H_mat)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        h_freq = F.linear(x, self.H)
        h_mod = F.silu(self.proj(h_freq))
        out = F.linear(h_mod, self.H.t())
        return out


class ResidualBlock(nn.Module):
    def __init__(self, d_model, ffn_type="dense_ffn"):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if ffn_type == "dense_ffn":
            self.fn = DenseFFN(d_model)
        elif ffn_type == "spectral_hadamard_ffn":
            self.fn = SpectralHadamardFFN(d_model)
        elif ffn_type == "spectral_phase_ffn":
            self.fn = SpectralPhaseFFN(d_model)
        elif ffn_type == "hybrid_spectral_ffn":
            self.fn = HybridSpectralFFN(d_model)
            
    def forward(self, x):
        return x + self.fn(self.norm(x))


class SequenceModel(nn.Module):
    def __init__(self, vocab_size=64, d_model=128, ffn_type="dense_ffn"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.block1 = ResidualBlock(d_model, ffn_type=ffn_type)
        self.block2 = ResidualBlock(d_model, ffn_type=ffn_type)
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


def run_experiment(ffn_type, epochs=10):
    start_time = time.time()
    x_data, y_data = generate_structured_data()
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = SequenceModel(vocab_size=64, d_model=128, ffn_type=ffn_type)
    # Regla de Oro: weight_decay = 0.0 para redes espectrales
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--- Probando FFN: {ffn_type.upper()} (Params: {num_params:,}) ---")
    
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
        "ffn_type": ffn_type,
        "params": num_params,
        "final_loss": final_loss,
        "wall_clock_time": wall_clock_time,
        "eval_time": eval_time_accum,
        "overhead_time": overhead_time,
        "pei": pei
    }


if __name__ == "__main__":
    results = []
    
    print("[REGLA DE ORO] Ejecutando las Capas Espectrales CANDIDATAS en primer lugar...")
    results.append(run_experiment("spectral_hadamard_ffn", epochs=10))
    results.append(run_experiment("spectral_phase_ffn", epochs=10))
    results.append(run_experiment("hybrid_spectral_ffn", epochs=10))
    
    print("\nEjecutando baseline de comparación (Capa Densa Tradicional FFN)...")
    results.append(run_experiment("dense_ffn", epochs=10))
    
    print("\n" + "="*85)
    print("RESUMEN BENCHMARK CAPAS DENSAS VS CAPAS ESPECTRALES (v321)")
    print("="*85)
    print(f"{'Modelo FFN':<26} | {'Params':<10} | {'Loss Final':<10} | {'Wall Clock (s)':<15} | {'PEI':<8}")
    print("-" * 85)
    for r in results:
        print(f"{r['ffn_type']:<26} | {r['params']:<10,} | {r['final_loss']:<10.4f} | {r['wall_clock_time']:<15.2f} | {r['pei']:<8.4f}")
    print("="*85)
    
    # Modelo con mayor PEI
    best_pei = max(results, key=lambda x: x["pei"])
    best_loss = min(results, key=lambda x: x["final_loss"])
    
    print(f"\n-> Mayor Eficiencia Paramétrica (PEI): {best_pei['ffn_type']} (PEI: {best_pei['pei']:.4f})")
    print(f"-> Menor Loss Absoluta: {best_loss['ffn_type']} (Loss: {best_loss['final_loss']:.4f})")
    
    ledger_entry = {
        "experiment_id": "v321",
        "fecha": "2026-08-09",
        "familia": "espectral_vs_densa_ffn",
        "dataset": "sintetico_patron_2k",
        "n_eval": best_pei["params"],
        "metric_name": "pei",
        "value": round(best_pei["pei"], 4),
        "SE": None,
        "params": best_pei["params"],
        "nivel_rigor": 1,
        "etiqueta": "ANCLA"
    }
    print("\nLínea para master_ledger.jsonl:")
    print(json.dumps(ledger_entry))
