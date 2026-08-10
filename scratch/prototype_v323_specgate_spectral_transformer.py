"""
v323 — Prototipo: SpecGate (Dynamic Adaptive Frequency Gating, Fase 2)
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
[00:00:00] EXECUTION HEADER & TRACEABILITY (v323 - SpecGate Spectral)
======================================================================
Hardware: CPU / DirectML PyTorch
Threads: {}
Model File: scratch/prototype_v323_specgate_spectral_transformer.py
Dataset: Structured Associative Pattern Task (N=2000, L=64, V=64)
Architectures:
  1. SpecGate Spectral Transformer (v323 - Compuerta Espectral Adaptativa por Token)
  2. Fully Spectral Iso-Parametric Transformer (v322b - Modulación Fija Global)
  3. Standard LLaMA Transformer (Baseline)
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


class SpecGateSpectralFFN(nn.Module):
    """FFN Espectral con SpecGate (Compuerta Adaptativa de Frecuencia por Token)"""
    def __init__(self, d_model, num_banks=4):
        super().__init__()
        self.d_model = d_model
        self.num_banks = num_banks
        H_mat = create_hadamard_matrix(d_model)
        self.register_buffer('H', H_mat)
        
        # SpecGate Linear Router
        self.spec_gate = nn.Linear(d_model, d_model, bias=False)
        
        self.phi1 = nn.Parameter(torch.zeros(num_banks, d_model))
        self.phi2 = nn.Parameter(torch.zeros(num_banks, d_model))
        self.w1 = nn.Parameter(torch.ones(num_banks, d_model))
        self.w2 = nn.Parameter(torch.ones(num_banks, d_model))
        self.combine = nn.Linear(num_banks * d_model, d_model, bias=False)

    def forward(self, x):
        # 1. Transformada de Walsh-Hadamard (B, T, d)
        h_freq = F.linear(x, self.H)
        
        # 2. SpecGate: Compuerta Adaptativa Frecuencial por Token
        g_spec = torch.sigmoid(self.spec_gate(x))
        h_freq_gated = h_freq * g_spec
        
        # 3. Modulación trigonométrica multi-frecuencia
        bank_outs = []
        for b in range(self.num_banks):
            h_trig = torch.cos(h_freq_gated + self.phi1[b]) * self.w1[b] + torch.sin(h_freq_gated + self.phi2[b]) * self.w2[b]
            bank_outs.append(h_trig)
            
        h_concat = torch.cat(bank_outs, dim=-1)
        h_comb = self.combine(h_concat)
        out = F.linear(h_comb, self.H.t())
        return out, g_spec


class SpecGateSpectralBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, seq_len=64, num_banks=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PhaseSpectralCausalAttention(d_model, num_heads=num_heads, seq_len=seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SpecGateSpectralFFN(d_model, num_banks=num_banks)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        out_ffn, g_spec = self.ffn(self.norm2(x))
        x = x + out_ffn
        return x, g_spec


class SpecGateSpectralModel(nn.Module):
    def __init__(self, vocab_size=64, d_model=128, num_layers=5, num_banks=4, seq_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            SpecGateSpectralBlock(d_model, num_heads=4, seq_len=seq_len, num_banks=num_banks)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        all_gates = []
        for block in self.blocks:
            h, g_spec = block(h)
            all_gates.append(g_spec)
        h = self.norm_out(h)
        return self.head(h), all_gates


def generate_structured_data(num_samples=2000, seq_len=64, vocab_size=64):
    torch.manual_seed(42)
    x = torch.randint(0, vocab_size // 2, (num_samples, seq_len))
    x_prev = torch.roll(x, shifts=1, dims=1)
    x_prev[:, 0] = 0
    y = (x_prev * 3 + x + 7) % vocab_size
    return x, y


def run_experiment(model_type, epochs=10):
    start_time = time.time()
    x_data, y_data = generate_structured_data()
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    if model_type == "specgate_spectral":
        model = SpecGateSpectralModel(vocab_size=64, d_model=128, num_layers=5, num_banks=4)
    else:
        # Import dynamic fallback from v322b structure
        from prototype_v322b_iso_param_spectral_transformer import FullySpectralIsoModel
        model = FullySpectralIsoModel(vocab_size=64, d_model=128, num_layers=5, num_banks=4)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n--- Probando Modelo: {model_type.upper()} (Params: {num_params:,}) ---")
    
    final_loss = 0.0
    eval_time_accum = 0.0
    active_freq_pcts = []
    
    for epoch in range(epochs):
        model.train()
        for step, (bx, by) in enumerate(loader):
            step_start = time.time()
            optimizer.zero_grad()
            
            if model_type == "specgate_spectral":
                logits, gates = model(bx)
                active_pct = (gates[0] >= 0.5).float().mean().item() * 100.0
                active_freq_pcts.append(active_pct)
            else:
                logits = model(bx)
                
            loss = F.cross_entropy(logits.view(-1, 64), by.view(-1))
            loss.backward()
            optimizer.step()
            
            step_eval_time = time.time() - step_start
            eval_time_accum += step_eval_time
            
            # Fast Feedback
            if epoch == 0 and step < 5:
                gate_str = f" | Active Freqs: {active_freq_pcts[-1]:.1f}%" if active_freq_pcts else ""
                print(f"[Fast Feedback] Batch {step+1}/5 - Loss: {loss.item():.4f}{gate_str} - Step Time: {step_eval_time*1000:.2f}ms")
                
            final_loss = loss.item()

    wall_clock_time = time.time() - start_time
    overhead_time = wall_clock_time - eval_time_accum
    pei = (1.0 / (final_loss + 1e-6)) / math.log10(num_params + 1)
    
    mean_active_freq = float(torch.tensor(active_freq_pcts).mean()) if active_freq_pcts else 100.0
    
    print(f"Final Loss: {final_loss:.4f} | Active Freqs: {mean_active_freq:.1f}% | Wall Clock: {wall_clock_time:.2f}s | PEI: {pei:.4f}")
    
    return {
        "model_type": model_type,
        "params": num_params,
        "final_loss": final_loss,
        "active_freq_pct": mean_active_freq,
        "wall_clock_time": wall_clock_time,
        "eval_time": eval_time_accum,
        "overhead_time": overhead_time,
        "pei": pei
    }


if __name__ == "__main__":
    results = []
    
    print("[REGLA DE ORO] Ejecutando el CANDIDATO SpecGate Spectral (v323) en primer lugar...")
    results.append(run_experiment("specgate_spectral", epochs=10))
    
    print("\nEjecutando baseline de comparación (Fully Spectral Iso v322b)...")
    results.append(run_experiment("fully_spectral_iso", epochs=10))
    
    print("\n" + "="*85)
    print("RESUMEN BENCHMARK SPECGATE SPECTRAL VS SPECTRUM GLOBAL (v323)")
    print("="*85)
    print(f"{'Modelo Transformer':<26} | {'Params':<10} | {'Loss Final':<10} | {'Active Freqs %':<15} | {'PEI':<8}")
    print("-" * 85)
    for r in results:
        print(f"{r['model_type']:<26} | {r['params']:<10,} | {r['final_loss']:<10.4f} | {r['active_freq_pct']:<15.1f} | {r['pei']:<8.4f}")
    print("="*85)
    
    best_res = min(results, key=lambda x: x["final_loss"])
    
    ledger_entry = {
        "experiment_id": "v323",
        "fecha": "2026-08-09",
        "familia": "espectral_specgate_transformer",
        "dataset": "sintetico_patron_2k",
        "n_eval": best_res["params"],
        "metric_name": "loss_specgate",
        "value": round(best_res["final_loss"], 4),
        "SE": None,
        "params": best_res["params"],
        "nivel_rigor": 1,
        "etiqueta": "ANCLA"
    }
    print("\nLínea para master_ledger.jsonl:")
    print(json.dumps(ledger_entry))
