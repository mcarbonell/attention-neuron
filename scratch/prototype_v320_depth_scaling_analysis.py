"""
v320 — Prototipo: Análisis de Rango Capa por Capa en 8 Capas (Fase 13)
Línea de investigación: Dynamic Low-Rank Adaptations & Depth Hierarchy
"""

import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_HEADER = """
======================================================================
[00:00:00] EXECUTION HEADER & TRACEABILITY (v320 - 8-Layer Depth Analysis)
======================================================================
Hardware: CPU / DirectML PyTorch
Threads: {}
Model File: scratch/prototype_v320_depth_scaling_analysis.py
Dataset: Structured Associative Pattern Task (N=2000, L=64, V=64)
Architecture Depth: 8 Residual Layers (Layer 1 to Layer 8)
Hyperparameters:
  - Batch Size: 32
  - Seq Length: 64
  - d_model: 128
  - rank r: 16
  - K experts: 4
  - Learning Rate: 1e-3
  - Epochs: 10
======================================================================
""".format(torch.get_num_threads())

print(LOG_HEADER)


# --- Capas ---
class HardBinaryDyRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=16, num_experts=4, alpha=16.0):
        super().__init__()
        self.scaling = alpha / rank
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.lora_A = nn.Parameter(torch.zeros(num_experts, rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(num_experts, out_features, rank))
        for k in range(num_experts):
            nn.init.kaiming_uniform_(self.lora_A[k], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[k])
        self.router = nn.Linear(in_features, num_experts, bias=False)
        self.rank_gate = nn.Linear(in_features, num_experts * rank, bias=False)
        nn.init.constant_(self.rank_gate.weight, 0.01)

    def forward(self, x):
        B_sz, T_sz, _ = x.shape
        base_out = F.linear(x, self.weight)
        m_cont = torch.sigmoid(self.rank_gate(x)).view(B_sz, T_sz, -1, self.lora_A.shape[1])
        m_hard = (m_cont >= 0.5).float()
        m_ste = m_cont + (m_hard - m_cont).detach()
        h = torch.einsum('bti,kri->btkr', x, self.lora_A)
        h_gated = h * m_ste
        adapter_outs = torch.einsum('btkr,kor->btko', h_gated, self.lora_B) * self.scaling
        gating_weights = F.softmax(self.router(x), dim=-1)
        dynamic_delta = torch.einsum('btko,btk->bto', adapter_outs, gating_weights)
        return base_out + dynamic_delta, m_ste


class FastDynamicGatedLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=16, num_experts=4, alpha=16.0):
        super().__init__()
        self.scaling = alpha / rank
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.lora_A = nn.Parameter(torch.zeros(num_experts, rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(num_experts, out_features, rank))
        for k in range(num_experts):
            nn.init.kaiming_uniform_(self.lora_A[k], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[k])
        self.router = nn.Linear(in_features, num_experts, bias=False)

    def forward(self, x):
        base_out = F.linear(x, self.weight)
        gating_weights = F.softmax(self.router(x), dim=-1)
        h = torch.einsum('bti,kri->btkr', x, self.lora_A)
        adapter_outs = torch.einsum('btkr,kor->btko', h, self.lora_B) * self.scaling
        dynamic_delta = torch.einsum('btko,btk->bto', adapter_outs, gating_weights)
        return base_out + dynamic_delta, None


class StandardLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
    def forward(self, x): return self.linear(x), None


class ResidualBlock(nn.Module):
    def __init__(self, d_model, layer_type="hard_binary_dyrank", rank=16, num_experts=4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if layer_type == "hard_binary_dyrank":
            self.fn = HardBinaryDyRankLinear(d_model, d_model, rank=rank, num_experts=num_experts)
        elif layer_type == "fast_molora":
            self.fn = FastDynamicGatedLoRALinear(d_model, d_model, rank=rank, num_experts=num_experts)
        else:
            self.fn = StandardLinear(d_model, d_model)
            
    def forward(self, x):
        out_fn, mask = self.fn(self.norm(x))
        return x + F.silu(out_fn), mask


class Deep8LayerModel(nn.Module):
    """Modelo profundo con 8 capas residuales"""
    def __init__(self, vocab_size=64, d_model=128, layer_type="hard_binary_dyrank", rank=16, num_experts=4, num_layers=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            ResidualBlock(d_model, layer_type=layer_type, rank=rank, num_experts=num_experts)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        masks = []
        for block in self.blocks:
            h, mask = block(h)
            masks.append(mask)
        h = self.norm_out(h)
        logits = self.head(h)
        return logits, masks


def generate_structured_data(num_samples=2000, seq_len=64, vocab_size=64):
    torch.manual_seed(42)
    x = torch.randint(0, vocab_size // 2, (num_samples, seq_len))
    x_prev = torch.roll(x, shifts=1, dims=1)
    x_prev[:, 0] = 0
    y = (x_prev * 3 + x + 7) % vocab_size
    return x, y


def run_experiment(model_type, num_layers=8, rank=16, num_experts=4, epochs=10):
    start_time = time.time()
    x_data, y_data = generate_structured_data()
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = Deep8LayerModel(vocab_size=64, d_model=128, layer_type=model_type, rank=rank, num_experts=num_experts, num_layers=num_layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--- Probando Modelo: {model_type.upper()} ({num_layers} Capas Profundas, Params: {num_params:,}) ---")
    
    final_loss = 0.0
    eval_time_accum = 0.0
    layer_active_ranks = [[] for _ in range(num_layers)]
    
    for epoch in range(epochs):
        model.train()
        for step, (bx, by) in enumerate(loader):
            step_start = time.time()
            optimizer.zero_grad()
            logits, masks = model(bx)
            loss = F.cross_entropy(logits.view(-1, 64), by.view(-1))
            loss.backward()
            optimizer.step()
            
            step_eval_time = time.time() - step_start
            eval_time_accum += step_eval_time
            
            for idx, m in enumerate(masks):
                if m is not None:
                    active_pct = (m >= 0.5).float().mean().item() * 100.0
                    layer_active_ranks[idx].append(active_pct)
                
            # Fast Feedback
            if epoch == 0 and step < 5:
                l1_str = f" | L1 Active: {layer_active_ranks[0][-1]:.1f}% | L8 Active: {layer_active_ranks[-1][-1]:.1f}%" if layer_active_ranks[0] else ""
                print(f"[Fast Feedback] Batch {step+1}/5 - Loss: {loss.item():.4f}{l1_str} - Step Time: {step_eval_time*1000:.2f}ms")
                
            final_loss = loss.item()

    wall_clock_time = time.time() - start_time
    overhead_time = wall_clock_time - eval_time_accum
    pei = (1.0 / (final_loss + 1e-6)) / math.log10(num_params + 1)
    
    layer_means = [float(torch.tensor(layer_active_ranks[i]).mean()) if layer_active_ranks[i] else 100.0 for i in range(num_layers)]
    
    print(f"Final Loss ({num_layers} Capas): {final_loss:.4f} | Wall Clock: {wall_clock_time:.2f}s | PEI: {pei:.4f}")
    if layer_means[0] < 100.0:
        print("Perfil de Rango Activo Capa por Capa (L1 -> L8):")
        for idx, lm in enumerate(layer_means):
            print(f"  - Capa {idx+1}: {lm:.1f}% Active Rank ({100.0-lm:.1f}% Sparsity)")
            
    return {
        "model_type": model_type,
        "num_layers": num_layers,
        "params": num_params,
        "final_loss": final_loss,
        "layer_active_means": layer_means,
        "wall_clock_time": wall_clock_time,
        "eval_time": eval_time_accum,
        "overhead_time": overhead_time,
        "pei": pei
    }


if __name__ == "__main__":
    results = []
    print("[REGLA DE ORO] Ejecutando el CANDIDATO Hard Binary DyRank STE en 8 Capas (v320) en primer lugar...")
    results.append(run_experiment("hard_binary_dyrank", num_layers=8, rank=16, num_experts=4, epochs=10))
    
    print("\nEjecutando baselines de comparación en 8 Capas...")
    results.append(run_experiment("fast_molora", num_layers=8, rank=16, num_experts=4, epochs=10))
    results.append(run_experiment("standard_dense", num_layers=8, epochs=10))
    
    print("\n" + "="*85)
    print("RESUMEN ANÁLISIS DE PROFUNDIDAD EN 8 CAPAS (v320)")
    print("="*85)
    print(f"{'Modelo':<25} | {'Params':<10} | {'Loss Final (8 Capas)':<22} | {'Wall Clock (s)':<12}")
    print("-" * 85)
    for r in results:
        print(f"{r['model_type']:<25} | {r['params']:<10,} | {r['final_loss']:<22.4f} | {r['wall_clock_time']:<12.2f}")
    print("="*85)
    
    best_res = min(results, key=lambda x: x["final_loss"])
    
    ledger_entry = {
        "experiment_id": "v320",
        "fecha": "2026-08-09",
        "familia": "low_rank_dinamico_depth_scaling",
        "dataset": "sintetico_patron_2k",
        "n_eval": best_res["params"],
        "metric_name": "loss_8layers",
        "value": round(best_res["final_loss"], 4),
        "SE": None,
        "params": best_res["params"],
        "nivel_rigor": 1,
        "etiqueta": "ANCLA"
    }
    print("\nLínea para master_ledger.jsonl:")
    print(json.dumps(ledger_entry))
