"""
v316 — Prototipo: DyRank MoLoRA (Asignación Dinámica de Rango por Token, Fase 9)
Línea de investigación: Dynamic Low-Rank Adaptations & Dynamic Sparsity
"""

import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_HEADER = """
======================================================================
[00:00:00] EXECUTION HEADER & TRACEABILITY (v316 - DyRank MoLoRA)
======================================================================
Hardware: CPU / DirectML PyTorch
Threads: {}
Model File: scratch/prototype_v316_dyrank_molora.py
Dataset: Structured Associative Pattern Task (N=2000, L=64, V=64)
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


class DyRankDynamicGatedLoRALinear(nn.Module):
    """
    Capa DyRank MoLoRA (Asignación Dinámica de Rango por Token).
    Aplica una compuerta sigmoidal de rango m_k(x) en (B, T, K, r) sobre los subespacios:
    
    h_k = (A_k · x) ⊙ m_k(x)
    y = W_0 * x + sum_{k=1}^K g_k(x) * (alpha/r) * (B_k * h_k)
    """
    def __init__(self, in_features, out_features, rank=16, num_experts=4, alpha=16.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.num_experts = num_experts
        self.scaling = alpha / rank
        
        # Sustrato lineal base W_0
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # K adaptadores de bajo rango
        self.lora_A = nn.Parameter(torch.zeros(num_experts, rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(num_experts, out_features, rank))
        for k in range(num_experts):
            nn.init.kaiming_uniform_(self.lora_A[k], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[k])
            
        # Router dinámico por token (x -> K logits)
        self.router = nn.Linear(in_features, num_experts, bias=False)
        
        # Compuerta dinámico-estocástica de rango por token (x -> K * r gates)
        self.rank_gate = nn.Linear(in_features, num_experts * rank, bias=False)
        nn.init.zeros_(self.rank_gate.weight)

    def forward(self, x):
        B_sz, T_sz, D_in = x.shape
        base_out = F.linear(x, self.weight)  # (B, T, d_out)
        
        # 1. Mascaramiento dinámico de rango m(x) -> (B, T, K, r)
        rank_mask = torch.sigmoid(self.rank_gate(x)).view(B_sz, T_sz, self.num_experts, self.rank)
        
        # 2. Proyección A con filtrado de rango: h_k = (A_k · x) ⊙ m_k(x)
        h = torch.einsum('bti,kri->btkr', x, self.lora_A)  # (B, T, K, r)
        h_gated = h * rank_mask                            # (B, T, K, r)
        
        # 3. Proyección B
        adapter_outs = torch.einsum('btkr,kor->btko', h_gated, self.lora_B) * self.scaling
        
        # 4. Ruteo Softmax
        gating_weights = F.softmax(self.router(x), dim=-1)
        dynamic_delta = torch.einsum('btko,btk->bto', adapter_outs, gating_weights)
        
        return base_out + dynamic_delta, rank_mask


class FastDynamicGatedLoRALinear(nn.Module):
    """MoLoRA Convencional de Rango Fijo (Fase 3/4)"""
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


class StaticLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=64, alpha=16.0):
        super().__init__()
        self.scaling = alpha / rank
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = F.linear(x, self.weight)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out, None


class StandardLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.linear(x), None


class ResidualBlock(nn.Module):
    def __init__(self, d_model, layer_type="dyrank_molora", rank=16, num_experts=4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if layer_type == "dyrank_molora":
            self.fn = DyRankDynamicGatedLoRALinear(d_model, d_model, rank=rank, num_experts=num_experts)
        elif layer_type == "fast_molora":
            self.fn = FastDynamicGatedLoRALinear(d_model, d_model, rank=rank, num_experts=num_experts)
        elif layer_type == "static_lora":
            self.fn = StaticLoRALinear(d_model, d_model, rank=64)
        else:
            self.fn = StandardLinear(d_model, d_model)
            
    def forward(self, x):
        out_fn, mask = self.fn(self.norm(x))
        return x + F.silu(out_fn), mask


class StructuredSequenceModel(nn.Module):
    def __init__(self, vocab_size=64, d_model=128, layer_type="dyrank_molora", rank=16, num_experts=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.block1 = ResidualBlock(d_model, layer_type=layer_type, rank=rank, num_experts=num_experts)
        self.block2 = ResidualBlock(d_model, layer_type=layer_type, rank=rank, num_experts=num_experts)
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        h, mask1 = self.block1(h)
        h, mask2 = self.block2(h)
        h = self.norm_out(h)
        logits = self.head(h)
        return logits, (mask1, mask2)


def generate_structured_data(num_samples=2000, seq_len=64, vocab_size=64):
    torch.manual_seed(42)
    x = torch.randint(0, vocab_size // 2, (num_samples, seq_len))
    x_prev = torch.roll(x, shifts=1, dims=1)
    x_prev[:, 0] = 0
    y = (x_prev * 3 + x + 7) % vocab_size
    return x, y


def run_experiment(model_type, rank=16, num_experts=4, epochs=10):
    start_time = time.time()
    x_data, y_data = generate_structured_data()
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = StructuredSequenceModel(vocab_size=64, d_model=128, layer_type=model_type, rank=rank, num_experts=num_experts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--- Probando Modelo: {model_type.upper()} (Params: {num_params:,}) ---")
    
    final_loss = 0.0
    eval_time_accum = 0.0
    active_ranks = []
    
    for epoch in range(epochs):
        model.train()
        for step, (bx, by) in enumerate(loader):
            step_start = time.time()
            optimizer.zero_grad()
            logits, (m1, m2) = model(bx)
            loss = F.cross_entropy(logits.view(-1, 64), by.view(-1))
            loss.backward()
            optimizer.step()
            
            step_eval_time = time.time() - step_start
            eval_time_accum += step_eval_time
            
            if m1 is not None:
                active_ratio = (m1 > 0.5).float().mean().item() * 100.0
                active_ranks.append(active_ratio)
                
            # Fast Feedback
            if epoch == 0 and step < 5:
                sparsity_str = f" | Active Rank: {active_ranks[-1]:.1f}%" if active_ranks else ""
                print(f"[Fast Feedback] Batch {step+1}/5 - Loss: {loss.item():.4f}{sparsity_str} - Step Time: {step_eval_time*1000:.2f}ms")
                
            final_loss = loss.item()

    wall_clock_time = time.time() - start_time
    overhead_time = wall_clock_time - eval_time_accum
    pei = (1.0 / (final_loss + 1e-6)) / math.log10(num_params + 1)
    mean_active_rank = float(torch.tensor(active_ranks).mean()) if active_ranks else 100.0
    
    print(f"Final Loss: {final_loss:.4f} | Active Rank %: {mean_active_rank:.1f}% | Wall Clock: {wall_clock_time:.2f}s | PEI: {pei:.4f}")
    
    return {
        "model_type": model_type,
        "params": num_params,
        "final_loss": final_loss,
        "active_rank_pct": mean_active_rank,
        "wall_clock_time": wall_clock_time,
        "eval_time": eval_time_accum,
        "overhead_time": overhead_time,
        "pei": pei
    }


if __name__ == "__main__":
    results = []
    print("[REGLA DE ORO] Ejecutando el CANDIDATO DyRank MoLoRA (v316) en primer lugar...")
    results.append(run_experiment("dyrank_molora", rank=16, num_experts=4, epochs=10))
    
    print("\nEjecutando baselines de comparación...")
    results.append(run_experiment("fast_molora", rank=16, num_experts=4, epochs=10))
    results.append(run_experiment("static_lora", rank=64, epochs=10))
    results.append(run_experiment("standard_dense", epochs=10))
    
    print("\n" + "="*85)
    print("RESUMEN COMPARATIVO (v316 DyRank MoLoRA)")
    print("="*85)
    print(f"{'Modelo':<22} | {'Params':<10} | {'Loss Final':<10} | {'Active Rank (%)':<15} | {'Wall Clock (s)':<12}")
    print("-" * 85)
    for r in results:
        print(f"{r['model_type']:<22} | {r['params']:<10,} | {r['final_loss']:<10.4f} | {r['active_rank_pct']:<15.1f}% | {r['wall_clock_time']:<12.2f}")
    print("="*85)
    
    best_res = min(results, key=lambda x: x["final_loss"])
    
    ledger_entry = {
        "experiment_id": "v316",
        "fecha": "2026-08-09",
        "familia": "low_rank_dinamico_dyrank",
        "dataset": "sintetico_patron_2k",
        "n_eval": best_res["params"],
        "metric_name": "loss",
        "value": round(best_res["final_loss"], 4),
        "SE": None,
        "params": best_res["params"],
        "nivel_rigor": 1,
        "etiqueta": "ANCLA"
    }
    print("\nLínea para master_ledger.jsonl:")
    print(json.dumps(ledger_entry))
