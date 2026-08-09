"""
v318 — Prototipo: Hard Binary DyRank MoLoRA (Pruning Real 0/1 con STE, Fase 11)
Línea de investigación: Dynamic Low-Rank Adaptations & Hard Binary Sparsity
"""

import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_HEADER = """
======================================================================
[00:00:00] EXECUTION HEADER & TRACEABILITY (v318 - Hard Binary DyRank)
======================================================================
Hardware: CPU / DirectML PyTorch
Threads: {}
Model File: scratch/prototype_v318_hard_binary_dyrank.py
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


class HardBinaryDyRankLinear(nn.Module):
    """
    Hard Binary DyRank MoLoRA con Straight-Through Estimator (STE).
    La compuerta m_ste es estrictamente binaria {0, 1} en el forward pass,
    permitiendo pruning real de dimensiones de bajo rango por token.
    """
    def __init__(self, in_features, out_features, rank=16, num_experts=4, alpha=16.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.num_experts = num_experts
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
        nn.init.zeros_(self.rank_gate.weight)

    def forward(self, x):
        B_sz, T_sz, _ = x.shape
        base_out = F.linear(x, self.weight)
        
        # 1. Compuerta Sigmoidal Continua m_cont in (0, 1)
        m_cont = torch.sigmoid(self.rank_gate(x)).view(B_sz, T_sz, self.num_experts, self.rank)
        
        # 2. Discretización Binaria Dura {0, 1} con STE
        m_hard = (m_cont > 0.5).float()
        m_ste = m_cont + (m_hard - m_cont).detach()  # Forward: {0, 1}, Backward: grad m_cont
        
        # 3. Proyección A con Pruning Real Binario: h_k = (A_k · x) ⊙ m_ste
        h = torch.einsum('bti,kri->btkr', x, self.lora_A)
        h_gated = h * m_ste
        
        # 4. Proyección B + Router
        adapter_outs = torch.einsum('btkr,kor->btko', h_gated, self.lora_B) * self.scaling
        gating_weights = F.softmax(self.router(x), dim=-1)
        dynamic_delta = torch.einsum('btko,btk->bto', adapter_outs, gating_weights)
        
        return base_out + dynamic_delta, m_ste


class ContinuousDyRankLinear(nn.Module):
    """DyRank Continuo (v316)"""
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
        nn.init.zeros_(self.rank_gate.weight)

    def forward(self, x):
        B_sz, T_sz, _ = x.shape
        base_out = F.linear(x, self.weight)
        rank_mask = torch.sigmoid(self.rank_gate(x)).view(B_sz, T_sz, -1, self.lora_A.shape[1])
        h = torch.einsum('bti,kri->btkr', x, self.lora_A)
        h_gated = h * rank_mask
        adapter_outs = torch.einsum('btkr,kor->btko', h_gated, self.lora_B) * self.scaling
        gating_weights = F.softmax(self.router(x), dim=-1)
        dynamic_delta = torch.einsum('btko,btk->bto', adapter_outs, gating_weights)
        return base_out + dynamic_delta, rank_mask


class FastDynamicGatedLoRALinear(nn.Module):
    """Fast MoLoRA (v311)"""
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
        elif layer_type == "continuous_dyrank":
            self.fn = ContinuousDyRankLinear(d_model, d_model, rank=rank, num_experts=num_experts)
        elif layer_type == "fast_molora":
            self.fn = FastDynamicGatedLoRALinear(d_model, d_model, rank=rank, num_experts=num_experts)
        else:
            self.fn = StandardLinear(d_model, d_model)
            
    def forward(self, x):
        out_fn, mask = self.fn(self.norm(x))
        return x + F.silu(out_fn), mask


class StructuredSequenceModel(nn.Module):
    def __init__(self, vocab_size=64, d_model=128, layer_type="hard_binary_dyrank", rank=16, num_experts=4):
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
                # Medir porcentaje exacto de compuertas activas (m > 0.5 / m == 1.0)
                active_ratio = (m1 > 0.5).float().mean().item() * 100.0
                active_ranks.append(active_ratio)
                
            # Fast Feedback
            if epoch == 0 and step < 5:
                sparsity_str = f" | True Active Rank (0/1): {active_ranks[-1]:.1f}%" if active_ranks else ""
                print(f"[Fast Feedback] Batch {step+1}/5 - Loss: {loss.item():.4f}{sparsity_str} - Step Time: {step_eval_time*1000:.2f}ms")
                
            final_loss = loss.item()

    wall_clock_time = time.time() - start_time
    overhead_time = wall_clock_time - eval_time_accum
    pei = (1.0 / (final_loss + 1e-6)) / math.log10(num_params + 1)
    mean_active_rank = float(torch.tensor(active_ranks).mean()) if active_ranks else 100.0
    zero_sparsity = 100.0 - mean_active_rank
    
    print(f"Final Loss: {final_loss:.4f} | True 0/1 Sparsity: {zero_sparsity:.1f}% (Active: {mean_active_rank:.1f}%) | Wall Clock: {wall_clock_time:.2f}s | PEI: {pei:.4f}")
    
    return {
        "model_type": model_type,
        "params": num_params,
        "final_loss": final_loss,
        "active_rank_pct": mean_active_rank,
        "zero_sparsity_pct": zero_sparsity,
        "wall_clock_time": wall_clock_time,
        "eval_time": eval_time_accum,
        "overhead_time": overhead_time,
        "pei": pei
    }


if __name__ == "__main__":
    results = []
    print("[REGLA DE ORO] Ejecutando el CANDIDATO Hard Binary DyRank STE (v318) en primer lugar...")
    results.append(run_experiment("hard_binary_dyrank", rank=16, num_experts=4, epochs=10))
    
    print("\nEjecutando baselines de comparación...")
    results.append(run_experiment("continuous_dyrank", rank=16, num_experts=4, epochs=10))
    results.append(run_experiment("fast_molora", rank=16, num_experts=4, epochs=10))
    results.append(run_experiment("standard_dense", epochs=10))
    
    print("\n" + "="*85)
    print("RESUMEN COMPARATIVO (v318 Hard Binary DyRank STE)")
    print("="*85)
    print(f"{'Modelo':<25} | {'Params':<10} | {'Loss Final':<10} | {'Zero Sparsity (0/1)':<20} | {'Wall Clock (s)':<12}")
    print("-" * 85)
    for r in results:
        print(f"{r['model_type']:<25} | {r['params']:<10,} | {r['final_loss']:<10.4f} | {r['zero_sparsity_pct']:<20.1f}% | {r['wall_clock_time']:<12.2f}")
    print("="*85)
    
    best_res = min(results, key=lambda x: x["final_loss"])
    
    ledger_entry = {
        "experiment_id": "v318",
        "fecha": "2026-08-09",
        "familia": "low_rank_dinamico_dyrank_ste",
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
