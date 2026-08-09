"""
v317 — Prototipo: Conformal Spherical MoLoRA (Proyección Esférica L2 en S^(n-1), Fase 10)
Línea de investigación: Dynamic Low-Rank Adaptations & Conformal Geometry
"""

import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_HEADER = """
======================================================================
[00:00:00] EXECUTION HEADER & TRACEABILITY (v317 - Conformal Spherical MoLoRA)
======================================================================
Hardware: CPU / DirectML PyTorch
Threads: {}
Model File: scratch/prototype_v317_conformal_spherical_molora.py
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


class ConformalSphericalMoLORALinear(nn.Module):
    """
    Conformal Spherical MoLoRA.
    Aplica proyectores L2 esféricos en el cuello de botella de bajo rango r
    y en la salida del adaptador d_out:
      h_k = L2_Norm(A_k · x)
      v_k = L2_Norm(B_k · h_k)
      y = W_0 * x + sum_{k=1}^K g_k(x) * scaling * v_k
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

    def forward(self, x):
        base_out = F.linear(x, self.weight)
        
        # 1. Proyección A + Normalización Esférica L2 en S^(r-1)
        h_raw = torch.einsum('bti,kri->btkr', x, self.lora_A)             # (B, T, K, r)
        h_spherical = F.normalize(h_raw, p=2, dim=-1, eps=1e-8)           # (B, T, K, r)
        
        # 2. Proyección B + Normalización Esférica L2 en S^(d_out-1)
        adapter_raw = torch.einsum('btkr,kor->btko', h_spherical, self.lora_B) # (B, T, K, d_out)
        adapter_outs = F.normalize(adapter_raw, p=2, dim=-1, eps=1e-8) * self.scaling
        
        # 3. Suma Ponderada por Router
        gating_weights = F.softmax(self.router(x), dim=-1)
        dynamic_delta = torch.einsum('btko,btk->bto', adapter_outs, gating_weights)
        
        return base_out + dynamic_delta


class DyRankDynamicGatedLoRALinear(nn.Module):
    """DyRank MoLoRA (v316)"""
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
        return base_out + dynamic_delta


class FastDynamicGatedLoRALinear(nn.Module):
    """Fast MoLoRA Estándar (v311)"""
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
        return base_out + dynamic_delta


class StandardLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
    def forward(self, x): return self.linear(x)


class ResidualBlock(nn.Module):
    def __init__(self, d_model, layer_type="conformal_spherical_molora", rank=16, num_experts=4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if layer_type == "conformal_spherical_molora":
            self.fn = ConformalSphericalMoLORALinear(d_model, d_model, rank=rank, num_experts=num_experts)
        elif layer_type == "dyrank_molora":
            self.fn = DyRankDynamicGatedLoRALinear(d_model, d_model, rank=rank, num_experts=num_experts)
        elif layer_type == "fast_molora":
            self.fn = FastDynamicGatedLoRALinear(d_model, d_model, rank=rank, num_experts=num_experts)
        else:
            self.fn = StandardLinear(d_model, d_model)
            
    def forward(self, x):
        return x + F.silu(self.fn(self.norm(x)))


class StructuredSequenceModel(nn.Module):
    def __init__(self, vocab_size=64, d_model=128, layer_type="conformal_spherical_molora", rank=16, num_experts=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.block1 = ResidualBlock(d_model, layer_type=layer_type, rank=rank, num_experts=num_experts)
        self.block2 = ResidualBlock(d_model, layer_type=layer_type, rank=rank, num_experts=num_experts)
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
    
    print(f"Final Loss: {final_loss:.4f} | Wall Clock: {wall_clock_time:.2f}s | Internal Overhead: {overhead_time:.2f}s | PEI: {pei:.4f}")
    
    return {
        "model_type": model_type,
        "params": num_params,
        "final_loss": final_loss,
        "wall_clock_time": wall_clock_time,
        "eval_time": eval_time_accum,
        "overhead_time": overhead_time,
        "pei": pei
    }


if __name__ == "__main__":
    results = []
    print("[REGLA DE ORO] Ejecutando el CANDIDATO Conformal Spherical MoLoRA (v317) en primer lugar...")
    results.append(run_experiment("conformal_spherical_molora", rank=16, num_experts=4, epochs=10))
    
    print("\nEjecutando baselines de comparación...")
    results.append(run_experiment("dyrank_molora", rank=16, num_experts=4, epochs=10))
    results.append(run_experiment("fast_molora", rank=16, num_experts=4, epochs=10))
    results.append(run_experiment("standard_dense", epochs=10))
    
    print("\n" + "="*80)
    print("RESUMEN COMPARATIVO (v317 Conformal Spherical MoLoRA)")
    print("="*80)
    print(f"{'Modelo':<30} | {'Params':<10} | {'Loss Final':<10} | {'Wall Clock (s)':<15} | {'PEI':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['model_type']:<30} | {r['params']:<10,} | {r['final_loss']:<10.4f} | {r['wall_clock_time']:<15.2f} | {r['pei']:<8.4f}")
    print("="*80)
    
    best_res = min(results, key=lambda x: x["final_loss"])
    
    ledger_entry = {
        "experiment_id": "v317",
        "fecha": "2026-08-09",
        "familia": "low_rank_dinamico_conformal",
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
