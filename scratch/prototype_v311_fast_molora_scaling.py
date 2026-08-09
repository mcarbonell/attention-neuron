"""
v311 — Prototipo: Fast MoLoRA & Scaling Sweep
Línea de investigación: Dynamic Low-Rank Adaptations
"""

import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_HEADER = """
======================================================================
[00:00:00] EXECUTION HEADER & TRACEABILITY (v311 - Fast MoLoRA)
======================================================================
Hardware: CPU / DirectML PyTorch
Threads: {}
Model File: scratch/prototype_v311_fast_molora_scaling.py
Dataset: Structured Associative Pattern Task (N=1000, L=64, V=64)
Hyperparameters:
  - Batch Size: 32
  - Seq Length: 64
  - d_model: 128
  - Iso-budget Product (K * r): 64
  - Swept K values: [2, 4, 8, 16]
  - Learning Rate: 1e-3
  - Epochs: 10
======================================================================
""".format(torch.get_num_threads())

print(LOG_HEADER)


class FastDynamicGatedLoRALinear(nn.Module):
    """
    Capa MoLoRA / Dynamic Gated LoRA Ultra-rápida optimizada con torch.einsum.
    Evita desdoblamientos 5D y broadcasting lento en CPU.
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
            
        # Router dinámico ultra-ligero por token
        self.router = nn.Linear(in_features, num_experts, bias=False)

    def forward(self, x):
        # x shape: (B, T, d_in)
        base_out = F.linear(x, self.weight)  # (B, T, d_out)
        
        # Pesos de ruteo por token -> (B, T, K)
        gating_weights = F.softmax(self.router(x), dim=-1)  # (B, T, K)
        
        # Operaciones tensoriales optimizadas con einsum:
        # 1. Proyección A: (B, T, d_in) x (K, r, d_in) -> (B, T, K, r)
        h = torch.einsum('bti,kri->btkr', x, self.lora_A)
        
        # 2. Proyección B: (B, T, K, r) x (K, d_out, r) -> (B, T, K, d_out)
        adapter_outs = torch.einsum('btkr,kor->btko', h, self.lora_B) * self.scaling
        
        # 3. Suma ponderada por token: (B, T, K, d_out) x (B, T, K) -> (B, T, d_out)
        dynamic_delta = torch.einsum('btko,btk->bto', adapter_outs, gating_weights)
        
        return base_out + dynamic_delta


class StaticLoRALinear(nn.Module):
    """LoRA Estándar Monolítico"""
    def __init__(self, in_features, out_features, rank=64, alpha=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = F.linear(x, self.weight)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out


class StandardLinear(nn.Module):
    """Capa lineal densa estándar"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.linear(x)


class ResidualBlock(nn.Module):
    def __init__(self, d_model, layer_type="fast_molora", rank=16, num_experts=4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if layer_type == "fast_molora":
            self.fn = FastDynamicGatedLoRALinear(d_model, d_model, rank=rank, num_experts=num_experts)
        elif layer_type == "static_lora":
            self.fn = StaticLoRALinear(d_model, d_model, rank=64)
        else:
            self.fn = StandardLinear(d_model, d_model)
            
    def forward(self, x):
        return x + F.silu(self.fn(self.norm(x)))


class StructuredSequenceModel(nn.Module):
    def __init__(self, vocab_size=64, d_model=128, layer_type="fast_molora", rank=16, num_experts=4):
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


def generate_structured_data(num_samples=1000, seq_len=64, vocab_size=64):
    """Genera tarea con patrón asociativo estructurado"""
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
    label = f"{model_type}_K{num_experts}_r{rank}" if model_type == "fast_molora" else model_type
    print(f"\n--- Probando Configuración: {label.upper()} (Params: {num_params:,}) ---")
    
    global_step = 0
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
            global_step += 1
            
            # Fast Feedback
            if epoch == 0 and step < 5:
                print(f"[Fast Feedback] Batch {step+1}/5 - Loss: {loss.item():.4f} - Step Time: {step_eval_time*1000:.2f}ms")
                
            final_loss = loss.item()

    wall_clock_time = time.time() - start_time
    overhead_time = wall_clock_time - eval_time_accum
    pei = (1.0 / (final_loss + 1e-6)) / math.log10(num_params + 1)
    
    print(f"Final Loss: {final_loss:.4f} | Wall Clock: {wall_clock_time:.2f}s | Internal Overhead: {overhead_time:.2f}s | PEI: {pei:.4f}")
    
    return {
        "model_type": label,
        "params": num_params,
        "final_loss": final_loss,
        "wall_clock_time": wall_clock_time,
        "eval_time": eval_time_accum,
        "overhead_time": overhead_time,
        "pei": pei
    }


if __name__ == "__main__":
    results = []
    
    print("[REGLA DE ORO] Ejecutando el CANDIDATO optimizado con einsum en primer lugar...")
    # Barrido K in [2, 4, 8, 16] con K * r = 64
    for K, r in [(2, 32), (4, 16), (8, 8), (16, 4)]:
        res = run_experiment("fast_molora", rank=r, num_experts=K, epochs=10)
        results.append(res)
        
    print("\nEjecutando baselines de comparación...")
    res_static = run_experiment("static_lora", rank=64, epochs=10)
    results.append(res_static)
    
    res_dense = run_experiment("standard_dense", epochs=10)
    results.append(res_dense)
    
    print("\n" + "="*75)
    print("RESUMEN COMPARATIVO BARRIDO ISO-PARAMÉTRICO (v311 Fast MoLoRA & Scaling)")
    print("="*75)
    print(f"{'Configuración':<28} | {'Params':<10} | {'Loss Final':<10} | {'Wall Clock (s)':<15} | {'PEI':<8}")
    print("-" * 75)
    for r in results:
        print(f"{r['model_type']:<28} | {r['params']:<10,} | {r['final_loss']:<10.4f} | {r['wall_clock_time']:<15.2f} | {r['pei']:<8.4f}")
    print("="*75)
    
    # Encontrar mejor modelo
    best_res = min(results, key=lambda x: x["final_loss"])
    
    # Registro en Master Ledger
    ledger_entry = {
        "experiment_id": "v311",
        "fecha": "2026-08-09",
        "familia": "low_rank_dinamico",
        "dataset": "sintetico_patron_1k",
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
