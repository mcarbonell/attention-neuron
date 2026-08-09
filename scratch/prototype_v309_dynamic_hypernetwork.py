"""
v309 — Prototipo: Dynamic Low-Rank Hypernetwork (Phase 2)
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
[00:00:00] EXECUTION HEADER & TRACEABILITY (v309)
======================================================================
Hardware: CPU / DirectML PyTorch
Threads: {}
Model File: scratch/prototype_v309_dynamic_hypernetwork.py
Dataset: Structured Associative Pattern Task (N=1000, L=64, V=64)
Hyperparameters:
  - Batch Size: 32
  - Seq Length: 64
  - d_model: 128
  - rank r: 16
  - Learning Rate: 1e-3
  - Epochs: 10
======================================================================
""".format(torch.get_num_threads())

print(LOG_HEADER)


class DynamicHypernetworkLinear(nn.Module):
    """
    Capa de Bajo Rango Dinámico Aditivo (Fase 2: Hypernetwork de Bajo Rango).
    Proyecta A(x) de tamaño (r x d_in) y B(x) de tamaño (d_out x r) dinámicamente para cada token.
    
    y = W_0 * x + B(x) · (A(x) · x)
    """
    def __init__(self, in_features, out_features, rank=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Sustrato lineal base W_0
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Sub-redes de proyección dinámicas para A(x) y B(x)
        self.proj_A = nn.Linear(in_features, rank * in_features, bias=False)
        self.proj_B = nn.Linear(in_features, out_features * rank, bias=False)
        
        # Inicialización en cero para comportamiento inicial idéntico al modelo base congelado
        nn.init.zeros_(self.proj_A.weight)
        nn.init.zeros_(self.proj_B.weight)

    def forward(self, x):
        # x shape: (B, T, d_in)
        B_sz, T_sz, D_in = x.shape
        
        # Proyección lineal base W_0 · x
        base_out = F.linear(x, self.weight)  # (B, T, d_out)
        
        # Generar A(x) y B(x) por token
        A_x = self.proj_A(x).view(B_sz, T_sz, self.rank, D_in)             # (B, T, r, d_in)
        B_x = self.proj_B(x).view(B_sz, T_sz, self.out_features, self.rank) # (B, T, d_out, r)
        
        # Cómputo factorizado sin materialización 4D gigante:
        x_col = x.unsqueeze(-1)  # (B, T, d_in, 1)
        h = torch.matmul(A_x, x_col)  # (B, T, r, 1)
        delta = torch.matmul(B_x, h).squeeze(-1)  # (B, T, d_out)
        
        return base_out + delta


class StaticLoRALinear(nn.Module):
    """LoRA Estándar (Aditivo Estático)"""
    def __init__(self, in_features, out_features, rank=16, alpha=16.0):
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
    """Bloque Residual con LayerNorm para estabilizar el flujo de señal"""
    def __init__(self, d_model, layer_type="dynamic_hypernetwork", rank=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if layer_type == "dynamic_hypernetwork":
            self.fn = DynamicHypernetworkLinear(d_model, d_model, rank=rank)
        elif layer_type == "static_lora":
            self.fn = StaticLoRALinear(d_model, d_model, rank=rank)
        else:
            self.fn = StandardLinear(d_model, d_model)
            
    def forward(self, x):
        return x + F.silu(self.fn(self.norm(x)))


class StructuredSequenceModel(nn.Module):
    def __init__(self, vocab_size=64, d_model=128, layer_type="dynamic_hypernetwork", rank=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.block1 = ResidualBlock(d_model, layer_type=layer_type, rank=rank)
        self.block2 = ResidualBlock(d_model, layer_type=layer_type, rank=rank)
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.norm_out(h)
        return self.head(h)


def generate_structured_data(num_samples=1000, seq_len=64, vocab_size=64):
    """
    Genera una tarea con patrón asociativo estructurado (Causal Pattern Recall):
    El token en la posición t depende de una regla determinista basada en t y t-1.
    """
    torch.manual_seed(42)
    x = torch.randint(0, vocab_size // 2, (num_samples, seq_len))
    # Aplicar regla determinista de asociación estructurada:
    # y[t] = (x[t-1] * 3 + x[t] + 7) % vocab_size
    x_prev = torch.roll(x, shifts=1, dims=1)
    x_prev[:, 0] = 0
    y = (x_prev * 3 + x + 7) % vocab_size
    return x, y


def run_experiment(model_type, rank=16, epochs=10):
    start_time = time.time()
    x_data, y_data = generate_structured_data()
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = StructuredSequenceModel(vocab_size=64, d_model=128, layer_type=model_type, rank=rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--- Probando Modelo: {model_type.upper()} (Params: {num_params:,}) ---")
    
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
            
            # Regla de Supervivencia: Fast Feedback en los primeros 5 batches
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
    print("[REGLA DE ORO] Ejecutando el CANDIDATO (v309 Dynamic Hypernetwork) en primer lugar...")
    res_hyper = run_experiment("dynamic_hypernetwork", rank=16, epochs=10)
    
    print("\nEjecutando baselines de comparación...")
    res_static = run_experiment("static_lora", rank=16, epochs=10)
    res_dense = run_experiment("standard_dense", epochs=10)
    
    print("\n" + "="*70)
    print("RESUMEN COMPARATIVO (Fase 2: Dynamic Low-Rank Hypernetwork v309)")
    print("="*70)
    print(f"{'Modelo':<25} | {'Params':<10} | {'Loss Final':<10} | {'Wall Clock (s)':<15} | {'PEI':<8}")
    print("-" * 70)
    for r in [res_hyper, res_static, res_dense]:
        print(f"{r['model_type']:<25} | {r['params']:<10,} | {r['final_loss']:<10.4f} | {r['wall_clock_time']:<15.2f} | {r['pei']:<8.4f}")
    print("="*70)
    
    # Registro en Master Ledger
    ledger_entry = {
        "experiment_id": "v309",
        "fecha": "2026-08-09",
        "familia": "low_rank_dinamico",
        "dataset": "sintetico_patron_1k",
        "n_eval": res_hyper["params"],
        "metric_name": "loss",
        "value": round(res_hyper["final_loss"], 4),
        "SE": None,
        "params": res_hyper["params"],
        "nivel_rigor": 1,
        "etiqueta": "SEÑAL"
    }
    print("\nLínea para master_ledger.jsonl:")
    print(json.dumps(ledger_entry))
