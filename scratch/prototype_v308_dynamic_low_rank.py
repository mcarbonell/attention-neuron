"""
v308 — Prototipo: Dynamic Multiplicative Low-Rank Adaptations (Phase 1)
Línea de investigación: Dynamic Low-Rank Adaptations
"""

import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# Metadatos e información de trazabilidad
LOG_HEADER = """
======================================================================
[00:00:00] EXECUTION HEADER & TRACEABILITY
======================================================================
Hardware: CPU / DirectML PyTorch
Threads: {}
Model File: scratch/prototype_v308_dynamic_low_rank.py
Tokenizer File: N/A (Synthetic Sequence Task)
Dataset: Synthetic Associative Recall / Sequence Task (N=1000)
Hyperparameters:
  - Batch Size: 32
  - Seq Length: 64
  - d_model: 128
  - rank r: 16
  - Learning Rate: 1e-3
  - Epochs: 5
======================================================================
""".format(torch.get_num_threads())

print(LOG_HEADER)


class DynamicMultiplicativeLowRankLinear(nn.Module):
    """
    Capa de Modulación Dinámica de Bajo Rango (Fase 1: Dynamic Gating Dual).
    Aplica compuertas multiplicativas a la entrada y a la salida calculadas dinámicamente
    mediante cuellos de botella de bajo rango r.
    
    y = sigmoid(gate_out(x)) * (W_0 * (sigmoid(gate_in(x)) * x))
    """
    def __init__(self, in_features, out_features, rank=16, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Sustrato lineal base W_0
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
            
        # Compuerta dinámica de entrada x -> r -> in_features
        self.gate_in = nn.Sequential(
            nn.Linear(in_features, rank, bias=False),
            nn.SiLU(),
            nn.Linear(rank, in_features, bias=False)
        )
        
        # Compuerta dinámica de salida x -> r -> out_features
        self.gate_out = nn.Sequential(
            nn.Linear(in_features, rank, bias=False),
            nn.SiLU(),
            nn.Linear(rank, out_features, bias=False)
        )
        
    def forward(self, x):
        # x shape: (B, T, d_in) o (B, d_in)
        g_in = torch.sigmoid(self.gate_in(x))    # (B, T, d_in)
        g_out = torch.sigmoid(self.gate_out(x))  # (B, T, d_out)
        
        # Modulación multiplicativa de entrada
        x_gated = x * g_in
        
        # Proyección lineal base W_0
        out = F.linear(x_gated, self.weight, self.bias)
        
        # Modulación multiplicativa de salida
        y = out * g_out
        return y


class StaticLoRALinear(nn.Module):
    """
    LoRA Estándar (Aditivo Estático): W_final = W_0 + (alpha/r) * (B * A)
    Las matrices A y B son estáticas para todas las entradas.
    """
    def __init__(self, in_features, out_features, rank=16, alpha=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        self.rank = rank
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
    """Capa lineal densa tradicional de control"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.linear(x)


class SimpleSequenceModel(nn.Module):
    def __init__(self, vocab_size=64, d_model=128, layer_type="dynamic_low_rank", rank=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        
        if layer_type == "dynamic_low_rank":
            self.l1 = DynamicMultiplicativeLowRankLinear(d_model, d_model, rank=rank)
            self.l2 = DynamicMultiplicativeLowRankLinear(d_model, d_model, rank=rank)
        elif layer_type == "static_lora":
            self.l1 = StaticLoRALinear(d_model, d_model, rank=rank)
            self.l2 = StaticLoRALinear(d_model, d_model, rank=rank)
        else: # standard
            self.l1 = StandardLinear(d_model, d_model)
            self.l2 = StandardLinear(d_model, d_model)
            
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        h = F.silu(self.l1(h))
        h = F.silu(self.l2(h))
        logits = self.head(h)
        return logits


def generate_synthetic_data(num_samples=1000, seq_len=64, vocab_size=64):
    """Genera datos sintéticos para tarea de modelado de secuencia"""
    torch.manual_seed(42)
    x = torch.randint(0, vocab_size, (num_samples, seq_len))
    # El target es predecir el siguiente token desplazado en 1
    y = torch.roll(x, shifts=-1, dims=1)
    return x, y


def run_experiment(model_type, rank=16, epochs=5):
    start_time = time.time()
    x_data, y_data = generate_synthetic_data()
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = SimpleSequenceModel(vocab_size=64, d_model=128, layer_type=model_type, rank=rank)
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
    print("[REGLA DE ORO] Ejecutando el CANDIDATO (v308 Dynamic Low-Rank) en primer lugar...")
    res_dynamic = run_experiment("dynamic_low_rank", rank=16)
    
    print("\nEjecutando baselines de comparación...")
    res_static = run_experiment("static_lora", rank=16)
    res_dense = run_experiment("standard_dense")
    
    print("\n" + "="*70)
    print("RESUMEN COMPARATIVO (Fase 1: Dynamic Multiplicative Low-Rank v308)")
    print("="*70)
    print(f"{'Modelo':<25} | {'Params':<10} | {'Loss Final':<10} | {'Wall Clock (s)':<15} | {'PEI':<8}")
    print("-" * 70)
    for r in [res_dynamic, res_static, res_dense]:
        print(f"{r['model_type']:<25} | {r['params']:<10,} | {r['final_loss']:<10.4f} | {r['wall_clock_time']:<15.2f} | {r['pei']:<8.4f}")
    print("="*70)
    
    # Registro en Master Ledger
    ledger_entry = {
        "experiment_id": "v308",
        "fecha": "2026-08-09",
        "familia": "low_rank_dinamico",
        "dataset": "sintetico_secuencia_1k",
        "n_eval": res_dynamic["params"],
        "metric_name": "loss",
        "value": round(res_dynamic["final_loss"], 4),
        "SE": None,
        "params": res_dynamic["params"],
        "nivel_rigor": 1,
        "etiqueta": "SEÑAL"
    }
    print("\nLínea para master_ledger.jsonl:")
    print(json.dumps(ledger_entry))
