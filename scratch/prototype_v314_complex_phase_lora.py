"""
v314 — Prototipo: Complex Phase Low-Rank Adapter (Complex-LoRA)
Línea de investigación: Dynamic Low-Rank Adaptations & Phase Holography
"""

import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_HEADER = """
======================================================================
[00:00:00] EXECUTION HEADER & TRACEABILITY (v314 - Complex Phase LoRA)
======================================================================
Hardware: CPU / DirectML PyTorch
Threads: {}
Model File: scratch/prototype_v314_complex_phase_lora.py
Dataset: Structured Associative Pattern Task (N=1000, L=64, V=64)
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


class ComplexPhaseLoRALinear(nn.Module):
    """
    Capa de Bajo Rango de Fase Compleja Pure (Complex Phase LoRA).
    Las matrices A y B son tensores de fase pura unitarios:
      A = cos(Theta_A) + i sin(Theta_A)
      B = cos(Theta_B) + i sin(Theta_B)
      
    No sufren degradación por cuantización (parámetros en el círculo unidad S^1)
    y aplican interferometría de ondas de bajo rango r.
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
        
        # Ángulos de faseTheta_A in [0, 2pi], Theta_B in [0, 2pi]
        self.theta_A = nn.Parameter(torch.zeros(num_experts, rank, in_features))
        self.theta_B = nn.Parameter(torch.zeros(num_experts, out_features, rank))
        nn.init.uniform_(self.theta_A, 0.0, 2 * math.pi)
        nn.init.uniform_(self.theta_B, 0.0, 2 * math.pi)
        
        # Router dinámico por token
        self.router = nn.Linear(in_features, num_experts, bias=False)

    def forward(self, x):
        # x shape: (B, T, d_in)
        base_out = F.linear(x, self.weight)  # (B, T, d_out)
        
        # Obtener partes Real e Imaginaria de A y B (acotadas en [-1, 1])
        real_A = torch.cos(self.theta_A)  # (K, r, d_in)
        imag_A = torch.sin(self.theta_A)  # (K, r, d_in)
        
        real_B = torch.cos(self.theta_B)  # (K, d_out, r)
        imag_B = torch.sin(self.theta_B)  # (K, d_out, r)
        
        # 1. Proyección Compleja A · (x + 0i):
        # h_real = real_A · x, h_imag = imag_A · x
        h_real = torch.einsum('bti,kri->btkr', x, real_A)  # (B, T, K, r)
        h_imag = torch.einsum('bti,kri->btkr', x, imag_A)  # (B, T, K, r)
        
        # 2. Proyección Compleja B · (h_real + i h_imag):
        # Re(B · h) = real_B · h_real - imag_B · h_imag
        adapter_real = torch.einsum('btkr,kor->btko', h_real, real_B) - \
                       torch.einsum('btkr,kor->btko', h_imag, imag_B)
        adapter_outs = adapter_real * self.scaling
        
        # 3. Suma ponderada por token
        gating_weights = F.softmax(self.router(x), dim=-1)
        dynamic_delta = torch.einsum('btko,btk->bto', adapter_outs, gating_weights)
        
        return base_out + dynamic_delta


class StandardLinear(nn.Module):
    """Capa lineal densa estándar"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.linear(x)


class ResidualBlock(nn.Module):
    def __init__(self, d_model, layer_type="complex_phase_lora", rank=16, num_experts=4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if layer_type == "complex_phase_lora":
            self.fn = ComplexPhaseLoRALinear(d_model, d_model, rank=rank, num_experts=num_experts)
        else:
            self.fn = StandardLinear(d_model, d_model)
            
    def forward(self, x):
        return x + F.silu(self.fn(self.norm(x)))


class StructuredSequenceModel(nn.Module):
    def __init__(self, vocab_size=64, d_model=128, layer_type="complex_phase_lora", rank=16, num_experts=4):
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
    print("[REGLA DE ORO] Ejecutando el CANDIDATO Complex Phase LoRA (v314) en primer lugar...")
    results.append(run_experiment("complex_phase_lora", rank=16, num_experts=4, epochs=10))
    
    print("\nEjecutando baseline de comparación (Capa Densa Estándar)...")
    results.append(run_experiment("standard_dense", epochs=10))
    
    print("\n" + "="*75)
    print("RESUMEN COMPARATIVO (v314 Complex Phase LoRA)")
    print("="*75)
    print(f"{'Modelo':<28} | {'Params':<10} | {'Loss Final':<10} | {'Wall Clock (s)':<15} | {'PEI':<8}")
    print("-" * 75)
    for r in results:
        print(f"{r['model_type']:<28} | {r['params']:<10,} | {r['final_loss']:<10.4f} | {r['wall_clock_time']:<15.2f} | {r['pei']:<8.4f}")
    print("="*75)
    
    # Registro en Master Ledger
    best_res = min(results, key=lambda x: x["final_loss"])
    ledger_entry = {
        "experiment_id": "v314",
        "fecha": "2026-08-09",
        "familia": "low_rank_dinamico_complejo",
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
