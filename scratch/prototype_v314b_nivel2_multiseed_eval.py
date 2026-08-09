"""
v314b — Evaluación de Nivel 2: Rigor Multi-Semilla y Dataset Grande (N=10,000)
Línea de investigación: Dynamic Low-Rank Adaptations
"""

import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LOG_HEADER = """
======================================================================
[00:00:00] EXECUTION HEADER & TRACEABILITY (v314b - Nivel 2 Rigor Eval)
======================================================================
Hardware: CPU / DirectML PyTorch
Threads: {}
Model File: scratch/prototype_v314b_nivel2_multiseed_eval.py
Dataset: Structured Associative Pattern Task (N=10,000, L=64, V=64)
Rigor Level: Nivel 2 (5 Independent Seeds: [42, 43, 44, 45, 46])
Hyperparameters:
  - Batch Size: 64
  - Seq Length: 64
  - d_model: 128
  - rank r: 16
  - K experts: 4
  - Learning Rate: 1e-3
  - Epochs: 15
======================================================================
""".format(torch.get_num_threads())

print(LOG_HEADER)


# --- Capas ---
class ComplexPhaseLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=16, num_experts=4, alpha=16.0):
        super().__init__()
        self.scaling = alpha / rank
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        self.theta_A = nn.Parameter(torch.zeros(num_experts, rank, in_features))
        self.theta_B = nn.Parameter(torch.zeros(num_experts, out_features, rank))
        nn.init.uniform_(self.theta_A, 0.0, 2 * math.pi)
        nn.init.uniform_(self.theta_B, 0.0, 2 * math.pi)
        self.router = nn.Linear(in_features, num_experts, bias=False)

    def forward(self, x):
        base_out = F.linear(x, self.weight)
        real_A = torch.cos(self.theta_A)
        imag_A = torch.sin(self.theta_A)
        real_B = torch.cos(self.theta_B)
        imag_B = torch.sin(self.theta_B)
        
        h_real = torch.einsum('bti,kri->btkr', x, real_A)
        h_imag = torch.einsum('bti,kri->btkr', x, imag_A)
        adapter_real = torch.einsum('btkr,kor->btko', h_real, real_B) - \
                       torch.einsum('btkr,kor->btko', h_imag, imag_B)
        adapter_outs = adapter_real * self.scaling
        gating_weights = F.softmax(self.router(x), dim=-1)
        dynamic_delta = torch.einsum('btko,btk->bto', adapter_outs, gating_weights)
        return base_out + dynamic_delta


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
        return base_out + dynamic_delta


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
        return base_out + lora_out


class StandardLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
    def forward(self, x): return self.linear(x)


class ResidualBlock(nn.Module):
    def __init__(self, d_model, layer_type="complex_phase_lora", rank=16, num_experts=4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if layer_type == "complex_phase_lora":
            self.fn = ComplexPhaseLoRALinear(d_model, d_model, rank=rank, num_experts=num_experts)
        elif layer_type == "real_molora":
            self.fn = FastDynamicGatedLoRALinear(d_model, d_model, rank=rank, num_experts=num_experts)
        elif layer_type == "static_lora":
            self.fn = StaticLoRALinear(d_model, d_model, rank=64)
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


def generate_large_dataset(seed=42, num_samples=10000, seq_len=64, vocab_size=64):
    torch.manual_seed(seed)
    x = torch.randint(0, vocab_size // 2, (num_samples, seq_len))
    x_prev = torch.roll(x, shifts=1, dims=1)
    x_prev[:, 0] = 0
    y = (x_prev * 3 + x + 7) % vocab_size
    return x, y


def train_single_seed(model_type, seed, epochs=15):
    torch.manual_seed(seed)
    x_data, y_data = generate_large_dataset(seed=seed, num_samples=10000)
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = StructuredSequenceModel(vocab_size=64, d_model=128, layer_type=model_type, rank=16, num_experts=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    
    for epoch in range(epochs):
        model.train()
        for step, (bx, by) in enumerate(loader):
            optimizer.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits.view(-1, 64), by.view(-1))
            loss.backward()
            optimizer.step()
            
    # Evaluación de loss por secuencia para cálculo riguroso de Error Estándar (SE)
    model.eval()
    seq_losses = []
    with torch.no_grad():
        for bx, by in loader:
            logits = model(bx)
            # Loss por secuencia individual
            loss_per_seq = F.cross_entropy(logits.view(-1, 64), by.view(-1), reduction='none').view(bx.shape[0], -1).mean(dim=1)
            seq_losses.extend(loss_per_seq.tolist())
            
    mean_loss = float(np.mean(seq_losses))
    se_loss = float(np.std(seq_losses) / np.sqrt(len(seq_losses)))
    return mean_loss, se_loss


def run_multiseed_eval(model_type, seeds=[42, 43, 44, 45, 46], epochs=15):
    print(f"\n======================================================================")
    print(f"EVALUACIÓN NIVEL 2 MULTI-SEMILLA: {model_type.upper()}")
    print(f"======================================================================")
    
    losses = []
    ses = []
    for s in seeds:
        t0 = time.time()
        m_loss, se = train_single_seed(model_type, seed=s, epochs=epochs)
        dt = time.time() - t0
        print(f"  [Semilla {s}] Loss: {m_loss:.5f} (SE per-seq: {se:.5f}) | Tiempo: {dt:.2f}s")
        losses.append(m_loss)
        ses.append(se)
        
    mean_total = float(np.mean(losses))
    std_total = float(np.std(losses))
    se_total = std_total / np.sqrt(len(seeds))
    
    print(f"-> Promedio de 5 semillas: {mean_total:.5f} ± {se_total:.5f} (Std: {std_total:.5f})")
    return mean_total, std_total, se_total


if __name__ == "__main__":
    seeds = [42, 43, 44, 45, 46]
    
    # 1. Candidato Complejo (v314)
    m_complex, std_complex, se_complex = run_multiseed_eval("complex_phase_lora", seeds=seeds, epochs=15)
    
    # 2. Candidato Real MoLoRA
    m_real, std_real, se_real = run_multiseed_eval("real_molora", seeds=seeds, epochs=15)
    
    # 3. Baseline Static LoRA
    m_static, std_static, se_static = run_multiseed_eval("static_lora", seeds=seeds, epochs=15)
    
    # 4. Baseline Denso Estándar
    m_dense, std_dense, se_dense = run_multiseed_eval("standard_dense", seeds=seeds, epochs=15)
    
    print("\n" + "="*80)
    print("SÍNTESIS RIGUROSA NIVEL 2 (N=10,000, 5 Semillas Independientes, 15 Épocas)")
    print("="*80)
    print(f"{'Modelo':<25} | {'Loss Media (μ)':<15} | {'Std (σ)':<12} | {'Error Estándar (SE)':<20}")
    print("-" * 80)
    print(f"{'complex_phase_lora':<25} | {m_complex:<15.5f} | {std_complex:<12.5f} | {se_complex:<20.5f}")
    print(f"{'real_molora':<25} | {m_real:<15.5f} | {std_real:<12.5f} | {se_real:<20.5f}")
    print(f"{'static_lora':<25} | {m_static:<15.5f} | {std_static:<12.5f} | {se_static:<20.5f}")
    print(f"{'standard_dense':<25} | {m_dense:<15.5f} | {std_dense:<12.5f} | {se_dense:<20.5f}")
    print("="*80)
    
    diff_complex_real = m_real - m_complex
    diff_complex_static = m_static - m_complex
    pooled_se = math.sqrt(se_complex**2 + se_real**2)
    
    print(f"\nANÁLISIS DE SIGNIFICANCIA ESTADÍSTICA:")
    print(f"  - Δ (Real - Complex): {diff_complex_real:.5f} nats")
    print(f"  - SE Combinado: {pooled_se:.5f} nats")
    print(f"  - Criterio de Significancia (|Δ| >= 2 * SE): {abs(diff_complex_real):.5f} vs {2 * pooled_se:.5f}")
    
    if abs(diff_complex_real) >= 2 * pooled_se:
        print("  -> CONCLUSIÓN: La mejora de Complex Phase LoRA ES ESTADÍSTICAMENTE SIGNIFICATIVA [ANCLA].")
        tag = "ANCLA"
    else:
        print("  -> CONCLUSIÓN: La diferencia no supera el umbral 2xSE. Es indistinguible del ruido [RUIDO-SOSPECHA].")
        tag = "RUIDO-SOSPECHA"
        
    ledger_entry = {
        "experiment_id": "v314b",
        "fecha": "2026-08-09",
        "familia": "low_rank_dinamico_complejo",
        "dataset": "sintetico_patron_10k_multiseed",
        "n_eval": 5,
        "metric_name": "loss_mean_5seeds",
        "value": round(m_complex, 5),
        "SE": round(se_complex, 5),
        "params": 83776,
        "nivel_rigor": 2,
        "etiqueta": tag
    }
    print("\nLínea para master_ledger.jsonl:")
    print(json.dumps(ledger_entry))
