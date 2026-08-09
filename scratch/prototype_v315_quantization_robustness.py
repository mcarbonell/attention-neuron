"""
v315 — Prototipo: Resistencia a la Cuantización Post-Entrenamiento a 4 Bits (FP32 vs INT4/4-bit Phase)
Línea de investigación: Dynamic Low-Rank Adaptations & Phase Quantization
"""

import time
import math
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_HEADER = """
======================================================================
[00:00:00] EXECUTION HEADER & TRACEABILITY (v315 - 4-Bit Quantization)
======================================================================
Hardware: CPU / DirectML PyTorch
Threads: {}
Model File: scratch/prototype_v315_quantization_robustness.py
Dataset: Structured Associative Pattern Task (N=2000, L=64, V=64)
Evaluation: FP32 Baseline vs 4-Bit Post-Training Quantization
  - Complex MoLoRA: 4-bit Uniform Angle Discretization in S^1 (16 bins)
  - Real Models: 4-bit Uniform Min-Max Quantization in R (16 levels)
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

    def quantize_4bit(self):
        """Cuantización de Ángulos de Fase a 4 Bits (16 bins uniformes en S^1)"""
        with torch.no_grad():
            # Discretizar theta_A y theta_B en 16 posiciones uniformes en [0, 2pi]
            self.theta_A.copy_(torch.round(self.theta_A / (2 * math.pi) * 16) * (2 * math.pi / 16))
            self.theta_B.copy_(torch.round(self.theta_B / (2 * math.pi) * 16) * (2 * math.pi / 16))
            # Cuantizar también la matriz base W_0 a 4 bits min-max
            w_min, w_max = self.weight.min(), self.weight.max()
            self.weight.copy_(torch.round((self.weight - w_min) / (w_max - w_min + 1e-8) * 15) / 15 * (w_max - w_min) + w_min)


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

    def quantize_4bit(self):
        """Cuantización Min-Max Uniforme a 4 Bits (INT4 en R)"""
        with torch.no_grad():
            for param in [self.weight, self.lora_A, self.lora_B]:
                p_min, p_max = param.min(), param.max()
                param.copy_(torch.round((param - p_min) / (p_max - p_min + 1e-8) * 15) / 15 * (p_max - p_min) + p_min)


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

    def quantize_4bit(self):
        with torch.no_grad():
            for param in [self.weight, self.lora_A, self.lora_B]:
                p_min, p_max = param.min(), param.max()
                param.copy_(torch.round((param - p_min) / (p_max - p_min + 1e-8) * 15) / 15 * (p_max - p_min) + p_min)


class StandardLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x): return self.linear(x)

    def quantize_4bit(self):
        with torch.no_grad():
            p = self.linear.weight
            p_min, p_max = p.min(), p.max()
            p.copy_(torch.round((p - p_min) / (p_max - p_min + 1e-8) * 15) / 15 * (p_max - p_min) + p_min)


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

    def apply_4bit_quantization(self):
        """Aplica la cuantización a 4 bits en las capas ocultas"""
        self.block1.fn.quantize_4bit()
        self.block2.fn.quantize_4bit()


def generate_data(num_samples=2000, seq_len=64, vocab_size=64):
    torch.manual_seed(42)
    x = torch.randint(0, vocab_size // 2, (num_samples, seq_len))
    x_prev = torch.roll(x, shifts=1, dims=1)
    x_prev[:, 0] = 0
    y = (x_prev * 3 + x + 7) % vocab_size
    return x, y


def evaluate_quantization(model_type, epochs=10):
    torch.manual_seed(42)
    x_data, y_data = generate_data()
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = StructuredSequenceModel(vocab_size=64, d_model=128, layer_type=model_type, rank=16, num_experts=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--- Entrenando Modelo en FP32: {model_type.upper()} ({num_params:,} params) ---")
    
    # 1. Entrenamiento en FP32
    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits.view(-1, 64), by.view(-1))
            loss.backward()
            optimizer.step()
            
    # Evaluar Loss FP32
    model.eval()
    with torch.no_grad():
        fp32_losses = []
        for bx, by in loader:
            logits = model(bx)
            loss = F.cross_entropy(logits.view(-1, 64), by.view(-1))
            fp32_losses.append(loss.item())
    loss_fp32 = sum(fp32_losses) / len(fp32_losses)
    
    # 2. Copia y Aplicar Cuantización Post-Entrenamiento a 4-Bits
    model_quantized = copy.deepcopy(model)
    model_quantized.apply_4bit_quantization()
    model_quantized.eval()
    
    with torch.no_grad():
        q4_losses = []
        for bx, by in loader:
            logits = model_quantized(bx)
            loss = F.cross_entropy(logits.view(-1, 64), by.view(-1))
            q4_losses.append(loss.item())
    loss_4bit = sum(q4_losses) / len(q4_losses)
    
    deg = loss_4bit - loss_fp32
    rel_deg_pct = (deg / loss_fp32) * 100.0
    
    print(f"  Loss FP32: {loss_fp32:.4f}")
    print(f"  Loss 4-Bit: {loss_4bit:.4f}")
    print(f"  Degradación por Cuantización (Δ Loss): {deg:+.4f} ({rel_deg_pct:+.2f}%)")
    
    return {
        "model_type": model_type,
        "params": num_params,
        "loss_fp32": loss_fp32,
        "loss_4bit": loss_4bit,
        "degradation": deg,
        "rel_degradation_pct": rel_deg_pct
    }


if __name__ == "__main__":
    results = []
    
    print("[REGLA DE ORO] Evaluando el CANDIDATO Complex Phase LoRA (v315) en primer lugar...")
    results.append(evaluate_quantization("complex_phase_lora", epochs=10))
    
    print("\nEvaluando baselines en el dominio Real R...")
    results.append(evaluate_quantization("real_molora", epochs=10))
    results.append(evaluate_quantization("static_lora", epochs=10))
    results.append(evaluate_quantization("standard_dense", epochs=10))
    
    print("\n" + "="*85)
    print("RESUMEN BENCHMARK RESISTENCIA A CUANTIZACIÓN A 4-BITS (v315)")
    print("="*85)
    print(f"{'Modelo':<22} | {'Params':<10} | {'Loss FP32':<12} | {'Loss 4-Bit':<12} | {'Δ Loss (Degradación)':<20}")
    print("-" * 85)
    for r in results:
        print(f"{r['model_type']:<22} | {r['params']:<10,} | {r['loss_fp32']:<12.4f} | {r['loss_4bit']:<12.4f} | {r['degradation']:+.4f} ({r['rel_degradation_pct']:+.2f}%)")
    print("="*85)
    
    # Encontrar modelo con menor degradación
    best_quant = min(results, key=lambda x: x["degradation"])
    
    ledger_entry = {
        "experiment_id": "v315",
        "fecha": "2026-08-09",
        "familia": "cuantizacion_fase_compleja",
        "dataset": "sintetico_patron_2k",
        "n_eval": best_quant["params"],
        "metric_name": "quant_degradation_delta_loss",
        "value": round(best_quant["degradation"], 4),
        "SE": None,
        "params": best_quant["params"],
        "nivel_rigor": 1,
        "etiqueta": "ANCLA"
    }
    print("\nLínea para master_ledger.jsonl:")
    print(json.dumps(ledger_entry))
