"""
v319 — Prototipo: Benchmark Vocabulario Zipf Ley de Potencias (V=4096, Fase 12)
Línea de investigación: Dynamic Low-Rank Adaptations & Zipfian Natural Language Distribution
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
[00:00:00] EXECUTION HEADER & TRACEABILITY (v319 - Zipf Vocab V=4096)
======================================================================
Hardware: CPU / DirectML PyTorch
Threads: {}
Model File: scratch/prototype_v319_zipf_vocab_test.py
Dataset: Zipfian Power-Law Token Distribution (N=2000, L=64, V=4096)
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
        # Sesgo positivo inicial para permitir apertura progresiva por token
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


class ContinuousDyRankLinear(nn.Module):
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
        rank_mask = torch.sigmoid(self.rank_gate(x)).view(B_sz, T_sz, -1, self.lora_A.shape[1])
        h = torch.einsum('bti,kri->btkr', x, self.lora_A)
        h_gated = h * rank_mask
        adapter_outs = torch.einsum('btkr,kor->btko', h_gated, self.lora_B) * self.scaling
        gating_weights = F.softmax(self.router(x), dim=-1)
        dynamic_delta = torch.einsum('btko,btk->bto', adapter_outs, gating_weights)
        return base_out + dynamic_delta, rank_mask


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
        elif layer_type == "continuous_dyrank":
            self.fn = ContinuousDyRankLinear(d_model, d_model, rank=rank, num_experts=num_experts)
        elif layer_type == "fast_molora":
            self.fn = FastDynamicGatedLoRALinear(d_model, d_model, rank=rank, num_experts=num_experts)
        else:
            self.fn = StandardLinear(d_model, d_model)
            
    def forward(self, x):
        out_fn, mask = self.fn(self.norm(x))
        return x + F.silu(out_fn), mask


class ZipfSequenceModel(nn.Module):
    def __init__(self, vocab_size=4096, d_model=128, layer_type="hard_binary_dyrank", rank=16, num_experts=4):
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


def generate_zipf_data(num_samples=2000, seq_len=64, vocab_size=4096, s=1.07):
    """Genera datos siguiendo la ley de potencias de Zipf P(k) ~ 1/k^s"""
    np.random.seed(42)
    torch.manual_seed(42)
    
    ranks = np.arange(1, vocab_size + 1)
    probs = 1.0 / (ranks ** s)
    probs /= probs.sum()
    
    flat_tokens = np.random.choice(vocab_size, size=(num_samples, seq_len), p=probs)
    x = torch.from_numpy(flat_tokens).long()
    
    x_prev = torch.roll(x, shifts=1, dims=1)
    x_prev[:, 0] = 0
    y = (x_prev * 3 + x + 7) % vocab_size
    return x, y


def run_experiment(model_type, vocab_size=4096, rank=16, num_experts=4, epochs=10):
    start_time = time.time()
    x_data, y_data = generate_zipf_data(num_samples=2000, seq_len=64, vocab_size=vocab_size)
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = ZipfSequenceModel(vocab_size=vocab_size, d_model=128, layer_type=model_type, rank=rank, num_experts=num_experts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--- Probando Modelo: {model_type.upper()} en Vocabulario Zipf V={vocab_size} (Params: {num_params:,}) ---")
    
    final_loss = 0.0
    eval_time_accum = 0.0
    active_freq_tokens = []
    active_rare_tokens = []
    
    for epoch in range(epochs):
        model.train()
        for step, (bx, by) in enumerate(loader):
            step_start = time.time()
            optimizer.zero_grad()
            logits, (m1, m2) = model(bx)
            loss = F.cross_entropy(logits.view(-1, vocab_size), by.view(-1))
            loss.backward()
            optimizer.step()
            
            step_eval_time = time.time() - step_start
            eval_time_accum += step_eval_time
            
            if m1 is not None:
                # bx shape: (B, T). Top 5% Frequent tokens: ID <= 200. Rare tokens: ID > 200
                freq_mask = (bx <= 200).unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
                rare_mask = (bx > 200).unsqueeze(-1).unsqueeze(-1)   # (B, T, 1, 1)
                
                m1_vals = (m1 >= 0.5).float()
                if freq_mask.sum() > 0:
                    active_freq = (m1_vals * freq_mask).sum() / (freq_mask.sum() * rank * num_experts)
                    active_freq_tokens.append(active_freq.item() * 100.0)
                if rare_mask.sum() > 0:
                    active_rare = (m1_vals * rare_mask).sum() / (rare_mask.sum() * rank * num_experts)
                    active_rare_tokens.append(active_rare.item() * 100.0)
                
            # Fast Feedback
            if epoch == 0 and step < 5:
                freq_str = f" | Active Rank Freq (Top 5%): {active_freq_tokens[-1]:.1f}% | Active Rank Rare: {active_rare_tokens[-1]:.1f}%" if active_freq_tokens else ""
                print(f"[Fast Feedback] Batch {step+1}/5 - Loss: {loss.item():.4f}{freq_str} - Step Time: {step_eval_time*1000:.2f}ms")
                
            final_loss = loss.item()

    wall_clock_time = time.time() - start_time
    overhead_time = wall_clock_time - eval_time_accum
    pei = (1.0 / (final_loss + 1e-6)) / math.log10(num_params + 1)
    
    mean_active_freq = float(torch.tensor(active_freq_tokens).mean()) if active_freq_tokens else 100.0
    mean_active_rare = float(torch.tensor(active_rare_tokens).mean()) if active_rare_tokens else 100.0
    
    print(f"Final Loss Zipf: {final_loss:.4f} | Freq Tokens (Top 5%) Active Rank: {mean_active_freq:.1f}% | Rare Tokens Active Rank: {mean_active_rare:.1f}% | Wall Clock: {wall_clock_time:.2f}s")
    
    return {
        "model_type": model_type,
        "params": num_params,
        "final_loss": final_loss,
        "active_freq_pct": mean_active_freq,
        "active_rare_pct": mean_active_rare,
        "wall_clock_time": wall_clock_time,
        "eval_time": eval_time_accum,
        "overhead_time": overhead_time,
        "pei": pei
    }


if __name__ == "__main__":
    results = []
    print("[REGLA DE ORO] Ejecutando el CANDIDATO Hard Binary DyRank STE (v319) en primer lugar...")
    results.append(run_experiment("hard_binary_dyrank", vocab_size=4096, rank=16, num_experts=4, epochs=10))
    
    print("\nEjecutando baselines de comparación...")
    results.append(run_experiment("continuous_dyrank", vocab_size=4096, rank=16, num_experts=4, epochs=10))
    results.append(run_experiment("fast_molora", vocab_size=4096, rank=16, num_experts=4, epochs=10))
    results.append(run_experiment("standard_dense", vocab_size=4096, epochs=10))
    
    print("\n" + "="*95)
    print("RESUMEN BENCHMARK VOCABULARIO ZIPF V=4096 (v319)")
    print("="*95)
    print(f"{'Modelo':<25} | {'Params':<10} | {'Loss Final':<10} | {'Active Rank Freq':<18} | {'Active Rank Rare':<18}")
    print("-" * 95)
    for r in results:
        print(f"{r['model_type']:<25} | {r['params']:<10,} | {r['final_loss']:<10.4f} | {r['active_freq_pct']:<18.1f}% | {r['active_rare_pct']:<18.1f}%")
    print("="*95)
    
    best_res = min(results, key=lambda x: x["final_loss"])
    
    ledger_entry = {
        "experiment_id": "v319",
        "fecha": "2026-08-09",
        "familia": "low_rank_dinamico_zipf",
        "dataset": "sintetico_zipf_v4096_2k",
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
