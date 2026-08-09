"""
v312 — Prototipo: Dynamic Gated LoRA (MoLoRA) en MQAR Benchmark (Phase 5)
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
[00:00:00] EXECUTION HEADER & TRACEABILITY (v312 - MoLoRA on MQAR)
======================================================================
Hardware: CPU / DirectML PyTorch
Threads: {}
Model File: scratch/prototype_v312_molora_mqar.py
Dataset: Multi-Query Associative Recall (MQAR, L=64, N_pairs=8, Vocab=128)
Hyperparameters:
  - Batch Size: 32
  - Seq Length: 64
  - n_pairs: 8
  - d_model: 128
  - Learning Rate: 1e-3
  - Steps: 400
======================================================================
""".format(torch.get_num_threads())

print(LOG_HEADER)


# --- 1. MQAR On-The-Fly Generator ---
def generate_mqar_batch(batch_size=32, n_pairs=8, seq_len=64, num_tokens=100, device="cpu"):
    pad_id = 0
    query_marker = num_tokens + 1
    
    tokens_needed = 2 * n_pairs
    rand_t = torch.rand(batch_size, num_tokens, device=device)
    sampled = torch.argsort(rand_t, dim=-1)[:, :tokens_needed] + 1
    keys = sampled[:, :n_pairs]
    vals = sampled[:, n_pairs:]
    
    x = torch.full((batch_size, seq_len), pad_id, dtype=torch.long, device=device)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
    
    kv = torch.stack([keys, vals], dim=2).view(batch_size, 2 * n_pairs)
    x[:, :2 * n_pairs] = kv
    
    q_perm = torch.argsort(torch.rand(batch_size, n_pairs, device=device), dim=-1)
    query_keys = torch.gather(keys, 1, q_perm)
    query_vals = torch.gather(vals, 1, q_perm)
    
    gap = 2
    pos_q = (2 * n_pairs + gap + 2 * torch.arange(n_pairs, device=device)).unsqueeze(0).expand(batch_size, -1)
             
    x.scatter_(1, pos_q, query_marker)
    x.scatter_(1, pos_q + 1, query_keys)
    y.scatter_(1, pos_q + 1, query_vals)
    return x, y


# --- 2. Capas y Modelos ---
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


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class MQARModel(nn.Module):
    def __init__(self, vocab_size=120, d_model=128, layer_type="fast_molora", rank=16, num_experts=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        if layer_type == "fast_molora":
            self.layer1 = FastDynamicGatedLoRALinear(d_model, d_model, rank=rank, num_experts=num_experts)
            self.layer2 = FastDynamicGatedLoRALinear(d_model, d_model, rank=rank, num_experts=num_experts)
        elif layer_type == "static_lora":
            self.layer1 = StaticLoRALinear(d_model, d_model, rank=64)
            self.layer2 = StaticLoRALinear(d_model, d_model, rank=64)
        elif layer_type == "mha":
            self.layer1 = CausalSelfAttention(d_model=d_model, n_heads=4)
            self.layer2 = CausalSelfAttention(d_model=d_model, n_heads=4)
        else: # dense
            self.layer1 = nn.Linear(d_model, d_model, bias=False)
            self.layer2 = nn.Linear(d_model, d_model, bias=False)
            
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        h = h + F.silu(self.layer1(self.norm1(h)))
        h = h + F.silu(self.layer2(self.norm2(h)))
        return self.head(h)


def eval_mqar(model_type, rank=16, num_experts=4, steps=400):
    start_time = time.time()
    torch.manual_seed(42)
    
    vocab_size = 120
    model = MQARModel(vocab_size=vocab_size, d_model=128, layer_type=model_type, rank=rank, num_experts=num_experts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    label = f"{model_type}_K{num_experts}_r{rank}" if model_type == "fast_molora" else model_type
    print(f"\n--- Evaluando MQAR: {label.UPPER() if hasattr(label, 'UPPER') else label.upper()} (Params: {num_params:,}) ---")
    
    final_acc = 0.0
    final_loss = 0.0
    eval_time_accum = 0.0
    
    for step in range(steps):
        step_start = time.time()
        bx, by = generate_mqar_batch(batch_size=32, n_pairs=8, seq_len=64, num_tokens=100)
        
        optimizer.zero_grad()
        logits = model(bx)
        
        # Loss solo sobre posiciones de respuesta (-100 ignorado)
        loss = F.cross_entropy(logits.view(-1, vocab_size), by.view(-1), ignore_index=-100)
        loss.backward()
        optimizer.step()
        
        step_eval_time = time.time() - step_start
        eval_time_accum += step_eval_time
        
        # Accuracy sobre targets válidos
        mask = (by != -100)
        preds = torch.argmax(logits, dim=-1)
        correct = (preds == by) & mask
        acc = (correct.sum().float() / mask.sum().float()).item() * 100.0
        
        if step < 5:
            print(f"[Fast Feedback] Step {step+1}/5 - Loss: {loss.item():.4f} | Target Acc: {acc:.2f}% | Step Time: {step_eval_time*1000:.2f}ms")
            
        final_loss = loss.item()
        final_acc = acc

    wall_clock_time = time.time() - start_time
    overhead_time = wall_clock_time - eval_time_accum
    pei = (final_acc / 100.0) / math.log10(num_params + 1)
    
    print(f"Final Target Acc: {final_acc:.2f}% | Final Loss: {final_loss:.4f} | Wall Clock: {wall_clock_time:.2f}s | PEI: {pei:.4f}")
    
    return {
        "model_type": label,
        "params": num_params,
        "target_acc": final_acc,
        "final_loss": final_loss,
        "wall_clock_time": wall_clock_time,
        "eval_time": eval_time_accum,
        "overhead_time": overhead_time,
        "pei": pei
    }


if __name__ == "__main__":
    results = []
    
    print("[REGLA DE ORO] Ejecutando los CANDIDATOS MoLoRA en primer lugar...")
    # Candidatos MoLoRA K=16 y K=8
    results.append(eval_mqar("fast_molora", rank=4, num_experts=16, steps=400))
    results.append(eval_mqar("fast_molora", rank=8, num_experts=8, steps=400))
    
    print("\nEjecutando baselines de comparación...")
    results.append(eval_mqar("static_lora", rank=64, steps=400))
    results.append(eval_mqar("dense", steps=400))
    results.append(eval_mqar("mha", steps=400))
    
    print("\n" + "="*80)
    print("RESUMEN COMPARATIVO BENCHMARK MQAR (v312 MoLoRA on MQAR)")
    print("="*80)
    print(f"{'Modelo':<28} | {'Params':<10} | {'Target Acc (%)':<15} | {'Loss':<10} | {'Wall Clock (s)':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['model_type']:<28} | {r['params']:<10,} | {r['target_acc']:<15.2f} | {r['final_loss']:<10.4f} | {r['wall_clock_time']:<12.2f}")
    print("="*80)
    
    best_res = max(results, key=lambda x: x["target_acc"])
    
    # Registro en Master Ledger
    ledger_entry = {
        "experiment_id": "v312",
        "fecha": "2026-08-09",
        "familia": "low_rank_dinamico",
        "dataset": "MQAR synthetic (L=64, pairs=8)",
        "n_eval": best_res["params"],
        "metric_name": "target_acc",
        "value": round(best_res["target_acc"], 2),
        "SE": None,
        "params": best_res["params"],
        "nivel_rigor": 1,
        "etiqueta": "ANCLA"
    }
    print("\nLínea para master_ledger.jsonl:")
    print(json.dumps(ledger_entry))
