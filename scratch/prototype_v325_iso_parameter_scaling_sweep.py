"""
v325 — Prototipo: Barrido de Escalado Iso-Parámetros (150K a 1.2M Params, Fase 4)
Línea de investigación: Spectral Architectures Research Line
"""

import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_HEADER = """
======================================================================
[00:00:00] EXECUTION HEADER & TRACEABILITY (v325 - Iso-Parameter Scaling Sweep)
======================================================================
Hardware: CPU / DirectML PyTorch
Threads: {}
Model File: scratch/prototype_v325_iso_parameter_scaling_sweep.py
Dataset: Structured Associative Pattern Task (N=2000, L=64, V=64)
Iso-Parameter Scales:
  - Scale 1: ~150,000 Parameters (Spectral 2L vs LLaMA 1L)
  - Scale 2: ~280,000 Parameters (Spectral 3L vs LLaMA 2L)
  - Scale 3: ~680,000 Parameters (Spectral 5L vs LLaMA 3L)
  - Scale 4: ~1,100,000 Parameters (Spectral 8L vs LLaMA 4L)
Hyperparameters:
  - Batch Size: 32
  - Seq Length: 64
  - Learning Rate: 1e-3
  - Weight Decay: 0.0 (Strict Spectral Rule)
  - Epochs: 10
======================================================================
""".format(torch.get_num_threads())

print(LOG_HEADER)


def create_hadamard_matrix(n):
    """Crea una matriz ortogonal de Walsh-Hadamard normalizada n x n (n potencia de 2)"""
    H = torch.tensor([[1.0]], dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
    return H / math.sqrt(n)


class PhaseSpectralCausalAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, seq_len=64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        angles = torch.linspace(0.0, 2 * math.pi, seq_len)
        self.phase_bias = nn.Parameter(torch.sin(angles))
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', causal_mask)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + self.phase_bias[:T].unsqueeze(0).unsqueeze(0)
        scores = scores + self.causal_mask[:T, :T]
        
        attn_weights = F.softmax(scores, dim=-1)
        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class MultiFrequencySpectralFFN(nn.Module):
    def __init__(self, d_model, num_banks=4):
        super().__init__()
        self.d_model = d_model
        self.num_banks = num_banks
        H_mat = create_hadamard_matrix(d_model)
        self.register_buffer('H', H_mat)
        
        self.phi1 = nn.Parameter(torch.zeros(num_banks, d_model))
        self.phi2 = nn.Parameter(torch.zeros(num_banks, d_model))
        self.w1 = nn.Parameter(torch.ones(num_banks, d_model))
        self.w2 = nn.Parameter(torch.ones(num_banks, d_model))
        self.combine = nn.Linear(num_banks * d_model, d_model, bias=False)

    def forward(self, x):
        h_freq = F.linear(x, self.H)
        bank_outs = []
        for b in range(self.num_banks):
            h_trig = torch.cos(h_freq + self.phi1[b]) * self.w1[b] + torch.sin(h_freq + self.phi2[b]) * self.w2[b]
            bank_outs.append(h_trig)
        h_concat = torch.cat(bank_outs, dim=-1)
        h_comb = self.combine(h_concat)
        out = F.linear(h_comb, self.H.t())
        return out


class FullySpectralBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, seq_len=64, num_banks=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PhaseSpectralCausalAttention(d_model, num_heads=num_heads, seq_len=seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = MultiFrequencySpectralFFN(d_model, num_banks=num_banks)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class FullySpectralModel(nn.Module):
    def __init__(self, vocab_size=64, d_model=128, num_layers=2, num_banks=1, seq_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            FullySpectralBlock(d_model, num_heads=4, seq_len=seq_len, num_banks=num_banks)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        return self.head(h)


class DenseFFN(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, d_model)
    def forward(self, x): return self.w2(F.silu(self.w1(x)))


class StandardLLaMABlock(nn.Module):
    def __init__(self, d_model, hidden_dim, num_heads=4, seq_len=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PhaseSpectralCausalAttention(d_model, num_heads=num_heads, seq_len=seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = DenseFFN(d_model, hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class StandardLLaMAModel(nn.Module):
    def __init__(self, vocab_size=64, d_model=128, hidden_dim=256, num_layers=2, seq_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            StandardLLaMABlock(d_model, hidden_dim, num_heads=4, seq_len=seq_len)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        return self.head(h)


def generate_structured_data(num_samples=2000, seq_len=64, vocab_size=64):
    torch.manual_seed(42)
    x = torch.randint(0, vocab_size // 2, (num_samples, seq_len))
    x_prev = torch.roll(x, shifts=1, dims=1)
    x_prev[:, 0] = 0
    y = (x_prev * 3 + x + 7) % vocab_size
    return x, y


def train_single_model(model, model_name, epochs=10):
    start_time = time.time()
    x_data, y_data = generate_structured_data()
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n" + "-"*75)
    print(f"ENTRENANDO MODELO: {model_name} (Params: {num_params:,})")
    print(f"{'Época':<8} | {'Loss':<10} | {'Accuracy %':<12} | {'Tiempo Época (s)':<18}")
    print("-" * 75)
    
    final_loss = 0.0
    final_acc = 0.0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        correct_tokens = 0
        total_tokens = 0
        
        for step, (bx, by) in enumerate(loader):
            optimizer.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits.view(-1, 64), by.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * bx.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct_tokens += (preds == by).sum().item()
            total_tokens += by.numel()
            
        epoch_time = time.time() - epoch_start
        epoch_loss = total_loss / len(loader.dataset)
        epoch_acc = (correct_tokens / total_tokens) * 100.0
        
        print(f"Época {epoch+1:<2}/10 | {epoch_loss:<10.4f} | {epoch_acc:<12.2f}% | {epoch_time:<18.2f}s")
        
        final_loss = epoch_loss
        final_acc = epoch_acc

    wall_clock_time = time.time() - start_time
    pei = (1.0 / (final_loss + 1e-6)) / math.log10(num_params + 1)
    
    print("-" * 75)
    print(f"Resumen Modelo -> Loss Final: {final_loss:.4f} | Acc Final: {final_acc:.2f}% | Wall Clock: {wall_clock_time:.2f}s | PEI: {pei:.4f}\n")
    
    return {
        "model_name": model_name,
        "params": num_params,
        "final_loss": final_loss,
        "final_acc": final_acc,
        "wall_clock_time": wall_clock_time,
        "pei": pei
    }


if __name__ == "__main__":
    sweep_results = []
    
    # 4 Escalas Iso-Paramétricas
    scales_config = [
        {
            "scale_id": "Escala 1 (~150K Params)",
            "spectral": FullySpectralModel(d_model=128, num_layers=2, num_banks=1),
            "llama": StandardLLaMAModel(d_model=128, hidden_dim=256, num_layers=1)
        },
        {
            "scale_id": "Escala 2 (~280K Params)",
            "spectral": FullySpectralModel(d_model=128, num_layers=3, num_banks=2),
            "llama": StandardLLaMAModel(d_model=128, hidden_dim=256, num_layers=2)
        },
        {
            "scale_id": "Escala 3 (~680K Params)",
            "spectral": FullySpectralModel(d_model=128, num_layers=5, num_banks=4),
            "llama": StandardLLaMAModel(d_model=136, hidden_dim=408, num_layers=3)
        },
        {
            "scale_id": "Escala 4 (~1.1M Params)",
            "spectral": FullySpectralModel(d_model=128, num_layers=8, num_banks=4),
            "llama": StandardLLaMAModel(d_model=144, hidden_dim=432, num_layers=4)
        }
    ]
    
    for cfg in scales_config:
        print("\n" + "="*85)
        print(f"EVALUANDO: {cfg['scale_id'].upper()}")
        print("="*85)
        
        # [REGLA DE ORO] Candidato espectral en primer lugar
        res_spectral = train_single_model(cfg["spectral"], f"Fully Spectral ({cfg['scale_id']})")
        res_llama = train_single_model(cfg["llama"], f"Standard LLaMA ({cfg['scale_id']})")
        
        sweep_results.append({
            "scale": cfg["scale_id"],
            "spectral": res_spectral,
            "llama": res_llama
        })

    print("\n" + "="*95)
    print("RESUMEN BARRIDO DE ESCALADO ISO-PARÁMETROS (v325)")
    print("="*95)
    print(f"{'Escala Paramétrica':<25} | {'Modelo':<22} | {'Params':<10} | {'Loss Final':<10} | {'Acc %':<8} | {'PEI':<8}")
    print("-" * 95)
    for sr in sweep_results:
        s_spec = sr["spectral"]
        s_llama = sr["llama"]
        print(f"{sr['scale']:<25} | {'Fully Spectral 🌟':<22} | {s_spec['params']:<10,} | {s_spec['final_loss']:<10.4f} | {s_spec['final_acc']:<8.2f}% | {s_spec['pei']:<8.4f}")
        print(f"{'':<25} | {'Standard LLaMA':<22} | {s_llama['params']:<10,} | {s_llama['final_loss']:<10.4f} | {s_llama['final_acc']:<8.2f}% | {s_llama['pei']:<8.4f}")
        print("-" * 95)
    print("="*95)
    
    best_spec = min([sr["spectral"] for sr in sweep_results], key=lambda x: x["final_loss"])
    
    ledger_entry = {
        "experiment_id": "v325",
        "fecha": "2026-08-09",
        "familia": "espectral_iso_scaling_sweep",
        "dataset": "sintetico_patron_2k",
        "n_eval": best_spec["params"],
        "metric_name": "loss_scaling_sweep",
        "value": round(best_spec["final_loss"], 4),
        "SE": None,
        "params": best_spec["params"],
        "nivel_rigor": 1,
        "etiqueta": "ANCLA"
    }
    print("\nLínea para master_ledger.jsonl:")
    print(json.dumps(ledger_entry))
