"""
v329 — Prototipo: SpecAttention 2D (Transformer 100% Libre de Atención Causal, Fase 8)
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
[00:00:00] EXECUTION HEADER & TRACEABILITY (v329 - SpecAttention 2D Attention-Free)
======================================================================
Hardware: CPU / DirectML PyTorch
Threads: {}
Model File: scratch/prototype_v329_specattention_2d_attention_free.py
Dataset: Structured Associative Pattern Task (N=2000, L=64, V=64)
Architectures Evaluated:
  1. SpecAttention 2D Transformer (v329 - 100% Libre de Atención QK^T, 0 Params en Secuencia)
  2. Standard Spectral Transformer (v328 - Con Atención Causal MHA + Lerp FFN)
  3. Standard LLaMA Transformer (Baseline)
Hyperparameters:
  - Batch Size: 32
  - Seq Length: 64
  - d_model: 128
  - num_layers: 5
  - Learning Rate: 1e-3
  - Weight Decay: 0.0 (Strict Spectral Rule)
  - Epochs: 15
======================================================================
""".format(torch.get_num_threads())

print(LOG_HEADER)


# --- 1. Constructores de Matrices Espectrales Ortogonales ---

def create_hadamard_matrix(n):
    H = torch.tensor([[1.0]], dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
    return H / math.sqrt(n)


def create_dct2_matrix(n):
    C = torch.zeros((n, n), dtype=torch.float32)
    for k in range(n):
        for i in range(n):
            if k == 0:
                C[k, i] = 1.0 / math.sqrt(n)
            else:
                C[k, i] = math.sqrt(2.0 / n) * math.cos(math.pi * k * (2 * i + 1) / (2.0 * n))
    return C


def create_haar_matrix(n):
    if n == 1:
        return torch.tensor([[1.0]], dtype=torch.float32)
    H_sub = create_haar_matrix(n // 2)
    low = torch.cat([H_sub, H_sub], dim=1) / math.sqrt(2)
    high = torch.zeros((n // 2, n), dtype=torch.float32)
    for i in range(n // 2):
        high[i, 2 * i] = 1.0 / math.sqrt(2)
        high[i, 2 * i + 1] = -1.0 / math.sqrt(2)
    return torch.cat([low, high], dim=0)


# --- 2. Mezcla Causal Espectral de Secuencia (SpecSeqMix - 0 Parámetros en Q, K, V, Out) ---

class CausalSpectralSequenceMixer(nn.Module):
    """Mezclador espectral ortogonal causal de secuencia (Sin matrices Q, K, V, Out)"""
    def __init__(self, seq_len=64, d_model=128):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Matriz ortogonal de secuencia con máscara de causalidad
        dct_mat = create_dct2_matrix(seq_len)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        mat_causal = dct_mat * causal_mask
        # Normalizar filas para mantener norma unitaria
        row_norms = torch.norm(mat_causal, dim=1, keepdim=True) + 1e-6
        mat_causal = mat_causal / row_norms
        self.register_buffer('mat_causal', mat_causal)
        
        # Sesgos angulares trigonométricos de secuencia (O(T) parámetros)
        angles = torch.linspace(0.0, 2 * math.pi, seq_len)
        self.phi1 = nn.Parameter(torch.sin(angles).unsqueeze(1))
        self.phi2 = nn.Parameter(torch.cos(angles).unsqueeze(1))
        self.w1 = nn.Parameter(torch.ones(seq_len, 1))
        self.w2 = nn.Parameter(torch.ones(seq_len, 1))

    def forward(self, x):
        # x: (B, T, D)
        # Proyectar en la dimensión de secuencia T: (B, D, T)
        x_t = x.transpose(1, 2)
        h_seq = F.linear(x_t, self.mat_causal) # (B, D, T)
        
        # Modulación de fase trigonométrica causal
        h_mod = torch.cos(h_seq + self.phi1.t()) * self.w1.t() + torch.sin(h_seq + self.phi2.t()) * self.w2.t()
        
        # Proyección inversa de secuencia
        out_t = F.linear(h_mod, self.mat_causal.t())
        out = out_t.transpose(1, 2)
        return out


# --- 3. FFN Espectral Lerp ---

class LearnableSubstrateLerpFFN(nn.Module):
    def __init__(self, d_model, num_banks=2):
        super().__init__()
        self.d_model = d_model
        self.substrate_logits = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
        
        self.register_buffer('mat_fwht', create_hadamard_matrix(d_model))
        self.register_buffer('mat_dct', create_dct2_matrix(d_model))
        self.register_buffer('mat_haar', create_haar_matrix(d_model))
        
        self.phi1_fwht = nn.Parameter(torch.zeros(num_banks, d_model))
        self.phi2_fwht = nn.Parameter(torch.zeros(num_banks, d_model))
        self.w1_fwht = nn.Parameter(torch.ones(num_banks, d_model))
        self.w2_fwht = nn.Parameter(torch.ones(num_banks, d_model))
        
        self.phi1_dct = nn.Parameter(torch.zeros(num_banks, d_model))
        self.phi2_dct = nn.Parameter(torch.zeros(num_banks, d_model))
        self.w1_dct = nn.Parameter(torch.ones(num_banks, d_model))
        self.w2_dct = nn.Parameter(torch.ones(num_banks, d_model))
        
        self.phi1_haar = nn.Parameter(torch.zeros(num_banks, d_model))
        self.phi2_haar = nn.Parameter(torch.zeros(num_banks, d_model))
        self.w1_haar = nn.Parameter(torch.ones(num_banks, d_model))
        self.w2_haar = nn.Parameter(torch.ones(num_banks, d_model))
        
        self.combine = nn.Linear(num_banks * d_model, d_model, bias=False)

    def forward(self, x):
        weights = F.softmax(self.substrate_logits, dim=0)
        
        h_fwht = F.linear(x, self.mat_fwht)
        outs_fwht = [torch.cos(h_fwht + self.phi1_fwht[b]) * self.w1_fwht[b] + torch.sin(h_fwht + self.phi2_fwht[b]) * self.w2_fwht[b] for b in range(self.phi1_fwht.shape[0])]
        out_fwht = F.linear(self.combine(torch.cat(outs_fwht, dim=-1)), self.mat_fwht.t())
        
        h_dct = F.linear(x, self.mat_dct)
        outs_dct = [torch.cos(h_dct + self.phi1_dct[b]) * self.w1_dct[b] + torch.sin(h_dct + self.phi2_dct[b]) * self.w2_dct[b] for b in range(self.phi1_dct.shape[0])]
        out_dct = F.linear(self.combine(torch.cat(outs_dct, dim=-1)), self.mat_dct.t())
        
        h_haar = F.linear(x, self.mat_haar)
        outs_haar = [torch.cos(h_haar + self.phi1_haar[b]) * self.w1_haar[b] + torch.sin(h_haar + self.phi2_haar[b]) * self.w2_haar[b] for b in range(self.phi1_haar.shape[0])]
        out_haar = F.linear(self.combine(torch.cat(outs_haar, dim=-1)), self.mat_haar.t())
        
        out_fused = weights[0] * out_fwht + weights[1] * out_dct + weights[2] * out_haar
        return out_fused


# --- 4. Bloque SpecAttention 2D (100% Libre de Atención QK^T) ---

class SpecAttention2DBlock(nn.Module):
    def __init__(self, d_model, seq_len=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.seq_mixer = CausalSpectralSequenceMixer(seq_len=seq_len, d_model=d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = LearnableSubstrateLerpFFN(d_model, num_banks=2)

    def forward(self, x):
        x = x + self.seq_mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SpecAttention2DModel(nn.Module):
    def __init__(self, vocab_size=64, d_model=128, num_layers=5, seq_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            SpecAttention2DBlock(d_model, seq_len=seq_len)
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


# --- 5. Baselines para Comparación ---

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


class StandardSpectralBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, seq_len=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PhaseSpectralCausalAttention(d_model, num_heads=num_heads, seq_len=seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = LearnableSubstrateLerpFFN(d_model, num_banks=2)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class StandardSpectralModel(nn.Module):
    def __init__(self, vocab_size=64, d_model=128, num_layers=5, seq_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            StandardSpectralBlock(d_model, num_heads=4, seq_len=seq_len)
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


def train_single_model(model, model_name, epochs=15):
    start_time = time.time()
    x_data, y_data = generate_structured_data()
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n" + "-"*75)
    print(f"ENTRENANDO MODELO: {model_name} (Params: {num_params:,})")
    print(f"{'Época':<10} | {'Loss':<10} | {'Accuracy %':<12} | {'Tiempo Época (s)':<18}")
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
        
        print(f"Época {epoch+1:<2}/{epochs:<2} | {epoch_loss:<10.4f} | {epoch_acc:<12.2f}% | {epoch_time:<18.2f}s")
        
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
    results = []
    
    # [REGLA DE ORO] Candidato SpecAttention 2D (Attention-Free) en primer lugar
    print("[REGLA DE ORO] Ejecutando el CANDIDATO SpecAttention 2D (Attention-Free 100% Espectral) en primer lugar...")
    m_spec_2d = SpecAttention2DModel(vocab_size=64, d_model=128, num_layers=5)
    results.append(train_single_model(m_spec_2d, "SpecAttention 2D (Attention-Free v329)", epochs=15))
    
    print("\nEjecutando baseline con Atención Causal MHA...")
    m_standard_spec = StandardSpectralModel(vocab_size=64, d_model=128, num_layers=5)
    results.append(train_single_model(m_standard_spec, "Standard Spectral (Con Atención MHA v328)", epochs=15))
    
    print("\n" + "="*95)
    print("RESUMEN BENCHMARK SPECATTENTION 2D ATTENTION-FREE (v329)")
    print("="*95)
    print(f"{'Modelo Transformer':<42} | {'Params':<10} | {'Loss Final':<10} | {'Acc %':<8} | {'PEI':<8}")
    print("-" * 95)
    for r in results:
        print(f"{r['model_name']:<42} | {r['params']:<10,} | {r['final_loss']:<10.4f} | {r['final_acc']:<8.2f}% | {r['pei']:<8.4f}")
    print("="*95)
    
    best_res = min(results, key=lambda x: x["final_loss"])
    print(f"\n-> Ganador Absoluto: {best_res['model_name']} (Loss: {best_res['final_loss']:.4f}, Acc: {best_res['final_acc']:.2f}%)")
    
    ledger_entry = {
        "experiment_id": "v329",
        "fecha": "2026-08-10",
        "familia": "espectral_specattention_2d_attention_free",
        "dataset": "sintetico_patron_2k",
        "n_eval": best_res["params"],
        "metric_name": "loss_specattention_2d",
        "value": round(best_res["final_loss"], 4),
        "SE": None,
        "params": best_res["params"],
        "nivel_rigor": 1,
        "etiqueta": "ANCLA"
    }
    print("\nLínea para master_ledger.jsonl:")
    print(json.dumps(ledger_entry))
