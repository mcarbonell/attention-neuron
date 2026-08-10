"""
v326 — Prototipo: Benchmark Comprensivo de Bases Espectrales (Configurable)
Línea de investigación: Spectral Architectures Research Line
"""

import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================================
# CONFIGURACIÓN EDITABLE DEL EXPERIMENTO (Modifica los valores aquí)
# ======================================================================
CONFIG = {
    "num_epochs": 25,          # Número de épocas de entrenamiento (ej: 10, 15, 20, 30)
    "learning_rate": 1e-3,     # Tasa de aprendizaje (Learning Rate)
    "num_samples": 2000,       # Número total de secuencias en el dataset
    "seq_len": 64,             # Longitud de secuencia L
    "d_model": 128,            # Dimensión del modelo d (potencia de 2 para FWHT/Haar)
    "num_layers": 5,           # Número de capas residuales espectrales
    "num_banks": 4,            # Banco multi-frecuencia K
    "batch_size": 32,          # Tamaño del batch
    "bases_to_run": [          # Bases ortogonales a evaluar
        "dct", 
        "fwht", 
        "dwt_haar", 
        "fft"
    ]
}

LOG_HEADER = """
======================================================================
[00:00:00] EXECUTION HEADER & TRACEABILITY (v326 - Configurable Spectral Basis)
======================================================================
Hardware: CPU / DirectML PyTorch
Threads: {}
Model File: scratch/prototype_v326_spectral_basis_comparison.py
Dataset: Structured Associative Pattern Task (N={}, L={}, V=64)
Editable Config:
  - Epochs: {}
  - Learning Rate: {}
  - d_model: {}
  - num_layers: {}
  - num_banks: {}
======================================================================
""".format(
    torch.get_num_threads(),
    CONFIG["num_samples"],
    CONFIG["seq_len"],
    CONFIG["num_epochs"],
    CONFIG["learning_rate"],
    CONFIG["d_model"],
    CONFIG["num_layers"],
    CONFIG["num_banks"]
)

print(LOG_HEADER)


# --- 1. Constructores de Matrices Espectrales Ortogonales ---

def create_hadamard_matrix(n):
    """Crea la matriz ortogonal de Walsh-Hadamard n x n"""
    H = torch.tensor([[1.0]], dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
    return H / math.sqrt(n)


def create_dct2_matrix(n):
    """Crea la matriz ortogonal de la Discrete Cosine Transform (DCT-II) n x n"""
    C = torch.zeros((n, n), dtype=torch.float32)
    for k in range(n):
        for i in range(n):
            if k == 0:
                C[k, i] = 1.0 / math.sqrt(n)
            else:
                C[k, i] = math.sqrt(2.0 / n) * math.cos(math.pi * k * (2 * i + 1) / (2.0 * n))
    return C


def create_haar_matrix(n):
    """Crea la matriz ortogonal de Ondículas de Haar (DWT) n x n (n potencia de 2)"""
    if n == 1:
        return torch.tensor([[1.0]], dtype=torch.float32)
    
    H_sub = create_haar_matrix(n // 2)
    low = torch.cat([H_sub, H_sub], dim=1) / math.sqrt(2)
    
    high = torch.zeros((n // 2, n), dtype=torch.float32)
    for i in range(n // 2):
        high[i, 2 * i] = 1.0 / math.sqrt(2)
        high[i, 2 * i + 1] = -1.0 / math.sqrt(2)
        
    return torch.cat([low, high], dim=0)


# --- 2. Bloques FFN Espectrales ---

class SpectralFFNBase(nn.Module):
    """Clase base para FFNs Espectrales con transformadas ortogonales"""
    def __init__(self, d_model, basis_type="fwht", num_banks=4):
        super().__init__()
        self.d_model = d_model
        self.basis_type = basis_type
        self.num_banks = num_banks
        
        if basis_type == "fwht":
            self.register_buffer('mat', create_hadamard_matrix(d_model))
        elif basis_type == "dct":
            self.register_buffer('mat', create_dct2_matrix(d_model))
        elif basis_type == "dwt_haar":
            self.register_buffer('mat', create_haar_matrix(d_model))
            
        if basis_type != "fft":
            self.phi1 = nn.Parameter(torch.zeros(num_banks, d_model))
            self.phi2 = nn.Parameter(torch.zeros(num_banks, d_model))
            self.w1 = nn.Parameter(torch.ones(num_banks, d_model))
            self.w2 = nn.Parameter(torch.ones(num_banks, d_model))
            self.combine = nn.Linear(num_banks * d_model, d_model, bias=False)
        else:
            freq_dim = d_model // 2 + 1
            self.w_complex_real = nn.Parameter(torch.ones(freq_dim))
            self.w_complex_imag = nn.Parameter(torch.zeros(freq_dim))
            self.b_complex_real = nn.Parameter(torch.zeros(freq_dim))
            self.b_complex_imag = nn.Parameter(torch.zeros(freq_dim))

    def forward(self, x):
        if self.basis_type in ["fwht", "dct", "dwt_haar"]:
            h_freq = F.linear(x, self.mat)
            bank_outs = []
            for b in range(self.num_banks):
                h_trig = torch.cos(h_freq + self.phi1[b]) * self.w1[b] + torch.sin(h_freq + self.phi2[b]) * self.w2[b]
                bank_outs.append(h_trig)
            h_concat = torch.cat(bank_outs, dim=-1)
            h_comb = self.combine(h_concat)
            out = F.linear(h_comb, self.mat.t())
            return out
        else:
            x_freq = torch.fft.rfft(x, norm="ortho")
            w_c = torch.complex(self.w_complex_real, self.w_complex_imag)
            b_c = torch.complex(self.b_complex_real, self.b_complex_imag)
            x_mod = torch.tanh(x_freq.real) * w_c.real + torch.tanh(x_freq.imag) * w_c.imag + b_c
            out = torch.fft.irfft(x_mod, n=self.d_model, norm="ortho")
            return out


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


class SpectralBasisBlock(nn.Module):
    def __init__(self, d_model, basis_type="fwht", num_heads=4, seq_len=64, num_banks=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PhaseSpectralCausalAttention(d_model, num_heads=num_heads, seq_len=seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SpectralFFNBase(d_model, basis_type=basis_type, num_banks=num_banks)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SpectralBasisModel(nn.Module):
    def __init__(self, vocab_size=64, d_model=128, basis_type="fwht", num_layers=5, num_banks=4, seq_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            SpectralBasisBlock(d_model, basis_type=basis_type, num_heads=4, seq_len=seq_len, num_banks=num_banks)
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


def train_basis_model(basis_type, basis_name, epochs=10):
    start_time = time.time()
    x_data, y_data = generate_structured_data(num_samples=CONFIG["num_samples"], seq_len=CONFIG["seq_len"])
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    
    model = SpectralBasisModel(
        vocab_size=64, 
        d_model=CONFIG["d_model"], 
        basis_type=basis_type, 
        num_layers=CONFIG["num_layers"], 
        num_banks=CONFIG["num_banks"],
        seq_len=CONFIG["seq_len"]
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=0.0)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n" + "-"*75)
    print(f"ENTRENANDO BASE ESPECTRAL: {basis_name} (Params: {num_params:,}, Épocas: {epochs})")
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
    print(f"Resumen Base -> Loss Final: {final_loss:.4f} | Acc Final: {final_acc:.2f}% | Wall Clock: {wall_clock_time:.2f}s | PEI: {pei:.4f}\n")
    
    return {
        "basis_type": basis_type,
        "basis_name": basis_name,
        "params": num_params,
        "final_loss": final_loss,
        "final_acc": final_acc,
        "wall_clock_time": wall_clock_time,
        "pei": pei
    }


if __name__ == "__main__":
    name_map = {
        "dct": "DCT-II (Discrete Cosine Real)",
        "fwht": "FWHT (Walsh-Hadamard Binary ±1)",
        "dwt_haar": "DWT Haar (Wavelet Multi-Resolución)",
        "fft": "FFT (Real Fast Fourier Complex)"
    }
    
    results = []
    
    for basis in CONFIG["bases_to_run"]:
        res = train_basis_model(basis, name_map.get(basis, basis), epochs=CONFIG["num_epochs"])
        results.append(res)
    
    print("\n" + "="*95)
    print(f"RESUMEN BENCHMARK ESPECTRAL CONFIGURABLE ({CONFIG['num_epochs']} ÉPOCAS)")
    print("="*95)
    print(f"{'Base Espectral':<35} | {'Params':<10} | {'Loss Final':<10} | {'Acc %':<8} | {'Wall Clock (s)':<14} | {'PEI':<8}")
    print("-" * 95)
    for r in results:
        print(f"{r['basis_name']:<35} | {r['params']:<10,} | {r['final_loss']:<10.4f} | {r['final_acc']:<8.2f}% | {r['wall_clock_time']:<14.2f} | {r['pei']:<8.4f}")
    print("="*95)
    
    best_res = min(results, key=lambda x: x["final_loss"])
    print(f"\n-> Ganador Absoluto: {best_res['basis_name']} (Loss: {best_res['final_loss']:.4f}, Acc: {best_res['final_acc']:.2f}%)")
