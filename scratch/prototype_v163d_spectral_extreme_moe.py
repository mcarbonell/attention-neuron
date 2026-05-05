import torch
import torch.nn.functional as F
import numpy as np
import time
import os
import json

# --- CONFIGURACIÓN DE DISPOSITIVO ---
device = torch.device('cpu')
try:
    import torch_directml
    device = torch_directml.device()
    print(f"Using DirectML device: {device}")
except ImportError:
    print("torch-directml not found, using CPU")

# --- TRANSFORMADA DE WALSH-HADAMARD RÁPIDA (FWHT) ---
def fwht(x):
    b, n = x.shape
    res = x.clone()
    h = 1
    while h < n:
        res = res.view(b, n // (2 * h), 2, h)
        a, b_ = res[:, :, 0, :], res[:, :, 1, :]
        res = torch.cat([a + b_, a - b_], dim=2)
        h *= 2
    return res.view(b, n) / (n ** 0.5)

# --- BLOQUE MOE ESPECTRAL EXTREMO ---
class ExtremeSpectralMoE(torch.nn.Module):
    def __init__(self, dim, num_experts=131072, top_k=16):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # El "Cerebro" del modelo: 131,072 firmas espectrales
        # Esto ocupa ~512 MB de VRAM (131072 * 1024 * 4 bytes)
        print(f"Inicializando Banco de Expertos: {num_experts:,} firmas...")
        # Usamos torch.empty + normal_ para ahorrar un poco de memoria temporal durante init
        self.expert_signatures = torch.nn.Parameter(torch.empty(num_experts, dim))
        torch.nn.init.normal_(self.expert_signatures, std=0.02)
        
        # Matriz de salida simplificada para el test
        self.output_projection = torch.nn.Parameter(torch.randn(num_experts, 10) * 0.01)

    def forward(self, x):
        # 1. Entrada -> Dominio Espectral
        x_spec = fwht(x)
        
        # 2. Gating: Búsqueda de Resonancia Masiva
        # (X @ E^T) -> (Batch, 131072)
        scores = torch.matmul(x_spec, self.expert_signatures.t())
        
        # 3. Selección Sparse (Top-K)
        top_scores, top_indices = torch.topk(scores, k=self.top_k, dim=1)
        
        # 4. Activación Softmax sobre los elegidos
        weights = F.softmax(top_scores, dim=1)
        
        # 5. Síntesis: Combinar votos de expertos
        # Extraemos solo los expertos necesarios (Gathering)
        selected_outputs = self.output_projection[top_indices] # (Batch, TopK, 10)
        final_output = (selected_outputs * weights.unsqueeze(-1)).sum(dim=1)
        
        return final_output

def run_extreme_moe_test():
    DIM = 1024
    NUM_EXPERTS = 131072 
    
    print(f"\n--- EXPERIMENTO V163d: EXTREME SPECTRAL-MoE (131K EXPERTOS) ---")
    
    t_init = time.perf_counter()
    model = ExtremeSpectralMoE(DIM, num_experts=NUM_EXPERTS).to(device)
    dt_init = time.perf_counter() - t_init
    print(f"Modelo inicializado en {dt_init:.2f}s")
    
    # Simulamos un batch de 32 tokens
    dummy_input = torch.randn(32, DIM).to(device)
    
    print(f"\nEjecutando Inferencia sobre 32 tokens con {NUM_EXPERTS:,} expertos...")
    
    # Benchmark de velocidad
    t_start = time.perf_counter()
    with torch.no_grad():
        # Hacemos unas cuantas pasadas para promediar
        for _ in range(5):
            output = model(dummy_input)
    dt = (time.perf_counter() - t_start) / 5 * 1000 # ms por batch
    
    print("\n" + "="*60)
    print(f"RESULTADO EXTREME SPECTRAL-MoE (V163d)")
    print(f"="*60)
    print(f"Total Expertos:     {NUM_EXPERTS:,}")
    print(f"Expertos Activos:   {model.top_k} (Sparsity: {model.top_k/NUM_EXPERTS*100:.4f}%)")
    print(f"Latencia por Batch: {dt:.2f} ms")
    print(f"Tokens/Segundo:     {32 / (dt/1000):,.0f} tokens/s")
    print(f"Memoria de Pesos:   {(NUM_EXPERTS * DIM * 4 + NUM_EXPERTS * 10 * 4) / 1024**2:.1f} MB")
    print("="*60)

    # Guardar resultados
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v163d_extreme_moe.json", "w") as f:
        json.dump({
            "num_experts": NUM_EXPERTS,
            "latency_ms": dt,
            "memory_mb": (NUM_EXPERTS * DIM * 4) / 1024**2
        }, f, indent=4)

if __name__ == "__main__":
    run_extreme_moe_test()
