"""
scratch/prototype_v201_resonant_neurons.py — Resonant Phase Interaction

Experimento de frontera (V201):
¿Podemos sustituir la suma ponderada tradicional por una INTERFERENCIA DE FASES?
Inspirado en la biología (Spiking Neurons) y la física de ondas.

Concepto:
1. Input: Codificado como una fase (ángulo entre 0 y 2pi).
2. Pesos: Son sintonizadores de fase (phase-offsets).
3. Activación: Interferencia Constructiva. La neurona dispara si la señal resuena con su sintonía.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import json

# --- Protocolo de Métricas (Mario Standard) ---
class ResearchLogger:
    def __init__(self, experiment_name):
        self.name = experiment_name
        self.start_time = time.time()
        self.results = {}

    def log(self, metrics):
        self.results.update(metrics)
        self.results['wall_clock_time'] = time.time() - self.start_time
        print(f"\n📊 [{self.name}] Hallazgos: {json.dumps(metrics, indent=2)}")

# --- Arquitectura de Resonancia ---

class ResonantLayer(nn.Module):
    """
    Capa de Resonancia de Fase.
    En lugar de y = Wx + b, calcula y = cos(x_phase - w_phase).
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Sintonizadores de fase (0 a 2pi)
        self.phase_sintonizer = nn.Parameter(torch.rand(out_features, in_features) * 2 * math.pi)
        # Ganancia (amplitud)
        self.magnitude = nn.Parameter(torch.ones(out_features, in_features))

    def forward(self, x_phase):
        # x_phase: (batch, in_features)
        
        # Calculamos la diferencia de fase (interferencia)
        # (batch, 1, in_features) - (1, out, in_features)
        diff = x_phase.unsqueeze(1) - self.phase_sintonizer.unsqueeze(0)
        
        # La respuesta es la coherencia de fase (Interferencia constructiva)
        # cos(diff) es 1 si las fases coinciden, -1 si están en oposición
        coherence = torch.cos(diff) * self.magnitude
        
        # Sumamos las resonancias de todos los inputs
        resonant_sum = coherence.sum(dim=-1)
        
        # Activación Lorentiziana (Sintonizador de Radio)
        # Actúa como una ReLU pero enfocada en la nitidez de la frecuencia
        return F.tanh(resonant_sum)

# --- Experimento: El Problema de la Coherencia Lógica ---

def run_resonance_experiment():
    logger = ResearchLogger("V201-Phase-Resonance")
    device = torch.device("cpu")
    
    # 1. Crear Dataset: XOR basado en Fase
    # 0 -> Fase 0, 1 -> Fase PI
    X = torch.tensor([
        [0.0, 0.0],
        [0.0, math.pi],
        [math.pi, 0.0],
        [math.pi, math.pi]
    ], device=device)
    
    Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], device=device) # XOR target

    # 2. Modelo Resonante vs MLP Tradicional
    model = nn.Sequential(
        ResonantLayer(2, 4),
        nn.Linear(4, 1)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 3. Entrenamiento
    print("🎬 Entrenando Neuronas de Resonancia...")
    t0 = time.time()
    
    for epoch in range(500):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"   Época {epoch} | Loss: {loss.item():.6f}")

    eval_time = time.time() - t0
    
    # 4. Resultados
    with torch.no_grad():
        preds = model(X)
        final_loss = criterion(preds, Y).item()
        accuracy = ((preds > 0.5) == Y).float().mean().item()

    logger.log({
        "final_objective": final_loss,
        "accuracy": accuracy,
        "function_evaluation_time": eval_time,
        "params": sum(p.numel() for p in model.parameters())
    })

    print("\n🔍 Verificación de Predicciones:")
    for i in range(4):
        print(f"   In: {X[i].tolist()} | Target: {Y[i].item()} | Pred: {preds[i].item():.4f}")

if __name__ == "__main__":
    run_resonance_experiment()
