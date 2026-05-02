"""
scratch/prototype_v202_resonant_neurons.py — Resonant Phase Interaction Fix

Experimento de frontera (V202):
Se mejora la neurona de resonancia para resolver el XOR correctamente.
Se añaden no-linealidades más adecuadas y un loss de BCE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import json

class ResearchLogger:
    def __init__(self, experiment_name):
        self.name = experiment_name
        self.start_time = time.time()
        self.results = {}

    def log(self, metrics):
        self.results.update(metrics)
        self.results['wall_clock_time'] = time.time() - self.start_time
        print(f"\n📊 [{self.name}] Hallazgos: {json.dumps(metrics, indent=2)}")

class ResonantLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Sintonizadores de fase (0 a 2pi)
        self.phase_sintonizer = nn.Parameter(torch.rand(out_features, in_features) * 2 * math.pi)
        # Ganancia (amplitud)
        self.magnitude = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x_phase):
        diff = x_phase.unsqueeze(1) - self.phase_sintonizer.unsqueeze(0)
        coherence = torch.cos(diff) * self.magnitude
        resonant_sum = coherence.sum(dim=-1)
        # ReLU permite que solo las resonancias positivas pasen
        return F.relu(resonant_sum)

def run_resonance_experiment():
    logger = ResearchLogger("V202-Phase-Resonance")
    device = torch.device("cpu")
    
    X = torch.tensor([
        [0.0, 0.0],
        [0.0, math.pi],
        [math.pi, 0.0],
        [math.pi, math.pi]
    ], device=device)
    
    Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], device=device)

    model = nn.Sequential(
        ResonantLayer(2, 8),
        nn.Linear(8, 1),
        nn.Sigmoid()
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.BCELoss()

    print("🎬 Entrenando Neuronas de Resonancia...")
    t0 = time.time()
    
    for epoch in range(1001):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
        
        if epoch < 5:
            print(f"   Batch {epoch} | Loss: {loss.item():.6f}")
        elif epoch % 100 == 0:
            print(f"   Época {epoch} | Loss: {loss.item():.6f}")

    eval_time = time.time() - t0
    
    with torch.no_grad():
        preds = model(X)
        final_loss = criterion(preds, Y).item()
        accuracy = ((preds > 0.5) == Y).float().mean().item()

    logger.log({
        "final_objective": final_loss,
        "accuracy": accuracy,
        "function_evaluation_time": eval_time,
        "internal_overhead_time": 0.0,
        "params": sum(p.numel() for p in model.parameters())
    })

    print("\n🔍 Verificación de Predicciones:")
    for i in range(4):
        print(f"   In: {X[i].tolist()} | Target: {Y[i].item()} | Pred: {preds[i].item():.4f}")

if __name__ == "__main__":
    run_resonance_experiment()
