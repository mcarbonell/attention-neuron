import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
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
    """ Fast Walsh-Hadamard Transform vectorizada """
    b, n = x.shape
    res = x.clone()
    h = 1
    while h < n:
        res = res.view(b, n // (2 * h), 2, h)
        a, b_ = res[:, :, 0, :], res[:, :, 1, :]
        res = torch.cat([a + b_, a - b_], dim=2)
        h *= 2
    return res.view(b, n) / (n ** 0.5)

# --- MODELO REFINADO ---
class RefinedSpectralPAC(nn.Module):
    def __init__(self, archetypes, labels):
        super().__init__()
        # Los arquetipos de PAC se convierten en pesos entrenables
        self.weights = nn.Parameter(archetypes.clone())
        self.register_buffer('labels', labels.clone())
        
    def forward(self, x_spec):
        # Normalización para Similitud Coseno
        norm_x = F.normalize(x_spec, p=2, dim=1)
        norm_w = F.normalize(self.weights, p=2, dim=1)
        
        # Matriz de similitudes (Batch, Num_Archetypes)
        similarities = torch.mm(norm_x, norm_w.t())
        
        # Proyección a clases (10 logits)
        # Logit de clase C = máxima similitud encontrada entre sus arquetipos
        batch_size = x_spec.size(0)
        logits = torch.full((batch_size, 10), -1.0).to(x_spec.device)
        
        for c in range(10):
            mask = (self.labels == c)
            if mask.any():
                # Obtenemos las similitudes de todos los arquetipos de la clase C
                class_sims = similarities[:, mask]
                # El logit es la mejor coincidencia (Max Pooling sobre arquetipos)
                logits[:, c] = torch.max(class_sims, dim=1)[0]
        
        # Escalado de temperatura para ayudar a la convergencia de CrossEntropy
        return logits * 15.0 

def run_experiment():
    print(f"\n--- EXPERIMENTO V142: GRADIENT-REFINED SPECTRAL PAC ---")
    
    # 1. CARGA DE DATOS
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=60000)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)
    
    data_train, targets_train = next(iter(train_loader))
    targets_train = targets_train.to(device)
    
    print("Pre-procesando Walsh...")
    def to_walsh(d):
        flat = d.view(d.size(0), -1)
        padded = torch.zeros(flat.size(0), 1024).to(device)
        padded[:, :784] = flat.to(device)
        return fwht(padded)
    
    spec_train = to_walsh(data_train)

    # 2. FASE PAC (Descubrimiento de Taxonomía - 8 Generaciones)
    print("Iniciando Fase PAC para descubrir arquetipos iniciales...")
    image_cluster_assignment = targets_train.clone()
    cluster_labels = {d: d for d in range(10)}
    next_id = 10
    
    for gen in range(8):
        ids = torch.unique(image_cluster_assignment)
        archs = torch.stack([spec_train[image_cluster_assignment == cid].mean(0) for cid in ids])
        labs = torch.tensor([cluster_labels[cid.item()] for cid in ids], device=device)
        
        sims = torch.mm(F.normalize(spec_train, p=2, dim=1), F.normalize(archs, p=2, dim=1).t())
        max_sim, best_idx = torch.max(sims, dim=1)
        correct = (labs[best_idx] == targets_train)
        
        # Purificación
        image_cluster_assignment[correct] = ids[best_idx[correct]]
        
        # Bifurcación simple para este experimento
        if gen < 7:
            for d in range(10):
                err = (~correct) & (targets_train == d)
                if err.any():
                    image_cluster_assignment[err] = next_id
                    cluster_labels[next_id] = d
                    next_id += 1
        
        acc = correct.float().mean().item()
        print(f"Gen {gen} | Archetypes: {len(ids):4d} | Acc: {acc*100:.2f}%")

    # 3. FASE DE REFINAMIENTO POR GRADIENTE
    print(f"\nConvirtiendo {len(ids)} arquetipos en parámetros entrenables...")
    model = RefinedSpectralPAC(archs, labs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Entrenando refinamiento (5 épocas)...")
    model.train()
    for epoch in range(5):
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(spec_train, targets_train), batch_size=256, shuffle=True)
        total_loss = 0
        for b_x, b_y in loader:
            optimizer.zero_grad()
            logits = model(b_x)
            loss = F.cross_entropy(logits, b_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Época {epoch+1} | Loss: {total_loss/len(loader):.4f}")

    # 4. EVALUACIÓN FINAL
    print("\nEvaluando en el set de test (10,000 imágenes)...")
    model.eval()
    correct_test = 0
    with torch.no_grad():
        for data, target in test_loader:
            spec_x = to_walsh(data)
            preds = torch.argmax(model(spec_x), dim=1)
            correct_test += preds.eq(target.to(device)).sum().item()
            
    final_acc = correct_test / 10000
    print("\n" + "="*55)
    print(f"RESULTADOS REFINAMIENTO GRADIENTE (V142)")
    print(f"="*55)
    print(f"Precisión Final Test: {final_acc*100:.2f}%")
    print(f"Arquetipos Utilizados: {len(ids)}")
    print(f"Método: PAC Discovery + Spectral Polish")
    print("="*55)

    # Guardar resultados
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v142_refined_spectral_pac.json", "w") as f:
        json.dump({"accuracy": final_acc, "num_archetypes": len(ids)}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
