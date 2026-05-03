"""
scratch/prototype_v221_safe_classifier.py - Safe Attention Classifier

Experimento V221:
Integrar el Familiarity Atlas (V220) como un "Muro de Seguridad".
La red solo clasifica si el input es estructuralmente familiar.
Si no, se abstiene (Refusal).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
import math
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- ATLAS (Reutilizado de V220) ---
def get_dct_matrix(n):
    matrix = np.zeros((n, n))
    for k in range(n):
        for i in range(n):
            matrix[k, i] = np.cos(np.pi * k * (2 * i + 1) / (2 * n))
        if k == 0:
            matrix[k, :] *= np.sqrt(1 / n)
        else:
            matrix[k, :] *= np.sqrt(2 / n)
    return torch.from_numpy(matrix).float()

class FamiliarityAtlas:
    def __init__(self):
        self.centroids = None
        self.dct_mat = get_dct_matrix(28).to(device)
        
    def get_signature(self, x):
        x_diff_v = torch.zeros_like(x)
        x_diff_v[1:] = x[1:] - x[:-1]
        x_diff_h = torch.zeros_like(x)
        x_diff_h[:, 1:] = x[:, 1:] - x[:, :-1]
        spec_v = self.dct_mat @ x_diff_v @ self.dct_mat.T
        spec_h = self.dct_mat @ x_diff_h @ self.dct_mat.T
        sig_v = spec_v[:12, :12].reshape(-1)
        sig_h = spec_h[:12, :12].reshape(-1)
        return torch.cat([sig_v, sig_h])

    def build_atlas(self, loader, n_samples=5000):
        class_signatures = [[] for _ in range(10)]
        count = 0
        for imgs, labels in loader:
            imgs = imgs.to(device)
            for i in range(imgs.size(0)):
                sig = self.get_signature(imgs[i, 0])
                class_signatures[labels[i].item()].append(sig)
                count += 1
            if count >= n_samples: break
        self.centroids = torch.stack([torch.stack(sigs).mean(dim=0) for sigs in class_signatures])

    def get_distance(self, x):
        sig = self.get_signature(x)
        dists = torch.norm(self.centroids - sig, dim=1)
        return torch.min(dists).item()

# --- CLASIFICADOR ---
class AttentionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Usamos una red muy pequeña para enfatizar el PEI
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x): return self.net(x)

# --- SAFE SYSTEM ---
class SafeSystem:
    def __init__(self, classifier, atlas, threshold=4.8):
        self.classifier = classifier
        self.atlas = atlas
        self.threshold = threshold # Distancia maxima permitida
        
    def predict(self, x):
        """Inferencia Protegida"""
        dist = self.atlas.get_distance(x)
        if dist > self.threshold:
            return -1, dist # ABSTENCION (Novedad detectada)
        
        with torch.no_grad():
            logits = self.classifier(x.unsqueeze(0).unsqueeze(0))
            pred = torch.argmax(logits, dim=1).item()
        return pred, dist

# --- ENGINE ---
def run_v221():
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
    
    # 1. Entrenar Clasificador
    print("Entrenando Clasificador base...")
    model = AttentionClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    for epoch in range(3): # Entrenamiento rapido para el prototipo
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(imgs), labels)
            loss.backward()
            optimizer.step()
    
    # 2. Construir Atlas
    atlas = FamiliarityAtlas()
    atlas.build_atlas(train_loader, n_samples=5000)
    
    # 3. Sistema Seguro (Umbral 4.8 basado en V220)
    safe_sys = SafeSystem(model, atlas, threshold=4.8)
    
    # 4. Benchmark de Seguridad
    print("\n--- BENCHMARK DE SEGURIDAD V221 ---")
    
    results = {"clean": [], "rotated": [], "noise": []}
    
    # Test Clean
    correct = 0
    total_clean = 200
    abstained_clean = 0
    for i in range(total_clean):
        img, label = test_set[i]
        pred, dist = safe_sys.predict(img[0].to(device))
        if pred == -1: abstained_clean += 1
        elif pred == label: correct += 1
    
    # Test Rotated
    total_rot = 200
    abstained_rot = 0
    for i in range(total_rot):
        img, _ = test_set[i]
        img_rot = torch.rot90(img[0], 1, [0, 1])
        pred, dist = safe_sys.predict(img_rot.to(device))
        if pred == -1: abstained_rot += 1
        
    # Test Noise
    total_noise = 200
    abstained_noise = 0
    for i in range(total_noise):
        img_noise = torch.rand(28, 28)
        pred, dist = safe_sys.predict(img_noise.to(device))
        if pred == -1: abstained_noise += 1

    print(f"Dataset Normal  | Abstenciones: {abstained_clean/total_clean:>4.1%} | Precision Filtrada: {correct/(total_clean-abstained_clean):.1%}")
    print(f"Dataset Rotado  | Abstenciones: {abstained_rot/total_rot:>4.1%} | (Exito en rechazo)")
    print(f"Dataset Ruido   | Abstenciones: {abstained_noise/total_noise:>4.1%} | (Exito en rechazo)")
    
    print("\nConclusion:")
    if abstained_rot/total_rot > 0.8 and abstained_clean/total_clean < 0.2:
        print("[SUCCESS] El sistema es Robusto: Filtra lo desconocido y acepta lo familiar.")
    else:
        print("[LIMITATION] El umbral de seguridad necesita ajuste fino.")

if __name__ == "__main__":
    run_v221()
