"""
scratch/prototype_v220_familiarity_atlas.py - Familiarity Atlas on MNIST

Experimento V220:
Detectar novedad mediante un "Atlas" de prototipos espectrales.
Si un numero es muy diferente a lo visto en entrenamiento, la red dispara 
una señal de "Incertidumbre Epistemica" (Baja Familiaridad).
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

def get_dct_matrix(n):
    """Genera una matriz DCT-II de tamaño n x n"""
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
    def __init__(self, n_features=64):
        self.n_features = n_features
        self.centroids = None # [10, n_features]
        self.dct_mat = get_dct_matrix(28).to(device)
        
    def get_signature(self, x):
        """Convierte imagen 28x28 a firma espectral (V+H Delta + DCT)"""
        # 1. Delta vertical y horizontal
        x_diff_v = torch.zeros_like(x)
        x_diff_v[1:] = x[1:] - x[:-1]
        
        x_diff_h = torch.zeros_like(x)
        x_diff_h[:, 1:] = x[:, 1:] - x[:, :-1]
        
        # 2. DCT 2D de ambas
        spec_v = self.dct_mat @ x_diff_v @ self.dct_mat.T
        spec_h = self.dct_mat @ x_diff_h @ self.dct_mat.T
        
        # 3. Cogemos los coeficientes 12x12 de ambas para mas detalle
        sig_v = spec_v[:12, :12].reshape(-1)
        sig_h = spec_h[:12, :12].reshape(-1)
        
        return torch.cat([sig_v, sig_h])

    def build_atlas(self, loader, n_samples=5000):
        print(f"Building Atlas from {n_samples} samples...")
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
        print("Atlas Built.")

    def get_familiarity(self, x):
        """Calcula la familiaridad (0 a 1) basandose en la distancia al centroide mas cercano"""
        sig = self.get_signature(x)
        # Distancia euclidea a todos los centroides
        dists = torch.norm(self.centroids - sig, dim=1)
        min_dist = torch.min(dists).item()
        
        # Escalar: Bajamos el denominador para que la diferencia de distancia sea mas punitiva
        # Probamos con 0.5 para maximizar el contraste
        familiarity = math.exp(-min_dist / 0.5)
        return familiarity, min_dist

# --- ENGINE ---
def run_v220():
    # 1. Preparar MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
    
    atlas = FamiliarityAtlas()
    atlas.build_atlas(train_loader, n_samples=5000)
    
    # 2. Test de Familiaridad
    print("\n--- TEST DE FAMILIARIDAD V220 ---")
    
    # Casos Normales
    normal_fams = []
    normal_dists = []
    for i in range(100):
        img, _ = test_set[i]
        fam, d = atlas.get_familiarity(img[0].to(device))
        normal_fams.append(fam)
        normal_dists.append(d)
    
    # Casos "Extraños" (Rotacion de 90 grados)
    strange_fams = []
    strange_dists = []
    for i in range(100):
        img, _ = test_set[i]
        img_rot = torch.rot90(img[0], 1, [0, 1])
        fam, d = atlas.get_familiarity(img_rot.to(device))
        strange_fams.append(fam)
        strange_dists.append(d)
        
    # Casos de Ruido Puro
    noise_fams = []
    noise_dists = []
    for i in range(100):
        img_noise = torch.rand(28, 28)
        fam, d = atlas.get_familiarity(img_noise.to(device))
        noise_fams.append(fam)
        noise_dists.append(d)

    print(f"Distancia Media (Normal):   {np.mean(normal_dists):.4f}")
    print(f"Distancia Media (Rotado):   {np.mean(strange_dists):.4f}")
    print(f"Distancia Media (Ruido):    {np.mean(noise_dists):.4f}")
    print("-" * 30)
    print(f"Familiaridad Media (MNIST Normal):   {np.mean(normal_fams):.4f}")
    print(f"Familiaridad Media (MNIST Rotado):   {np.mean(strange_fams):.4f}")
    print(f"Familiaridad Media (Ruido Puro):     {np.mean(noise_fams):.4f}")
    
    # Analisis
    ratio = np.mean(normal_fams) / (np.mean(strange_fams) + 1e-6)
    print(f"\nRatio de Discriminacion (Normal/Extraño): {ratio:.2f}x")
    
    if ratio > 2.0:
        print("\n[SUCCESS] El Atlas detecta correctamente la novedad estructural.")
    else:
        print("\n[FAILURE] El Atlas no discrimina lo suficiente. Ajustar escala de distancia.")

if __name__ == "__main__":
    run_v220()
