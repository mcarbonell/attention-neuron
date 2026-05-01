import torch
import torch.nn.functional as F
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

# --- MOTOR DE AUMENTACIÓN SINTÉTICA ---
def augment_image(img, angle=0, scale=1.0):
    """
    Aplica rotación y escalado a un tensor (1, 1, 28, 28) usando afinidad.
    """
    B, C, H, W = img.shape
    device = img.device
    angle_rad = angle * np.pi / 180.0
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    s = 1.0 / scale
    theta = torch.tensor([[[s*cos_a, s*sin_a, 0], [-s*sin_a, s*cos_a, 0]]], device=device).float()
    grid = F.affine_grid(theta, img.size(), align_corners=False)
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

# --- EXTRACTOR HÍBRIDO (WALSH + ISLAS) ---
def get_hybrid_vector(img_batch, batch_size=5000):
    results = []
    for i in range(0, img_batch.size(0), batch_size):
        batch = img_batch[i : i+batch_size].to(device)
        flat = batch.view(batch.size(0), -1)
        padded = torch.zeros(flat.size(0), 1024).to(device)
        padded[:, :784] = flat
        w = F.normalize(fwht(padded), p=2, dim=1)
        binary = (batch > 0.1).float()
        padded_h = F.pad(binary, (1, 0), value=0)
        diff_h = padded_h[:, :, :, 1:] - padded_h[:, :, :, :-1]
        islands_h = (diff_h == 1).float().sum(dim=3).squeeze(1)
        padded_v = F.pad(binary, (0, 0, 1, 0), value=0)
        diff_v = padded_v[:, :, 1:, :] - padded_v[:, :, :-1, :]
        islands_v = (diff_v == 1).float().sum(dim=2).squeeze(1)
        isl = F.normalize(torch.cat([islands_h, islands_v], dim=1), p=2, dim=1)
        results.append(torch.cat([w, isl], dim=1))
    return torch.cat(results, dim=0)

def run_experiment():
    print(f"\n--- EXPERIMENTO V152: AGGRESSIVE ARQUETYPE EXPANSION ---")
    
    # 1. CARGA DE DATOS
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, transform=transform)
    
    data_train = train_ds.data.unsqueeze(1).float() / 255.0
    targets_train = train_ds.targets.to(device)
    data_test = test_ds.data.unsqueeze(1).float() / 255.0
    targets_test = test_ds.targets.to(device)

    print("Pre-calculando firmas híbridas del dataset (60,000)...")
    all_feats = get_hybrid_vector(data_train)

    # 2. DESCUBRIMIENTO PAC AGRESIVO
    print("\nIniciando PAC Agresivo...")
    archetype_indices = []
    archetype_labels = []
    
    for c in range(10):
        idx = (targets_train == c).nonzero()[0].item()
        archetype_indices.append(idx)
        archetype_labels.append(c)

    for gen in range(40):
        arch_feats = all_feats[archetype_indices]
        sims = torch.mm(all_feats, arch_feats.t())
        best_arch_idx = torch.argmax(sims, dim=1)
        preds = torch.tensor(archetype_labels, device=device)[best_arch_idx]
        
        errors_mask = preds != targets_train
        num_errors = errors_mask.sum().item()
        acc = (1 - num_errors/60000) * 100
        
        if gen % 5 == 0 or gen == 39:
            print(f"Gen {gen:2} | Arquetipos: {len(archetype_indices):>4} | Train Acc: {acc:.2f}%")
        
        if num_errors < 100 or len(archetype_indices) > 2000: break
        
        # Añadimos múltiples arquetipos por clase (Bifurcación Agresiva)
        for c in range(10):
            c_errors = (errors_mask & (targets_train == c)).nonzero()
            # Añadimos hasta 8 nuevos arquetipos para las clases más difíciles
            num_to_add = min(8, len(c_errors) // 100 + 1)
            for j in range(num_to_add):
                if j < len(c_errors):
                    new_idx = c_errors[j].item()
                    if new_idx not in archetype_indices:
                        archetype_indices.append(new_idx)
                        archetype_labels.append(c)

    # 3. EXPANSIÓN SINTÉTICA (Aumentación)
    print(f"\nGenerando Memoria de Élite: {len(archetype_indices)} arquetipos base...")
    elite_bank = []
    elite_targets = []
    
    t_aug = time.perf_counter()
    for idx, label in zip(archetype_indices, archetype_labels):
        img = data_train[idx:idx+1].to(device)
        elite_bank.append(all_feats[idx:idx+1])
        elite_targets.append(label)
        
        # 4 variaciones por cada arquetipo para cubrir el espacio
        variations = [
            augment_image(img, angle=10, scale=1.0),
            augment_image(img, angle=-10, scale=1.0),
            augment_image(img, angle=0, scale=1.12),
            augment_image(img, angle=0, scale=0.88)
        ]
        for v_img in variations:
            elite_bank.append(get_hybrid_vector(v_img))
            elite_targets.append(label)

    elite_bank = torch.cat(elite_bank, dim=0)
    elite_targets = torch.tensor(elite_targets, device=device)
    print(f"Expansión completada en {time.perf_counter()-t_aug:.2f}s. Total: {len(elite_bank)} recuerdos.")

    # 4. EVALUACIÓN FINAL
    print("\nEvaluando contra el Test Set (10,000)...")
    test_feats = get_hybrid_vector(data_test)
    sims = torch.mm(test_feats, elite_bank.t())
    final_preds = elite_targets[torch.argmax(sims, dim=1)]
    final_acc = (final_preds == targets_test).float().mean().item()
    
    print("\n" + "="*55)
    print(f"RESULTADO MEMORIA AGRESIVA (V152)")
    print(f"="*55)
    print(f"Precisión Final: {final_acc*100:.2f}%")
    print(f"Arquetipos PAC:  {len(archetype_indices)}")
    print(f"Recuerdos Total: {len(elite_bank)} (Compresión {60000/len(elite_bank):.1f}x)")
    print("="*55)

if __name__ == "__main__":
    run_experiment()
