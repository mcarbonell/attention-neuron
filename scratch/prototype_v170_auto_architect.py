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
    """ Walsh-Hadamard Transform 1D batch """
    b, n = x.shape
    res = x.clone()
    h = 1
    while h < n:
        res = res.view(b, n // (2 * h), 2, h)
        a, b_ = res[:, :, 0, :], res[:, :, 1, :]
        res = torch.cat([a + b_, a - b_], dim=2)
        h *= 2
    return res.view(b, n) / (n ** 0.5)

# --- CLUSTERING ---
def get_ordered_clanes(feats, k=256):
    N, D = feats.shape
    if N < k: k = N
    centers = feats[torch.randperm(N)[:k]]
    for _ in range(8):
        sims = torch.mm(feats, centers.t())
        assignments = torch.argmax(sims, dim=1)
        for i in range(k):
            mask = assignments == i
            if mask.any(): 
                centers[i] = F.normalize(feats[mask].mean(0), p=2, dim=0)
    return centers

def get_v_matrix(clanes, feats, targets):
    K = clanes.size(0)
    sims = torch.mm(feats, clanes.t())
    assignments = torch.argmax(sims, dim=1)
    clan_labels = []
    targets_cpu = targets.cpu()
    for i in range(K):
        mask = (assignments == i).cpu()
        if mask.any():
            clan_labels.append(torch.bincount(targets_cpu[mask], minlength=10).argmax().item())
        else: clan_labels.append(0)
    labels_cpu = torch.tensor(clan_labels, dtype=torch.long)
    return F.one_hot(labels_cpu, num_classes=10).float().to(device)

def evaluate_and_get_errors(w_data, targets, layers_list, v_matrices):
    """ Evalúa el conjunto total y devuelve los errores """
    total_votes = torch.zeros(len(w_data), 10).to(device)
    batch_size = 5000
    for i in range(0, len(w_data), batch_size):
        b_w = w_data[i:i+batch_size]
        b_votes = torch.zeros(len(b_w), 10).to(device)
        for idx, (clanes, v_mat) in enumerate(zip(layers_list, v_matrices)):
            sims = torch.mm(b_w, clanes.t())
            # Potencia adaptativa por profundidad
            power = 12 + (idx * 2)
            votes = torch.mm(torch.pow(torch.clamp(sims, min=0.0), power), v_mat)
            b_votes += votes
        total_votes[i:i+batch_size] = b_votes
    
    preds = torch.argmax(total_votes, dim=1)
    mask_err = preds != targets
    acc = (preds == targets).float().mean().item()
    return acc, w_data[mask_err], targets[mask_err]

def run_experiment():
    print(f"\n--- EXPERIMENTO V170: AUTO-ARCHITECT (INFINITE NEUROGENESIS) ---")
    
    # 1. CARGA Y PREPROCESAMIENTO
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    raw_train = train_ds.data.float().to(device) / 255.0
    targets_train = train_ds.targets.to(device)
    
    padded_all = torch.zeros(60000, 32, 32).to(device)
    padded_all[:, :28, :28] = raw_train
    w_train_all = F.normalize(fwht(padded_all.reshape(60000, 1024)), p=2, dim=1)

    # 2. BUCLE DE CRECIMIENTO AUTOMÁTICO
    target_acc = 0.975
    max_layers = 10
    all_layers = []
    all_v_matrices = []
    current_train_acc = 0.0
    
    layer_idx = 1
    while current_train_acc < target_acc and layer_idx <= max_layers:
        print(f"\n[Capas: {layer_idx}] Auditando cerebro actual...")
        acc, err_w, err_t = evaluate_and_get_errors(w_train_all, targets_train, all_layers, all_v_matrices)
        current_train_acc = acc
        print(f"Precisión en Entrenamiento: {acc*100:.2f}% | Errores restantes: {len(err_w)}")
        
        if acc >= target_acc:
            print(f"¡Objetivo alcanzado! {acc*100:.2f}% >= {target_acc*100:.2f}%")
            break
            
        # Determinar clanes para la nueva capa (Neurogénesis)
        # Si quedan pocos errores, reducimos el número de clanes para evitar overfitting
        num_new_clanes = min(256, len(err_w) // 4)
        if num_new_clanes < 10: 
            print("Crecimiento saturado (pocos errores para una capa nueva).")
            break
            
        print(f"Inyectando Capa {layer_idx} con {num_new_clanes} especialistas...")
        
        if layer_idx == 1:
            # La primera capa se entrena con una muestra general
            indices = torch.randperm(60000)[:15000]
            new_clanes = get_ordered_clanes(w_train_all[indices], k=256)
            new_v = get_v_matrix(new_clanes, w_train_all[indices], targets_train[indices])
        else:
            # Capas sucesivas se entrenan solo con los errores
            new_clanes = get_ordered_clanes(err_w, k=num_new_clanes)
            new_v = get_v_matrix(new_clanes, err_w, err_t)
            
        all_layers.append(new_clanes)
        all_v_matrices.append(new_v)
        layer_idx += 1

    # 3. EVALUACIÓN FINAL EN TEST SET
    print("\nEvaluando Cerebro Auto-Arquitecto en Test Set...")
    raw_test = test_ds.data.float().to(device) / 255.0
    test_padded = torch.zeros(10000, 32, 32).to(device)
    test_padded[:, :28, :28] = raw_test
    test_walsh = F.normalize(fwht(test_padded.reshape(10000, 1024)), p=2, dim=1)
    
    t0 = time.perf_counter()
    
    final_votes = torch.zeros(10000, 10).to(device)
    for idx, (clanes, v_mat) in enumerate(zip(all_layers, all_v_matrices)):
        sims = torch.mm(test_walsh, clanes.t())
        power = 12 + (idx * 2) 
        votes = torch.mm(torch.pow(torch.clamp(sims, min=0.0), power), v_mat)
        final_votes += votes
    
    preds = torch.argmax(final_votes, dim=1)
    acc_test = (preds == test_ds.targets.to(device)).float().mean().item()
    dt = time.perf_counter() - t0
    
    print("\n" + "="*55)
    print(f"RESULTADO AUTO-ARCHITECT (V170)")
    print(f"="*55)
    print(f"Precisión Test:  {acc_test*100:.2f}%")
    print(f"Precisión Train: {current_train_acc*100:.2f}%")
    print(f"Total Capas:     {len(all_layers)}")
    print(f"Clanes Totales:  {sum([l.size(0) for l in all_layers])}")
    print(f"Compresión:      {60000 // sum([l.size(0) for l in all_layers])}x")
    print(f"Tiempo Total:    {dt:.4f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v170_auto_architect.json", "w") as f:
        json.dump({"accuracy": acc_test, "layers": len(all_layers)}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
