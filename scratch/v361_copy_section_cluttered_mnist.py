import sys
import os
import time
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

# Start time tracking for formatted logs
START_TIME = time.time()

def log_msg(msg):
    elapsed = time.time() - START_TIME
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    timestamp = f"[+{hours:02d}:{minutes:02d}:{seconds:05.2f}]"
    print(f"{timestamp} {msg}", flush=True)

# -----------------------------------------------------------------------------
# Core Layer: CopySection2D (Differentiable Grid Sampling Patch Extractor)
# -----------------------------------------------------------------------------
class CopySection2D(nn.Module):
    """
    CopySection2D: Extractor de parches espaciales diferenciable mediante grid_sample.
    """
    def __init__(self, out_size=28):
        super().__init__()
        self.out_size = out_size
        
        y_coords = torch.linspace(-1, 1, out_size)
        x_coords = torch.linspace(-1, 1, out_size)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        canonical_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        self.register_buffer('canonical_grid', canonical_grid)

    def forward(self, x, cx, cy, radio):
        B = x.size(0)
        cx_val = cx.view(B, 1, 1)
        cy_val = cy.view(B, 1, 1)
        r_val = radio.view(B, 1, 1, 1)
        
        scaled_grid = self.canonical_grid.repeat(B, 1, 1, 1) * r_val
        grid_x = scaled_grid[..., 0] + cx_val
        grid_y = scaled_grid[..., 1] + cy_val
        
        grid = torch.stack([grid_x, grid_y], dim=-1)
        patch = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return patch

# -----------------------------------------------------------------------------
# Dataset Sintético: Cluttered MNIST (Canvas de 60x60)
# -----------------------------------------------------------------------------
class ClutteredMNISTDataset(Dataset):
    """
    Construye imágenes de 60x60 píxeles.
    - Se coloca el Dígito MNIST Objetivo (28x28) en una posición aleatoria (tx, ty).
    - Se agregan 2 parches de distractores (14x14 recortados de otros dígitos aleatorios).
    - Ruido de fondo blanco leve.
    """
    def __init__(self, mnist_dataset, canvas_size=60, num_distractors=2, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.mnist = mnist_dataset
        self.canvas_size = canvas_size
        self.num_distractors = num_distractors
        self.n_samples = len(mnist_dataset)
        
        # Pre-construir canvas para entrenamiento acelerado y determinista
        log_msg(f"Construyendo Cluttered MNIST Dataset ({self.n_samples} muestras, canvas {canvas_size}x{canvas_size})...")
        self.canvas_images = torch.zeros(self.n_samples, 1, canvas_size, canvas_size)
        self.labels = torch.zeros(self.n_samples, dtype=torch.long)
        self.target_coords = torch.zeros(self.n_samples, 2)
        
        digit_size = 28
        max_offset = canvas_size - digit_size # 32 píxeles de margen
        
        for i in range(self.n_samples):
            target_img, label = self.mnist[i]
            self.labels[i] = label
            
            # Posición aleatoria del objetivo en píxeles
            px = torch.randint(0, max_offset + 1, (1,)).item()
            py = torch.randint(0, max_offset + 1, (1,)).item()
            
            # Convertir a coordenadas [-1, 1] del centro del objetivo
            cx = (px + digit_size / 2.0) / (canvas_size / 2.0) - 1.0
            cy = (py + digit_size / 2.0) / (canvas_size / 2.0) - 1.0
            self.target_coords[i] = torch.tensor([cx, cy])
            
            # Crear canvas con ruido de fondo
            canvas = torch.randn(1, canvas_size, canvas_size) * 0.05
            
            # Colocar objetivo
            canvas[:, py:py+digit_size, px:px+digit_size] += target_img[0]
            
            # Colocar distractores
            for _ in range(num_distractors):
                rand_idx = torch.randint(0, self.n_samples, (1,)).item()
                dist_img, _ = self.mnist[rand_idx]
                # Recortar un sub-parche de 14x14 del distractor
                dx = torch.randint(0, 14, (1,)).item()
                dy = torch.randint(0, 14, (1,)).item()
                d_patch = dist_img[0, dy:dy+14, dx:dx+14]
                
                # Posición aleatoria para el distractor
                d_px = torch.randint(0, canvas_size - 14, (1,)).item()
                d_py = torch.randint(0, canvas_size - 14, (1,)).item()
                canvas[:, d_py:d_py+14, d_px:d_px+14] += d_patch * 0.7
                
            self.canvas_images[i] = torch.clamp(canvas, 0.0, 1.0)
            
        log_msg("Dataset Cluttered MNIST construido con éxito.")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.canvas_images[idx], self.labels[idx], self.target_coords[idx]

# -----------------------------------------------------------------------------
# Modelo Foveal CopySectionNet (V361)
# -----------------------------------------------------------------------------
class FovealCopySectionMNIST(nn.Module):
    def __init__(self, out_patch_size=28, num_classes=10):
        super().__init__()
        # Localizador Liviano: toma la imagen 60x60 downsampled y predice (cx, cy, radio)
        self.localizer = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2), # 30x30
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2), # 15x15
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 3) # Emite (cx, cy, radio)
        )
        
        self.copy_section = CopySection2D(out_size=out_patch_size)
        
        # Clasificador de Parche (procesa el parche recortado de 28x28 = 784 píxeles)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_patch_size * out_patch_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        loc_out = self.localizer(x)
        cx = torch.tanh(loc_out[:, 0]) # [-1, 1]
        cy = torch.tanh(loc_out[:, 1]) # [-1, 1]
        radio = torch.sigmoid(loc_out[:, 2]) * 0.5 + 0.3 # [0.3, 0.8]
        
        patch = self.copy_section(x, cx, cy, radio)
        logits = self.classifier(patch)
        return logits, patch, cx, cy, radio

# -----------------------------------------------------------------------------
# Baseline 1: Red Densa Estándar
# -----------------------------------------------------------------------------
class BaselineDenseNet(nn.Module):
    def __init__(self, in_size=60, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_size * in_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------
# Baseline 2: CNN Estándar (ConvNet de 2 capas)
# -----------------------------------------------------------------------------
class BaselineCNNNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2), # 30x30
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), # 15x15
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------
# Bucle Principal del Experimento 2
# -----------------------------------------------------------------------------
def run_experiment():
    log_msg("=========================================================================")
    log_msg("  EXPERIMENTO 2: Cluttered MNIST con Neurona Copy-Section (v361)")
    log_msg("=========================================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_msg(f"Metadatos: Python {sys.version.split()[0]} | PyTorch {torch.__version__} | Dispositivo: {device}")
    
    seed = 42
    torch.manual_seed(seed)
    
    # Cargar MNIST base
    transform = transforms.Compose([transforms.ToTensor()])
    raw_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    raw_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Crear Cluttered MNIST Datasets (Usamos 10,000 train / 2,000 test para rapidez)
    train_subset = torch.utils.data.Subset(raw_train, range(10000))
    test_subset = torch.utils.data.Subset(raw_test, range(2000))
    
    train_dataset = ClutteredMNISTDataset(train_subset, canvas_size=60, num_distractors=2, seed=seed)
    test_dataset = ClutteredMNISTDataset(test_subset, canvas_size=60, num_distractors=2, seed=seed+1)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Instanciar Modelos
    foveal_net = FovealCopySectionMNIST(out_patch_size=28, num_classes=10).to(device)
    dense_net = BaselineDenseNet(in_size=60, num_classes=10).to(device)
    cnn_net = BaselineCNNNet(num_classes=10).to(device)
    
    foveal_params = sum(p.numel() for p in foveal_net.parameters() if p.requires_grad)
    dense_params = sum(p.numel() for p in dense_net.parameters() if p.requires_grad)
    cnn_params = sum(p.numel() for p in cnn_net.parameters() if p.requires_grad)
    
    log_msg("Inventario Arquitectónico:")
    log_msg(f"  - Foveal CopySectionNet (V361) Parámetros: {foveal_params}")
    log_msg(f"  - Baseline DenseNet Parámetros:            {dense_params}")
    log_msg(f"  - Baseline CNNNet Parámetros:              {cnn_params}")
    
    # Optimizadores
    opt_foveal = torch.optim.Adam(foveal_net.parameters(), lr=0.002)
    opt_dense = torch.optim.Adam(dense_net.parameters(), lr=0.002)
    opt_cnn = torch.optim.Adam(cnn_net.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 12
    history = {"foveal": [], "dense": [], "cnn": []}
    
    log_msg("\n--- Iniciando Entrenamiento (FAST FEEDBACK en primeros 5 batches) ---")
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        
        # 1. Entrenar Foveal
        foveal_net.train()
        f_loss = 0.0
        for batch_idx, (imgs, labels, target_coords) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            opt_foveal.zero_grad()
            logits, patch, cx, cy, radio = foveal_net(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            opt_foveal.step()
            f_loss += loss.item()
            
            if epoch == 1 and batch_idx < 5:
                log_msg(f"  [FAST-FEEDBACK Epoch 1 Batch {batch_idx+1}] Foveal Loss: {loss.item():.4f} | Avg Cx: {cx.mean().item():.3f}, Cy: {cy.mean().item():.3f}")
                
        # 2. Entrenar Dense Baseline
        dense_net.train()
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt_dense.zero_grad()
            logits_d = dense_net(imgs)
            loss_d = criterion(logits_d, labels)
            loss_d.backward()
            opt_dense.step()
            
        # 3. Entrenar CNN Baseline
        cnn_net.train()
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt_cnn.zero_grad()
            logits_c = cnn_net(imgs)
            loss_c = criterion(logits_c, labels)
            loss_c.backward()
            opt_cnn.step()
            
        # Evaluación
        foveal_net.eval()
        dense_net.eval()
        cnn_net.eval()
        
        f_correct, d_correct, c_correct, total = 0, 0, 0, 0
        loc_err_accum = 0.0
        
        with torch.no_grad():
            for imgs, labels, target_coords in test_loader:
                imgs, labels, target_coords = imgs.to(device), labels.to(device), target_coords.to(device)
                
                # Foveal
                logits_f, _, cx, cy, _ = foveal_net(imgs)
                f_correct += (logits_f.argmax(dim=1) == labels).sum().item()
                pred_coords = torch.stack([cx, cy], dim=1)
                loc_err_accum += torch.norm(pred_coords - target_coords, p=2, dim=1).sum().item()
                
                # Dense
                logits_d = dense_net(imgs)
                d_correct += (logits_d.argmax(dim=1) == labels).sum().item()
                
                # CNN
                logits_c = cnn_net(imgs)
                c_correct += (logits_c.argmax(dim=1) == labels).sum().item()
                
                total += imgs.size(0)
                
        f_acc = 100.0 * f_correct / total
        d_acc = 100.0 * d_correct / total
        c_acc = 100.0 * c_correct / total
        avg_loc_err = loc_err_accum / total
        
        t_epoch = time.time() - t0
        log_msg(f"Epoch {epoch:02d}/{epochs:02d} | Foveal Acc: {f_acc:.2f}% (LocErr: {avg_loc_err:.3f}) | Dense Acc: {d_acc:.2f}% | CNN Acc: {c_acc:.2f}% | Tiempo: {t_epoch:.2f}s")
        
        history["foveal"].append({"epoch": epoch, "acc": f_acc, "loc_err": avg_loc_err})
        history["dense"].append({"epoch": epoch, "acc": d_acc})
        history["cnn"].append({"epoch": epoch, "acc": c_acc})

    # Resumen Final
    log_msg("\n=========================================================================")
    log_msg("  RESUMEN FINAL DE RESULTADOS EXPERIMENTO 2 (CLUTTERED MNIST)")
    log_msg("=========================================================================")
    log_msg(f"Foveal CopySectionNet -> Acc Final: {history['foveal'][-1]['acc']:.2f}% (Params: {foveal_params})")
    log_msg(f"Baseline DenseNet     -> Acc Final: {history['dense'][-1]['acc']:.2f}% (Params: {dense_params})")
    log_msg(f"Baseline CNNNet       -> Acc Final: {history['cnn'][-1]['acc']:.2f}% (Params: {cnn_params})")

    # Persistir JSON
    os.makedirs("results/raw", exist_ok=True)
    res_path = "results/raw/v361_copy_section_results.json"
    res_data = {
        "experiment_id": "v361_copy_section_cluttered_mnist",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rigor_level": 1,
        "foveal_params": foveal_params,
        "dense_params": dense_params,
        "cnn_params": cnn_params,
        "final_foveal_acc": history['foveal'][-1]['acc'],
        "final_dense_acc": history['dense'][-1]['acc'],
        "final_cnn_acc": history['cnn'][-1]['acc'],
        "history": history
    }
    with open(res_path, "w") as f:
        json.dump(res_data, f, indent=2)
    log_msg(f"Resultados guardados en: {res_path}")

    # Ledger
    ledger_path = "results/master_ledger.jsonl"
    ledger_entry = {
        "experiment_id": "v361",
        "fecha": time.strftime("%Y-%m-%d"),
        "familia": " geometrico_atencion_foveal ",
        "dataset": "Cluttered MNIST (60x60, 10 clases)",
        "n_eval": 2000,
        "metric_name": "acc",
        "value": history['foveal'][-1]['acc'],
        "SE": None,
        "params": foveal_params,
        "nivel_rigor": 1,
        "etiqueta": "ANCLA" if history['foveal'][-1]['acc'] > max(history['dense'][-1]['acc'], history['cnn'][-1]['acc']) else "SEÑAL"
    }
    with open(ledger_path, "a") as f:
        f.write(json.dumps(ledger_entry) + "\n")
    log_msg(f"Registrado en Master Ledger: {ledger_path}")

if __name__ == '__main__':
    run_experiment()
