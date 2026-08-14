import sys
import os
import time
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    CopySection2D: Extractor de parches espaciales diferenciable.
    
    A partir de coordenadas continuas (cx, cy) acotadas en [-1, 1] y un radio r > 0,
    aplica muestreo por interpolación bilineal (grid_sample) para extraer un parche de
    dimensiones fijas (out_size x out_size) de la imagen de entrada.
    
    Parámetros por instancia/sample:
      - (cx, cy): centro de atención en coordenadas normalizadas [-1, 1]
      - radio: factor de escala del recorte
    """
    def __init__(self, out_size=14):
        super().__init__()
        self.out_size = out_size
        
        # Pre-crear cuadrícula base canónica en [-1, 1]^2
        y_coords = torch.linspace(-1, 1, out_size)
        x_coords = torch.linspace(-1, 1, out_size)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        # Forma: [1, out_size, out_size, 2] (coordenadas (x, y))
        canonical_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        self.register_buffer('canonical_grid', canonical_grid)

    def forward(self, x, cx, cy, radio):
        """
        x: [B, C, H, W]
        cx: [B] - Centro X en [-1, 1]
        cy: [B] - Centro Y en [-1, 1]
        radio: [B] - Radio / escala (> 0)
        """
        B = x.size(0)
        cx_val = cx.view(B, 1, 1)
        cy_val = cy.view(B, 1, 1)
        r_val = radio.view(B, 1, 1, 1)
        
        # Escalar y trasladar la cuadrícula canónica
        # Rejilla adaptada = canonical_grid * radio + (cx, cy)
        scaled_grid = self.canonical_grid.repeat(B, 1, 1, 1) * r_val
        grid_x = scaled_grid[..., 0] + cx_val
        grid_y = scaled_grid[..., 1] + cy_val
        
        grid = torch.stack([grid_x, grid_y], dim=-1)
        
        # Extracción bilineal diferenciable
        patch = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return patch

# -----------------------------------------------------------------------------
# Dataset Sintético: Canvas de 56x56 con Formas Geométricas Flotantes y Ruido
# -----------------------------------------------------------------------------
class SyntheticClutteredDataset(Dataset):
    """
    Genera imágenes de 56x56 con fondo ruidoso.
    En una posición aleatoria (target_x, target_y) coloca una forma entre 5 clases posibles:
      0: Cuadrado lleno
      1: Cruz / Más (+)
      2: Marco / Borde de cuadrado
      3: Punto / Círculo central
      4: Diagonal (X)
    """
    def __init__(self, num_samples=2000, canvas_size=56, patch_size=14, noise_level=0.15, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.num_samples = num_samples
        self.canvas_size = canvas_size
        self.patch_size = patch_size
        self.noise_level = noise_level
        
        # Pre-generar parches patrón [5, 1, 14, 14]
        self.patterns = torch.zeros(5, 1, patch_size, patch_size)
        
        # Clase 0: Cuadrado lleno
        self.patterns[0, 0, 3:11, 3:11] = 1.0
        
        # Clase 1: Cruz (+)
        self.patterns[1, 0, 6:8, 2:12] = 1.0
        self.patterns[1, 0, 2:12, 6:8] = 1.0
        
        # Clase 2: Marco / Borde
        self.patterns[2, 0, 2:12, 2:12] = 1.0
        self.patterns[2, 0, 4:10, 4:10] = 0.0
        
        # Clase 3: Punto central (Círculo acotado)
        for i in range(patch_size):
            for j in range(patch_size):
                if (i - 6.5)**2 + (j - 6.5)**2 <= 4.0**2:
                    self.patterns[3, 0, i, j] = 1.0
                    
        # Clase 4: Diagonal (X)
        for i in range(2, 12):
            self.patterns[4, 0, i, i] = 1.0
            self.patterns[4, 0, i, 13 - i] = 1.0
            
        # Pre-generar datos sintéticos para reproducibilidad rápida
        self.images = torch.randn(num_samples, 1, canvas_size, canvas_size) * noise_level
        self.labels = torch.randint(0, 5, (num_samples,))
        self.target_pos = (torch.rand(num_samples, 2) * 1.2 - 0.6) # pos en [-0.6, 0.6] (espacio [-1, 1])
        
        # Estampar patrones en las imágenes
        half_p = patch_size / canvas_size # radio aproximado
        for i in range(num_samples):
            cls = self.labels[i]
            tx, ty = self.target_pos[i] # Coordenadas normalizadas [-1, 1]
            
            # Convertir a píxeles
            px = int((tx.item() + 1.0) * 0.5 * (canvas_size - patch_size))
            py = int((ty.item() + 1.0) * 0.5 * (canvas_size - patch_size))
            
            self.images[i, 0, py:py+patch_size, px:px+patch_size] += self.patterns[cls, 0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.target_pos[idx]

# -----------------------------------------------------------------------------
# Modelo End-to-End con Neurona CopySection
# -----------------------------------------------------------------------------
class FovealCopySectionNet(nn.Module):
    """
    Red Foveal con Neurona Copy-Section.
    1. Localizador (CNN Ligera): Toma la imagen completa (56x56) y predice (cx, cy, radio)
    2. CopySection2D: Recorta diferenciablemente la sub-región atencional (14x14)
    3. Clasificador: Procesa ÚNICAMENTE el parche recortado de 14x14 para predecir la clase
    """
    def __init__(self, in_channels=1, patch_size=14, num_classes=5):
        super().__init__()
        # Red de localización (muy pequeña)
        self.localizer = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=5, stride=2, padding=2), # 28x28
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2), # 14x14
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # 1x1
            nn.Flatten(),
            nn.Linear(16, 3) # Emite (raw_cx, raw_cy, raw_radio)
        )
        
        self.copy_section = CopySection2D(out_size=patch_size)
        
        # Clasificador de parche (opera solo sobre 14x14 = 196 entradas)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(patch_size * patch_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # 1. Predecir atención espacial
        loc_out = self.localizer(x)
        cx = torch.tanh(loc_out[:, 0]) # acotado en [-1, 1]
        cy = torch.tanh(loc_out[:, 1]) # acotado en [-1, 1]
        # Radio acotado en [0.15, 0.8]
        radio = torch.sigmoid(loc_out[:, 2]) * 0.65 + 0.15
        
        # 2. Extracción de parche con CopySection
        patch = self.copy_section(x, cx, cy, radio)
        
        # 3. Clasificación sobre el parche extraído
        logits = self.classifier(patch)
        return logits, patch, cx, cy, radio

# -----------------------------------------------------------------------------
# Baseline: Red Densa / CNN Directa sin Capa Atencional
# -----------------------------------------------------------------------------
class BaselineDirectNet(nn.Module):
    """
    Baseline Denso Directo: intenta clasificar procesando directamente toda la imagen de 56x56.
    """
    def __init__(self, in_size=56, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_size * in_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------
# Bucle de Entrenamiento y Evaluación
# -----------------------------------------------------------------------------
def run_experiment():
    log_msg("=========================================================================")
    log_msg("  EXPERIMENTO 1: Copy-Section / Foveal Neuron (v360_copy_section_localizer)")
    log_msg("=========================================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_msg(f"Metadatos: Python {sys.version.split()[0]} | PyTorch {torch.__version__} | Dispositivo: {device}")
    
    seed = 42
    torch.manual_seed(seed)
    
    # Cargar Dataset
    train_dataset = SyntheticClutteredDataset(num_samples=2500, noise_level=0.15, seed=seed)
    test_dataset = SyntheticClutteredDataset(num_samples=500, noise_level=0.15, seed=seed+1)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Instanciar modelos
    foveal_net = FovealCopySectionNet(patch_size=14, num_classes=5).to(device)
    baseline_net = BaselineDirectNet(in_size=56, num_classes=5).to(device)
    
    foveal_params = sum(p.numel() for p in foveal_net.parameters() if p.requires_grad)
    baseline_params = sum(p.numel() for p in baseline_net.parameters() if p.requires_grad)
    
    log_msg(f"Inventario Arquitectura:")
    log_msg(f"  - Foveal CopySectionNet Parámetros: {foveal_params}")
    log_msg(f"  - Baseline Denso Directo Parámetros: {baseline_params}")
    
    # Optimización
    opt_foveal = torch.optim.Adam(foveal_net.parameters(), lr=0.003)
    opt_baseline = torch.optim.Adam(baseline_net.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 15
    foveal_history = []
    baseline_history = []
    
    log_msg("\n--- Iniciando Entrenamiento (REGLA FAST FEEDBACK en primeros 5 batches) ---")
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        
        # --- Entrenar Foveal CopySection ---
        foveal_net.train()
        foveal_train_loss = 0.0
        loc_error_accum = 0.0
        
        for batch_idx, (imgs, labels, target_pos) in enumerate(train_loader):
            imgs, labels, target_pos = imgs.to(device), labels.to(device), target_pos.to(device)
            
            opt_foveal.zero_grad()
            logits, patch, cx, cy, radio = foveal_net(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            opt_foveal.step()
            
            foveal_train_loss += loss.item()
            
            # Calcular error de localización euclidiana
            pred_pos = torch.stack([cx, cy], dim=1)
            loc_err = torch.norm(pred_pos - target_pos, p=2, dim=1).mean().item()
            loc_error_accum += loc_err
            
            # FAST FEEDBACK (Primeros 5 batches de la época 1)
            if epoch == 1 and batch_idx < 5:
                log_msg(f"  [FAST-FEEDBACK Epoch 1 Batch {batch_idx+1}] Foveal Loss: {loss.item():.4f} | Loc Error: {loc_err:.4f} | Avg Cx: {cx.mean().item():.3f}, Cy: {cy.mean().item():.3f}")
                
        # --- Entrenar Baseline ---
        baseline_net.train()
        baseline_train_loss = 0.0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt_baseline.zero_grad()
            logits_b = baseline_net(imgs)
            loss_b = criterion(logits_b, labels)
            loss_b.backward()
            opt_baseline.step()
            baseline_train_loss += loss_b.item()
            
        # --- Evaluación ---
        foveal_net.eval()
        baseline_net.eval()
        
        foveal_correct = 0
        baseline_correct = 0
        val_loc_err = 0.0
        total_eval = 0
        
        with torch.no_grad():
            for imgs, labels, target_pos in test_loader:
                imgs, labels, target_pos = imgs.to(device), labels.to(device), target_pos.to(device)
                
                # Eval Foveal
                logits_f, _, cx, cy, _ = foveal_net(imgs)
                pred_f = logits_f.argmax(dim=1)
                foveal_correct += (pred_f == labels).sum().item()
                
                pred_pos = torch.stack([cx, cy], dim=1)
                val_loc_err += torch.norm(pred_pos - target_pos, p=2, dim=1).sum().item()
                
                # Eval Baseline
                logits_b = baseline_net(imgs)
                pred_b = logits_b.argmax(dim=1)
                baseline_correct += (pred_b == labels).sum().item()
                
                total_eval += imgs.size(0)
                
        foveal_acc = 100.0 * foveal_correct / total_eval
        baseline_acc = 100.0 * baseline_correct / total_eval
        avg_val_loc_err = val_loc_err / total_eval
        
        t_epoch = time.time() - t0
        
        log_msg(f"Epoch {epoch:02d}/{epochs:02d} | Foveal Acc: {foveal_acc:.2f}% (Loc Err: {avg_val_loc_err:.3f}) | Baseline Acc: {baseline_acc:.2f}% | Tiempo: {t_epoch:.2f}s")
        
        foveal_history.append({"epoch": epoch, "acc": foveal_acc, "loc_err": avg_val_loc_err})
        baseline_history.append({"epoch": epoch, "acc": baseline_acc})

    # --- Resumen y Conclusiones ---
    log_msg("\n=========================================================================")
    log_msg("  RESUMEN FINAL DE RESULTADOS EXPERIMENTO 1")
    log_msg("=========================================================================")
    log_msg(f"Foveal CopySectionNet -> Accuracy Final: {foveal_history[-1]['acc']:.2f}% | Error Localización (dist. euclidiana): {foveal_history[-1]['loc_err']:.4f}")
    log_msg(f"Baseline Denso Directo -> Accuracy Final: {baseline_history[-1]['acc']:.2f}%")
    log_msg(f"Parámetros: Foveal={foveal_params} vs Baseline={baseline_params} ({baseline_params/foveal_params:.2f}x más grande)")

    # Guardar resultados JSON
    output_dir = "results/raw"
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "v360_copy_section_results.json")
    
    results_data = {
        "experiment_id": "v360_copy_section_localizer",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rigor_level": 1,
        "foveal_params": foveal_params,
        "baseline_params": baseline_params,
        "final_foveal_acc": foveal_history[-1]['acc'],
        "final_baseline_acc": baseline_history[-1]['acc'],
        "final_loc_error": foveal_history[-1]['loc_err'],
        "foveal_history": foveal_history,
        "baseline_history": baseline_history
    }
    
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    log_msg(f"Resultados persistidos en: {results_file}")

    # Añadir a Master Ledger
    ledger_path = "results/master_ledger.jsonl"
    os.makedirs("results", exist_ok=True)
    ledger_entry = {
        "experiment_id": "v360",
        "fecha": time.strftime("%Y-%m-%d"),
        "familia": " geometrico_atencion_foveal ",
        "dataset": "SyntheticCluttered (56x56, 5 clases)",
        "n_eval": 500,
        "metric_name": "acc",
        "value": foveal_history[-1]['acc'],
        "SE": None,
        "params": foveal_params,
        "nivel_rigor": 1,
        "etiqueta": "SEÑAL" if foveal_history[-1]['acc'] > baseline_history[-1]['acc'] else "RUIDO-SOSPECHA"
    }
    with open(ledger_path, "a") as f:
        f.write(json.dumps(ledger_entry) + "\n")
    log_msg(f"Entrada registrada en Master Ledger: {ledger_path}")

if __name__ == '__main__':
    run_experiment()
