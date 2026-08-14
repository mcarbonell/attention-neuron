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
# Core Layer: MultiHeadCopySection2D (Vectorizado)
# -----------------------------------------------------------------------------
class MultiHeadCopySection2D(nn.Module):
    def __init__(self, num_heads=4, out_size=28):
        super().__init__()
        self.num_heads = num_heads
        self.out_size = out_size
        
        y_coords = torch.linspace(-1, 1, out_size)
        x_coords = torch.linspace(-1, 1, out_size)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        canonical_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        self.register_buffer('canonical_grid', canonical_grid)

    def forward(self, x, cx, cy, radio):
        B, C, H, W = x.size()
        K = self.num_heads
        
        x_expanded = x.unsqueeze(1).repeat(1, K, 1, 1, 1).view(B * K, C, H, W)
        
        cx_flat = cx.view(B * K, 1, 1)
        cy_flat = cy.view(B * K, 1, 1)
        r_flat = radio.view(B * K, 1, 1, 1)
        
        scaled_grid = self.canonical_grid.repeat(B * K, 1, 1, 1) * r_flat
        grid_x = scaled_grid[..., 0] + cx_flat
        grid_y = scaled_grid[..., 1] + cy_flat
        grid = torch.stack([grid_x, grid_y], dim=-1)
        
        patches_flat = F.grid_sample(x_expanded, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        patches = patches_flat.view(B, K, C, self.out_size, self.out_size)
        return patches

# -----------------------------------------------------------------------------
# Dataset Sintético: Cluttered MNIST (Canvas 60x60)
# -----------------------------------------------------------------------------
class ClutteredMNISTDataset(Dataset):
    def __init__(self, mnist_dataset, canvas_size=60, num_distractors=2, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.mnist = mnist_dataset
        self.canvas_size = canvas_size
        self.num_distractors = num_distractors
        self.n_samples = len(mnist_dataset)
        
        self.canvas_images = torch.zeros(self.n_samples, 1, canvas_size, canvas_size)
        self.labels = torch.zeros(self.n_samples, dtype=torch.long)
        self.target_coords = torch.zeros(self.n_samples, 2)
        
        digit_size = 28
        max_offset = canvas_size - digit_size
        
        for i in range(self.n_samples):
            target_img, label = self.mnist[i]
            self.labels[i] = label
            
            px = torch.randint(0, max_offset + 1, (1,)).item()
            py = torch.randint(0, max_offset + 1, (1,)).item()
            
            cx = (px + digit_size / 2.0) / (canvas_size / 2.0) - 1.0
            cy = (py + digit_size / 2.0) / (canvas_size / 2.0) - 1.0
            self.target_coords[i] = torch.tensor([cx, cy])
            
            canvas = torch.randn(1, canvas_size, canvas_size) * 0.05
            canvas[:, py:py+digit_size, px:px+digit_size] += target_img[0]
            
            for _ in range(num_distractors):
                rand_idx = torch.randint(0, self.n_samples, (1,)).item()
                dist_img, _ = self.mnist[rand_idx]
                dx = torch.randint(0, 14, (1,)).item()
                dy = torch.randint(0, 14, (1,)).item()
                d_patch = dist_img[0, dy:dy+14, dx:dx+14]
                
                d_px = torch.randint(0, canvas_size - 14, (1,)).item()
                d_py = torch.randint(0, canvas_size - 14, (1,)).item()
                canvas[:, d_py:d_py+14, d_px:d_px+14] += d_patch * 0.7
                
            self.canvas_images[i] = torch.clamp(canvas, 0.0, 1.0)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.canvas_images[idx], self.labels[idx], self.target_coords[idx]

# -----------------------------------------------------------------------------
# Modelo Multi-Head Foveal Net Escilable
# -----------------------------------------------------------------------------
class ScalableMultiHeadFovealNet(nn.Module):
    def __init__(self, num_heads=4, out_patch_size=28, num_classes=10):
        super().__init__()
        self.num_heads = num_heads
        
        self.localizer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2), # 30x30
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), # 15x15
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_heads * 2)
        )
        
        # Inicialización sesgada en cuadrícula uniforme sobre [-0.5, 0.5]
        with torch.no_grad():
            grid_cols = math.ceil(math.sqrt(num_heads))
            grid_rows = math.ceil(num_heads / grid_cols)
            
            inits = []
            x_lin = torch.linspace(-0.5, 0.5, grid_cols) if grid_cols > 1 else torch.tensor([0.0])
            y_lin = torch.linspace(-0.5, 0.5, grid_rows) if grid_rows > 1 else torch.tensor([0.0])
            
            for y_val in y_lin:
                for x_val in x_lin:
                    if len(inits) < num_heads:
                        inits.append([x_val.item(), y_val.item()])
                        
            quad_inits = torch.tensor(inits, dtype=torch.float32)
            self.localizer[-1].weight.data.zero_()
            self.localizer[-1].bias.data.copy_(quad_inits.view(-1))
        
        self.copy_section = MultiHeadCopySection2D(num_heads=num_heads, out_size=out_patch_size)
        
        self.patch_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_patch_size * out_patch_size, 64),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, current_radio=0.5):
        B = x.size(0)
        K = self.num_heads
        
        loc_out = self.localizer(x).view(B, K, 2)
        cx = torch.tanh(loc_out[:, :, 0])
        cy = torch.tanh(loc_out[:, :, 1])
        
        radio = torch.full((B, K), current_radio, device=x.device)
        patches = self.copy_section(x, cx, cy, radio)
        
        patches_flat = patches.view(B * K, 1, 28, 28)
        feats_flat = self.patch_encoder(patches_flat)
        feats = feats_flat.view(B, K, 64)
        
        # Max pooling sobre las K cabezas
        pooled_feats = feats.max(dim=1)[0]
        logits = self.classifier(pooled_feats)
        return logits, cx, cy

# -----------------------------------------------------------------------------
# Barrido de Cabezas (Sweep K = 1, 2, 4, 8, 16)
# -----------------------------------------------------------------------------
def run_experiment():
    log_msg("=========================================================================")
    log_msg("  EXPERIMENTO 4: Barrido de Cabezas Foveales (v363_foveal_head_sweep)")
    log_msg("=========================================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_msg(f"Metadatos: Python {sys.version.split()[0]} | PyTorch {torch.__version__} | Dispositivo: {device}")
    
    seed = 42
    torch.manual_seed(seed)
    
    # Cargar Dataset común para comparar K en igualdad de datos
    transform = transforms.Compose([transforms.ToTensor()])
    raw_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    raw_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_subset = torch.utils.data.Subset(raw_train, range(10000))
    test_subset = torch.utils.data.Subset(raw_test, range(2000))
    
    train_dataset = ClutteredMNISTDataset(train_subset, canvas_size=60, num_distractors=2, seed=seed)
    test_dataset = ClutteredMNISTDataset(test_subset, canvas_size=60, num_distractors=2, seed=seed+1)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    head_configs = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    epochs = 12
    sweep_results = {}
    
    for K in head_configs:
        log_msg(f"\n>>> Evaluando Configuración: K = {K} Cabezas Foveales <<<")
        torch.manual_seed(seed)
        
        model = ScalableMultiHeadFovealNet(num_heads=K, out_patch_size=28, num_classes=10).to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        criterion = nn.CrossEntropyLoss()
        
        k_history = []
        
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            radio_t = 0.85 - 0.40 * (epoch - 1) / (epochs - 1)
            
            model.train()
            for batch_idx, (imgs, labels, _) in enumerate(train_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                logits, cx, cy = model(imgs, current_radio=radio_t)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                if epoch == 1 and batch_idx == 0:
                    log_msg(f"  [FAST-FEEDBACK K={K}] Epoch 1 Batch 1 Loss: {loss.item():.4f} | Radio: {radio_t:.2f}")
                    
            # Eval
            model.eval()
            correct = 0
            total = 0
            min_loc_error = 0.0
            
            with torch.no_grad():
                for imgs, labels, target_coords in test_loader:
                    imgs, labels, target_coords = imgs.to(device), labels.to(device), target_coords.to(device)
                    logits, cx, cy = model(imgs, current_radio=radio_t)
                    pred = logits.argmax(dim=1)
                    correct += (pred == labels).sum().item()
                    total += imgs.size(0)
                    
                    pred_heads = torch.stack([cx, cy], dim=-1)
                    target_exp = target_coords.unsqueeze(1)
                    dist_to_target = torch.norm(pred_heads - target_exp, p=2, dim=-1)
                    best_head_dist = dist_to_target.min(dim=1)[0].sum().item()
                    min_loc_error += best_head_dist
                    
            acc = 100.0 * correct / total
            avg_loc_err = min_loc_error / total
            t_epoch = time.time() - t0
            
            log_msg(f"  K={K:02d} | Epoch {epoch:02d}/{epochs:02d} | Acc: {acc:.2f}% | BestLocErr: {avg_loc_err:.4f} | Tiempo: {t_epoch:.2f}s")
            k_history.append({"epoch": epoch, "acc": acc, "loc_err": avg_loc_err})
            
        pei = k_history[-1]['acc'] / math.log10(total_params + 1)
        sweep_results[f"K_{K}"] = {
            "num_heads": K,
            "params": total_params,
            "final_acc": k_history[-1]['acc'],
            "final_loc_err": k_history[-1]['loc_err'],
            "pei": pei,
            "history": k_history
        }
        log_msg(f"=== RESULTADO K={K} -> Acc Final: {k_history[-1]['acc']:.2f}% | LocErr: {k_history[-1]['loc_err']:.4f} | PEI: {pei:.2f} | Params: {total_params} ===")

    # Tabulación Final
    log_msg("\n=========================================================================")
    log_msg("  RESUMEN FINAL BARRIDO DE CABEZAS FOVEALES (V363)")
    log_msg("=========================================================================")
    log_msg(f"{'Cabezas (K)':<12} | {'Parámetros':<12} | {'Test Acc (%)':<15} | {'Best LocErr':<15} | {'PEI':<10}")
    log_msg("-" * 72)
    for K in head_configs:
        res = sweep_results[f"K_{K}"]
        log_msg(f"{res['num_heads']:<12} | {res['params']:<12} | {res['final_acc']:<15.2f} | {res['final_loc_err']:<15.4f} | {res['pei']:<10.2f}")

    # Persistir JSON
    os.makedirs("results/raw", exist_ok=True)
    res_path = "results/raw/v363_copy_section_results.json"
    with open(res_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    log_msg(f"Resultados de barrido guardados en: {res_path}")

    # Ledger (Registrar la mejor configuración del barrido)
    best_k = max(head_configs, key=lambda k: sweep_results[f"K_{k}"]["final_acc"])
    best_res = sweep_results[f"K_{best_k}"]
    
    ledger_path = "results/master_ledger.jsonl"
    ledger_entry = {
        "experiment_id": "v363",
        "fecha": time.strftime("%Y-%m-%d"),
        "familia": " geometrico_atencion_foveal ",
        "dataset": "Cluttered MNIST (60x60, 10 clases)",
        "n_eval": 2000,
        "metric_name": "acc",
        "value": best_res['final_acc'],
        "SE": None,
        "params": best_res['params'],
        "nivel_rigor": 1,
        "etiqueta": "ANCLA" if best_res['final_acc'] > 45.0 else "SEÑAL"
    }
    with open(ledger_path, "a") as f:
        f.write(json.dumps(ledger_entry) + "\n")
    log_msg(f"Registrado en Master Ledger (Mejor K={best_k}): {ledger_path}")

if __name__ == '__main__':
    run_experiment()
