import os
import sys
import time
import json
import datetime
import platform
import subprocess
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DELTA_PHASE_PATH = os.path.join(os.path.dirname(PROJECT_ROOT), "delta-phase")
if DELTA_PHASE_PATH not in sys.path:
    sys.path.insert(0, DELTA_PHASE_PATH)

from src.mqar_dataset import get_mqar_dataloader, MQARDataset
from src.models.selective_conv1d_iir_v345 import CausalInductionTransformer
from delta_phase.layers import DeltaPhaseHolographicBlock

START_TIME = time.time()

def log_print(msg: str):
    elapsed = time.time() - START_TIME
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    timestamp = f"[+{hours:02d}:{minutes:02d}:{seconds:05.2f}]"
    print(f"{timestamp} {msg}", flush=True)

def get_git_commit():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=PROJECT_ROOT).decode('utf-8').strip()
        return commit
    except Exception:
        return "unknown"

def get_device():
    try:
        import torch_directml
        dev = torch_directml.device()
        return dev, "AMD GPU (torch-directml)"
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda"), "NVIDIA CUDA GPU"
    return torch.device("cpu"), f"CPU ({platform.processor() or 'Multicore'})"

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def print_architecture_inventory(name, model):
    log_print(f"--- INVENTARIO DE ARQUITECTURA: {name} ---")
    total, trainable = count_parameters(model)
    log_print(f"Parámetros Totales: {total:,} | Entrenables: {trainable:,}")
    for idx, (layer_name, module) in enumerate(model.named_children()):
        mod_params = sum(p.numel() for p in module.parameters())
        log_print(f" Capa {idx}: {layer_name:<15s} | Clase: {module.__class__.__name__:<45s} | Params: {mod_params:,}")
    log_print("---------------------------------------------------------")

class DeltaPhaseMQARModel(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 128, n_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DeltaPhaseHolographicBlock(d_model=d_model, n_heads=n_heads, chunk_size=32) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        out = self.embedding(x)
        for layer in self.layers:
            out, _ = layer(out)
        out = self.norm(out)
        logits = self.fc_out(out[:, -1, :])
        return logits

def evaluate_accuracy(model, dataset, batch_size=32, device='cpu'):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return (correct / total) * 100.0

def train_and_eval_config(name_prefix, model_class, num_pairs, seq_len, vocab_size=256, epochs=25, lr=2e-3, device='cpu'):
    train_loader = get_mqar_dataloader(
        num_samples=2400,
        seq_len=seq_len,
        num_pairs=num_pairs,
        batch_size=32,
        vocab_size=vocab_size
    )
    val_dataset = MQARDataset(
        num_samples=300,
        seq_len=seq_len,
        num_pairs=num_pairs,
        vocab_size=vocab_size
    )
    
    if model_class == DeltaPhaseMQARModel:
        model = DeltaPhaseMQARModel(vocab_size=vocab_size, d_model=128, n_heads=4, num_layers=2)
    else:
        model = CausalInductionTransformer(vocab_size=vocab_size, d_model=128, nhead=4, num_layers=2)
        
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    t_start = time.time()
    log_print(f"\n[Evaluación Frontera] {name_prefix} | Pairs: {num_pairs:2d} | SeqLen: {seq_len:4d}...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            if epoch == 1 and batch_idx <= 5 and "DeltaPhase" in name_prefix:
                log_print(f"   [Fast Feedback - Época 1 | Batch {batch_idx:02d}] Loss: {loss.item():.4f}")
                
        scheduler.step()
        train_acc = (correct / total) * 100.0
        
        if epoch % 5 == 0 or epoch == epochs:
            val_acc = evaluate_accuracy(model, val_dataset, device=device)
            log_print(f"   [Época {epoch:02d}/{epochs}] Train Loss: {total_loss/total:.4f} | Train Acc: {train_acc:6.2f}% | Val Acc: {val_acc:6.2f}%")
            
    wall_t = time.time() - t_start
    final_val_acc = evaluate_accuracy(model, val_dataset, device=device)
    return {
        "final_loss": total_loss / total,
        "train_acc": train_acc,
        "val_acc": final_val_acc,
        "wall_clock": wall_t,
        "params": count_parameters(model)[0]
    }

def run_experiment_v350():
    log_print("================================================================================")
    log_print(" EXPERIMENTO V350: BARRIDO DE LA FRONTERA DE CAPACIDAD MATRICIAL DELTAPHASE")
    log_print("================================================================================")
    
    device, device_desc = get_device()
    commit_hash = get_git_commit()
    date_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    log_print(f"ID Experimento: v350")
    log_print(f"Fecha (UTC): {date_str}")
    log_print(f"Commit Hash: {commit_hash}")
    log_print(f"Versión Python: {platform.python_version()} | PyTorch: {torch.__version__}")
    log_print(f"Dispositivo: {device} ({device_desc})")
    log_print(f"Plataforma OS: {platform.platform()}")
    
    test_configs = [
        {"pairs": 8,   "L": 128},
        {"pairs": 16,  "L": 256},
        {"pairs": 32,  "L": 512},
        {"pairs": 64,  "L": 1024}
    ]
    
    vocab_size = 256
    log_print(f"Configuración Barrido: VocabSize={vocab_size}, Pairs Sweep: [8, 16, 32, 64]")
    
    # REGLA DE ORO: Candidato DeltaPhase ejecuta primero en cada configuración
    results = {"DeltaPhase": {}, "Transformer": {}}
    
    for cfg in test_configs:
        p = cfg["pairs"]
        L = cfg["L"]
        key = f"P{p}_L{L}"
        
        # 1. Candidato DeltaPhase
        res_dp = train_and_eval_config("DeltaPhase (Complex C^(32x32))", DeltaPhaseMQARModel, num_pairs=p, seq_len=L, vocab_size=vocab_size, epochs=25, device=device)
        results["DeltaPhase"][key] = res_dp
        
        # 2. Baseline Transformer
        res_tf = train_and_eval_config("Causal Transformer (Baseline)", CausalInductionTransformer, num_pairs=p, seq_len=L, vocab_size=vocab_size, epochs=25, device=device)
        results["Transformer"][key] = res_tf

    log_print("\n================================================================================")
    log_print("          RESUMEN EXPERIMENTO V350 (FRONTERA DE CAPACIDAD MQAR)")
    log_print("================================================================================")
    log_print(f"{'Configuración':15s} | {'DeltaPhase Acc':16s} | {'Transformer Acc':16s} | {'DeltaPhase Loss':16s} | {'Ventaja DeltaPhase'}")
    log_print("-" * 95)
    
    for cfg in test_configs:
        p = cfg["pairs"]
        L = cfg["L"]
        key = f"P{p}_L{L}"
        acc_dp = results["DeltaPhase"][key]["val_acc"]
        acc_tf = results["Transformer"][key]["val_acc"]
        loss_dp = results["DeltaPhase"][key]["final_loss"]
        diff = acc_dp - acc_tf
        log_print(f"P={p:2d}, L={L:4d}     | {acc_dp:15.2f}% | {acc_tf:15.2f}% | {loss_dp:16.4f} | +{diff:.2f}%")
        
    log_print("================================================================================")

    # PERSISTENCIA
    results_dir = os.path.join(PROJECT_ROOT, "results", "raw")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "v350_results.json")
    
    full_output = {
        "metadata": {
            "experiment_id": "v350",
            "date_utc": date_str,
            "commit_hash": commit_hash,
            "device": str(device),
            "device_desc": device_desc,
            "platform": platform.platform()
        },
        "results": results
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    log_print(f"Resultados persistidos en: {json_path}")
    
    # LEDGER
    dp_p16_acc = results["DeltaPhase"]["P16_L256"]["val_acc"]
    dp_params = results["DeltaPhase"]["P16_L256"]["params"]
    
    ledger_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(ledger_dir, exist_ok=True)
    ledger_path = os.path.join(ledger_dir, "master_ledger.jsonl")
    
    ledger_entry = {
        "experiment_id": "v350",
        "fecha": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
        "familia": "fase_compleja / regladelta / capacidad",
        "dataset": "MQARDataset (Sweep P8..P64 / L128..L1024)",
        "n_eval": 2400 * 25,
        "metric_name": "accuracy",
        "value": dp_p16_acc,
        "SE": None,
        "params": dp_params,
        "nivel_rigor": 1,
        "etiqueta": "ANCLA" if dp_p16_acc > 50.0 else ("SEÑAL" if dp_p16_acc > 10.0 else "RUIDO-SOSPECHA")
    }
    
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ledger_entry) + "\n")
    log_print(f"Línea añadida a Master Ledger: {ledger_path}")

if __name__ == "__main__":
    run_experiment_v350()
