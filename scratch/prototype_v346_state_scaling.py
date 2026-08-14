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

from src.mqar_dataset import get_mqar_dataloader, MQARDataset
from src.models.selective_conv1d_iir_v346 import SelectiveConv1DIIRTransformerV346
from src.models.selective_conv1d_iir_v345 import SelectiveConv1DIIRTransformerV345, CausalInductionTransformer

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
        log_print(f" Capa {idx}: {layer_name:<15s} | Clase: {module.__class__.__name__:<35s} | Params: {mod_params:,}")
    log_print("---------------------------------------------------------")

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

def train_model(name, model, train_loader, val_dataset, epochs=40, lr=2e-3, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    eval_times = 0.0
    t_start_train = time.time()
    log_print(f"Comenzando entrenamiento para '{name}' ({epochs} épocas)...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (x, y) in enumerate(train_loader, 1):
            t_eval_start = time.time()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            t_eval_end = time.time()
            eval_times += (t_eval_end - t_eval_start)
            
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            # REGLA DE SUPERVIVENCIA (Fast Feedback): Imprimir primeros 5 batches de Época 1
            if epoch == 1 and batch_idx <= 5:
                log_print(f"   [Fast Feedback - Época 1 | Batch {batch_idx:02d}/{len(train_loader)}] Batch Loss: {loss.item():.4f}")
                
        scheduler.step()
        train_acc = (correct / total) * 100.0
        
        if epoch % 8 == 0 or epoch == epochs:
            val_acc = evaluate_accuracy(model, val_dataset, device=device)
            log_print(f"   [Época {epoch:02d}/{epochs}] Train Loss: {total_loss/total:.4f} | Train Acc: {train_acc:6.2f}% | Val Acc (L=128): {val_acc:6.2f}%")
            
    wall_clock_time = time.time() - t_start_train
    return {
        'model': model,
        'wall_clock_time': wall_clock_time,
        'eval_time': eval_times,
        'overhead_time': wall_clock_time - eval_times,
        'final_loss': total_loss / total,
        'final_train_acc': train_acc
    }

def run_experiment_v346():
    log_print("================================================================================")
    log_print(" EXPERIMENTO V346: ESCALADO DE ESTADO ESPACIAL (d_state=64) Y BENCHMARK MQAR")
    log_print("================================================================================")
    
    device, device_desc = get_device()
    commit_hash = get_git_commit()
    date_str = datetime.datetime.utcnow().isoformat() + "Z"
    
    log_print(f"ID Experimento: v346")
    log_print(f"Fecha (UTC): {date_str}")
    log_print(f"Commit Hash: {commit_hash}")
    log_print(f"Versión Python: {platform.python_version()} | PyTorch: {torch.__version__}")
    log_print(f"Dispositivo: {device} ({device_desc})")
    log_print(f"Plataforma OS: {platform.platform()}")
    
    config = {
        "experiment_id": "v346",
        "vocab_size": 64,
        "d_model": 128,
        "d_state_candidate": 64,
        "d_state_baseline": 16,
        "kernel_size": 4,
        "num_pairs": 8,
        "L_train": 128,
        "num_samples_train": 2400,
        "batch_size": 32,
        "epochs": 40,
        "learning_rate": 0.002,
        "scheduler": "CosineAnnealingLR",
        "rigor_level": 1,
        "eval_lengths": [128, 256, 512]
    }
    log_print(f"Configuración JSON: {json.dumps(config)}")
    
    train_loader = get_mqar_dataloader(
        num_samples=config["num_samples_train"],
        seq_len=config["L_train"],
        num_pairs=config["num_pairs"],
        batch_size=config["batch_size"],
        vocab_size=config["vocab_size"]
    )
    
    val_dataset_128 = MQARDataset(
        num_samples=400,
        seq_len=config["L_train"],
        num_pairs=config["num_pairs"],
        vocab_size=config["vocab_size"]
    )
    
    # REGLA DE ORO (Primero el Candidato):
    models = {
        "Selective-Conv1D IIR (v346 Candidato d_state=64)": SelectiveConv1DIIRTransformerV346(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            d_state=config["d_state_candidate"]
        ),
        "Causal Induction Transformer (Anthropic Circuit)": CausalInductionTransformer(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            nhead=4,
            num_layers=2
        ),
        "Selective-Conv1D IIR (v345 Baseline d_state=16)": SelectiveConv1DIIRTransformerV345(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            d_state=config["d_state_baseline"]
        )
    }
    
    for name, model in models.items():
        print_architecture_inventory(name, model)
        
    trained_results = {}
    
    log_print("--- FASE 1: ENTRENAMIENTO MQAR EN L = 128 (40 ÉPOCAS) ---")
    for name, model in models.items():
        log_print(f"\n[Entrenando] {name}...")
        res = train_model(
            name, model, train_loader, val_dataset_128,
            epochs=config["epochs"], lr=config["learning_rate"], device=device
        )
        log_print(f" -> {name} completado en {res['wall_clock_time']:.2f}s | Loss: {res['final_loss']:.4f}")
        trained_results[name] = res

    log_print("\n--- FASE 2: EVALUACIÓN ZERO-SHOT MQAR EN SECUENCIAS EXTENDIDAS ---")
    eval_results = {name: [] for name in models.keys()}
    num_eval_samples = 400
    
    for L_eval in config["eval_lengths"]:
        log_print(f"\n[Evaluando MQAR] Generando test con L = {L_eval}...")
        test_dataset = MQARDataset(
            num_samples=num_eval_samples,
            seq_len=L_eval,
            num_pairs=config["num_pairs"],
            vocab_size=config["vocab_size"]
        )
        
        for name, data in trained_results.items():
            acc = evaluate_accuracy(data['model'], test_dataset, batch_size=config["batch_size"], device=device)
            eval_results[name].append(acc)
            log_print(f"   * {name:50s} | Acc L={L_eval:4d}: {acc:6.2f}%")

    log_print("\n================================================================================")
    log_print("          RESUMEN EXPERIMENTO V346 (STATE SCALING d_state=64)")
    log_print("================================================================================")
    log_print(f"{'Modelo':50s} | {'L=128 (Train)':13s} | {'L=256':9s} | {'L=512':9s}")
    log_print("-" * 87)
    
    raw_results_data = {}
    for name in models.keys():
        accs = eval_results[name]
        log_print(f"{name:50s} | {accs[0]:12.2f}% | {accs[1]:8.2f}% | {accs[2]:8.2f}%")
        
        t_tot, t_trainable = count_parameters(models[name])
        raw_results_data[name] = {
            "params": t_tot,
            "final_loss": trained_results[name]["final_loss"],
            "wall_clock_time": trained_results[name]["wall_clock_time"],
            "eval_time": trained_results[name]["eval_time"],
            "overhead_time": trained_results[name]["overhead_time"],
            "accuracies": {
                "L_128": accs[0],
                "L_256": accs[1],
                "L_512": accs[2]
            }
        }
    log_print("================================================================================")

    # PERSISTENCIA
    results_dir = os.path.join(PROJECT_ROOT, "results", "raw")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "v346_results.json")
    
    full_output = {
        "metadata": {
            "experiment_id": "v346",
            "date_utc": date_str,
            "commit_hash": commit_hash,
            "device": str(device),
            "device_desc": device_desc,
            "platform": platform.platform(),
            "config": config
        },
        "models": raw_results_data
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    log_print(f"Resultados persistidos en: {json_path}")
    
    # LEDGER
    cand_acc = eval_results["Selective-Conv1D IIR (v346 Candidato d_state=64)"][0]
    cand_params = raw_results_data["Selective-Conv1D IIR (v346 Candidato d_state=64)"]["params"]
    
    ledger_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(ledger_dir, exist_ok=True)
    ledger_path = os.path.join(ledger_dir, "master_ledger.jsonl")
    
    ledger_entry = {
        "experiment_id": "v346",
        "fecha": datetime.datetime.utcnow().strftime("%Y-%m-%d"),
        "familia": "espectral / seleccional / state_scaling",
        "dataset": "MQARDataset (2400 seqs / L=128 / d_state=64)",
        "n_eval": 2400 * config["epochs"],
        "metric_name": "accuracy",
        "value": cand_acc,
        "SE": None,
        "params": cand_params,
        "nivel_rigor": 1,
        "etiqueta": "ANCLA" if cand_acc > 50.0 else ("SEÑAL" if cand_acc > 10.0 else "RUIDO-SOSPECHA")
    }
    
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ledger_entry) + "\n")
    log_print(f"Línea añadida a Master Ledger: {ledger_path}")

if __name__ == "__main__":
    run_experiment_v346()
