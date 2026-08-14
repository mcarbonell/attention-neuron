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

# Asegurar importación de paquetes del proyecto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import get_dataloader, DynamicAssociativeRecallDataset
from src.models.selective_conv1d_iir import SelectiveConv1DIIRTransformer
from src.models.dynamic_iir_filter import DynamicIIRTransformer
from src.models.standard_attention import StandardAttentionTransformer

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
        log_print(f" Capa {idx}: {layer_name:<15s} | Clase: {module.__class__.__name__:<30s} | Params: {mod_params:,}")
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

def train_model(name, model, train_loader, val_dataset, epochs=20, lr=3e-3, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    eval_times = 0.0
    overhead_times = 0.0
    
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
            t_eval_end = time.time()
            eval_times += (t_eval_end - t_eval_start)
            
            t_over_start = time.time()
            optimizer.step()
            t_over_end = time.time()
            overhead_times += (t_over_end - t_over_start)
            
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            # REGLA DE SUPERVIVENCIA (Fast Feedback): Imprimir primeros 5 batches de Época 1
            if epoch == 1 and batch_idx <= 5:
                log_print(f"   [Fast Feedback - Época 1 | Batch {batch_idx:02d}/{len(train_loader)}] Batch Loss: {loss.item():.4f}")
                
        scheduler.step()
        train_acc = (correct / total) * 100.0
        
        if epoch % 4 == 0 or epoch == epochs:
            val_acc = evaluate_accuracy(model, val_dataset, device=device)
            log_print(f"   [Época {epoch:02d}/{epochs}] Train Loss: {total_loss/total:.4f} | Train Acc: {train_acc:6.2f}% | Val Acc (L=256): {val_acc:6.2f}%")
            
    wall_clock_time = time.time() - t_start_train
    return {
        'model': model,
        'wall_clock_time': wall_clock_time,
        'eval_time': eval_times,
        'overhead_time': wall_clock_time - eval_times,
        'final_loss': total_loss / total,
        'final_train_acc': train_acc
    }

def run_experiment_v344():
    log_print("================================================================================")
    log_print(" EXPERIMENTO V344: CONVOLUCIÓN CAUSAL 1D + COMPUERTA SELECTIVA (CONV1D + SSM)")
    log_print("================================================================================")
    
    device, device_desc = get_device()
    commit_hash = get_git_commit()
    date_str = datetime.datetime.utcnow().isoformat() + "Z"
    
    # 1. METADATOS DE EJECUCIÓN COMPLETOS
    log_print(f"ID Experimento: v344")
    log_print(f"Fecha (UTC): {date_str}")
    log_print(f"Commit Hash: {commit_hash}")
    log_print(f"Versión Python: {platform.python_version()} | PyTorch: {torch.__version__}")
    log_print(f"Dispositivo: {device} ({device_desc})")
    log_print(f"Plataforma OS: {platform.platform()}")
    
    # 2. CONFIGURACIÓN REPRODUCIBLE COMPLETA (JSON)
    config = {
        "experiment_id": "v344",
        "vocab_size": 64,
        "d_model": 128,
        "d_state": 16,
        "kernel_size": 4,
        "L_train": 256,
        "num_samples_train": 1600,
        "batch_size": 32,
        "epochs": 20,
        "learning_rate": 0.003,
        "scheduler": "CosineAnnealingLR",
        "rigor_level": 1,
        "eval_lengths": [256, 512, 1024, 2048]
    }
    log_print(f"Configuración JSON: {json.dumps(config)}")
    
    train_loader = get_dataloader(
        num_samples=config["num_samples_train"],
        seq_len=config["L_train"],
        batch_size=config["batch_size"],
        vocab_size=config["vocab_size"]
    )
    
    val_dataset_256 = DynamicAssociativeRecallDataset(
        num_samples=400,
        seq_len=config["L_train"],
        num_pairs=4,
        vocab_size=config["vocab_size"]
    )
    
    # REGLA DE ORO (Primero el Candidato):
    # Candidato v344 se ejecuta PRIMERO, luego los baselines
    models = {
        "Selective-Conv1D IIR (v344 Candidato)": SelectiveConv1DIIRTransformer(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            d_state=config["d_state"]
        ),
        "Dynamic IIR Filter (Baseline v341)": DynamicIIRTransformer(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"]
        ),
        "Standard Attention (Baseline)": StandardAttentionTransformer(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            max_len=4096
        )
    }
    
    # Imprimir inventario de arquitecturas
    for name, model in models.items():
        print_architecture_inventory(name, model)
        
    trained_results = {}
    
    log_print("--- FASE 1: ENTRENAMIENTO DINÁMICO EN L = 256 ---")
    for name, model in models.items():
        log_print(f"\n[Entrenando] {name}...")
        res = train_model(
            name, model, train_loader, val_dataset_256,
            epochs=config["epochs"], lr=config["learning_rate"], device=device
        )
        log_print(f" -> {name} completado en {res['wall_clock_time']:.2f}s | Loss: {res['final_loss']:.4f}")
        trained_results[name] = res

    log_print("\n--- FASE 2: EVALUACIÓN ZERO-SHOT EN SECUENCIAS EXTENDIDAS ---")
    eval_results = {name: [] for name in models.keys()}
    num_eval_samples = 400
    
    for L_eval in config["eval_lengths"]:
        log_print(f"\n[Evaluando Zero-Shot] Generando test con L = {L_eval}...")
        test_dataset = DynamicAssociativeRecallDataset(
            num_samples=num_eval_samples,
            seq_len=L_eval,
            num_pairs=4,
            vocab_size=config["vocab_size"]
        )
        
        for name, data in trained_results.items():
            acc = evaluate_accuracy(data['model'], test_dataset, batch_size=config["batch_size"], device=device)
            eval_results[name].append(acc)
            log_print(f"   * {name:38s} | Acc L={L_eval:4d}: {acc:6.2f}%")

    log_print("\n================================================================================")
    log_print("          RESUMEN EXPERIMENTO V344 (CONV1D + SELECTIVE IIR)")
    log_print("================================================================================")
    log_print(f"{'Modelo':38s} | {'L=256 (Train)':13s} | {'L=512':9s} | {'L=1024':9s} | {'L=2048':9s}")
    log_print("-" * 85)
    
    raw_results_data = {}
    for name in models.keys():
        accs = eval_results[name]
        log_print(f"{name:38s} | {accs[0]:12.2f}% | {accs[1]:8.2f}% | {accs[2]:8.2f}% | {accs[3]:8.2f}%")
        
        t_tot, t_trainable = count_parameters(models[name])
        raw_results_data[name] = {
            "params": t_tot,
            "final_loss": trained_results[name]["final_loss"],
            "wall_clock_time": trained_results[name]["wall_clock_time"],
            "eval_time": trained_results[name]["eval_time"],
            "overhead_time": trained_results[name]["overhead_time"],
            "accuracies": {
                "L_256": accs[0],
                "L_512": accs[1],
                "L_1024": accs[2],
                "L_2048": accs[3]
            }
        }
    log_print("================================================================================")

    # PERSISTENCIA: Guardar resultados en results/raw/v344_results.json
    results_dir = os.path.join(PROJECT_ROOT, "results", "raw")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "v344_results.json")
    
    full_output = {
        "metadata": {
            "experiment_id": "v344",
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
    
    # LEDGER: Añadir línea a results/master_ledger.jsonl
    cand_acc = eval_results["Selective-Conv1D IIR (v344 Candidato)"][0]
    cand_params = raw_results_data["Selective-Conv1D IIR (v344 Candidato)"]["params"]
    
    ledger_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(ledger_dir, exist_ok=True)
    ledger_path = os.path.join(ledger_dir, "master_ledger.jsonl")
    
    ledger_entry = {
        "experiment_id": "v344",
        "fecha": datetime.datetime.utcnow().strftime("%Y-%m-%d"),
        "familia": "espectral / seleccional",
        "dataset": "DynamicAssociativeRecall (1600 seqs / L=256)",
        "n_eval": 1600 * config["epochs"],
        "metric_name": "accuracy",
        "value": cand_acc,
        "SE": None,
        "params": cand_params,
        "nivel_rigor": 1,
        "etiqueta": "SEÑAL" if cand_acc > 10.0 else "RUIDO-SOSPECHA"
    }
    
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ledger_entry) + "\n")
    log_print(f"Línea añadida a Master Ledger: {ledger_path}")

if __name__ == "__main__":
    run_experiment_v344()
