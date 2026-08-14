import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import get_dataloader, DynamicAssociativeRecallDataset
from src.models.standard_attention import StandardAttentionTransformer
from src.models.dynamic_iir_filter import DynamicIIRTransformer
from src.models.selective_iir_filter import SelectiveDynamicIIRTransformer

def get_device():
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

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

def train_model(model, train_loader, val_dataset, epochs=20, lr=3e-3, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
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
            
        scheduler.step()
        train_acc = (correct / total) * 100.0
        
        if epoch % 4 == 0 or epoch == epochs:
            val_acc = evaluate_accuracy(model, val_dataset, device=device)
            print(f"   [Época {epoch:2d}/{epochs}] Loss: {total_loss/total:.4f} | Train Acc: {train_acc:6.2f}% | Val Acc (L=256): {val_acc:6.2f}%")
            
    return model

def run_experiment_v343():
    print("=" * 80)
    print(" EXPERIMENTO V343: COMPUERTA SELECTIVA (SELECTIVE-GATE IIR) VS RUIDO")
    print("=" * 80)
    
    device = get_device()
    vocab_size = 64
    d_model = 128
    L_train = 256
    num_samples_train = 1600
    batch_size = 32
    epochs = 20
    
    print(f"\n[Config v343]")
    print(f" * Entrenando con Generación Dinámica en L_train = {L_train}")
    print(f" * Muestras: {num_samples_train} | Épocas: {epochs} | LR: 3e-3 Cosine")
    print(f" * Evaluando Zero-Shot en Longitudes L_eval = [256, 512, 1024, 2048]\n")
    
    train_loader = get_dataloader(
        num_samples=num_samples_train,
        seq_len=L_train,
        batch_size=batch_size,
        vocab_size=vocab_size
    )
    
    val_dataset_256 = DynamicAssociativeRecallDataset(
        num_samples=400,
        seq_len=L_train,
        num_pairs=4,
        vocab_size=vocab_size
    )
    
    models = {
        "Standard Attention (Baseline)": StandardAttentionTransformer(vocab_size=vocab_size, d_model=d_model, max_len=4096),
        "Dynamic IIR Filter (v341/v342)": DynamicIIRTransformer(vocab_size=vocab_size, d_model=d_model),
        "Selective-Gate IIR (v343 Nuevo)": SelectiveDynamicIIRTransformer(vocab_size=vocab_size, d_model=d_model),
    }
    
    trained_models = {}
    print("--- FASE 1: ENTRENAMIENTO DINÁMICO EN L = 256 ---")
    for name, model in models.items():
        print(f"\n[Entrenando] {name} en L={L_train}...")
        t0 = time.time()
        model = train_model(model, train_loader, val_dataset_256, epochs=epochs, lr=3e-3, device=device)
        elapsed = time.time() - t0
        print(f" -> Entrenamiento completado en {elapsed:.2f}s")
        trained_models[name] = model

    print("\n--- FASE 2: EVALUACIÓN ZERO-SHOT EN SECUENCIAS EXTENDIDAS ---")
    eval_lengths = [256, 512, 1024, 2048]
    eval_results = {name: [] for name in models.keys()}
    
    num_eval_samples = 400
    for L_eval in eval_lengths:
        print(f"\n[Evaluando] Generando dataset de test con L = {L_eval}...")
        test_dataset = DynamicAssociativeRecallDataset(
            num_samples=num_eval_samples,
            seq_len=L_eval,
            num_pairs=4,
            vocab_size=vocab_size
        )
        
        for name, model in trained_models.items():
            acc = evaluate_accuracy(model, test_dataset, batch_size=32, device=device)
            eval_results[name].append(acc)
            print(f"   * {name:32s} | Accuracy en L={L_eval:4d}: {acc:6.2f}%")

    print("\n" + "=" * 80)
    print("          RESUMEN EXPERIMENTO V343 (SELECTIVE-GATE IIR)")
    print("=" * 80)
    print(f"{'Modelo':32s} | {'L=256 (Train)':13s} | {'L=512':9s} | {'L=1024':9s} | {'L=2048':9s}")
    print("-" * 80)
    
    for name in models.keys():
        accs = eval_results[name]
        print(f"{name:32s} | {accs[0]:12.2f}% | {accs[1]:8.2f}% | {accs[2]:8.2f}% | {accs[3]:8.2f}%")
    
    print("=" * 80)

if __name__ == "__main__":
    run_experiment_v343()
