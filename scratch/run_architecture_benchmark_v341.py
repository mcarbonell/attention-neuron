import os
import sys
import time
import torch
import torch.nn as nn

# Asegurar que el directorio raíz del proyecto (padre de scratch/) esté en sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import get_dataloader
from src.models.standard_attention import StandardAttentionTransformer
from src.models.dynamic_iir_filter import DynamicIIRTransformer
from src.models.global_workspace import GlobalWorkspaceTransformer
from src.models.hybrid_iir_global import HybridIIRGlobalTransformer

def get_device():
    # 1. Probar DirectML para GPUs AMD Radeon en Windows
    try:
        import torch_directml
        device = torch_directml.device()
        print("[Device] Utilizando GPU AMD mediante torch-directml")
        return device
    except ImportError:
        pass
    
    # 2. Probar CUDA (si existiera)
    if torch.cuda.is_available():
        print("[Device] Utilizando GPU NVIDIA/CUDA")
        return torch.device("cuda")
        
    # 3. Fallback a CPU (Optimizado para Ryzen 7 8845HS)
    print("[Device] Utilizando CPU multinúcleo")
    return torch.device("cpu")

def train_and_eval(model, train_loader, epochs=12, lr=1e-3, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    history = {'loss': [], 'accuracy': []}
    
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
            
        epoch_loss = total_loss / total
        epoch_acc = (correct / total) * 100.0
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        
    return history

def measure_inference_latency(model, seq_len, batch_size=16, runs=15, device='cpu'):
    model = model.to(device)
    model.eval()
    dummy_input = torch.randint(2, 64, (batch_size, seq_len), dtype=torch.long).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
            
    # Medición
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)
    end_time = time.perf_counter()
    
    avg_latency_ms = ((end_time - start_time) / runs) * 1000.0
    return avg_latency_ms

def run_experiment_v341():
    print("=" * 75)
    print("   BENCHMARK V341: EXPERIMENTO DE ARQUITECTURAS INSPIRADAS EN SEÑALES")
    print("=" * 75)
    
    device = get_device()
    
    # Hiperparámetros de prueba v341
    vocab_size = 64
    d_model = 128
    seq_len_train = 256
    num_samples = 1200
    batch_size = 32
    epochs = 12
    
    print(f"\n[Config v341] Vocab Size: {vocab_size} | d_model: {d_model} | Seq Len Entrenar: {seq_len_train}")
    print(f"[Config v341] Muestras: {num_samples} | Épocas: {epochs} | Batch Size: {batch_size}\n")
    
    train_loader = get_dataloader(num_samples=num_samples, seq_len=seq_len_train, batch_size=batch_size, vocab_size=vocab_size)
    
    models = {
        "Standard Attention (Baseline)": StandardAttentionTransformer(vocab_size=vocab_size, d_model=d_model),
        "Dynamic IIR Filter (Idea 1)": DynamicIIRTransformer(vocab_size=vocab_size, d_model=d_model),
        "Global Workspace (Idea 6)": GlobalWorkspaceTransformer(vocab_size=vocab_size, d_model=d_model),
        "Hybrid IIR + Global": HybridIIRGlobalTransformer(vocab_size=vocab_size, d_model=d_model),
    }
    
    results = {}
    
    print("--- FASE 1: ENTRENAMIENTO Y CONVERGENCIA ---")
    for name, model in models.items():
        print(f"\n[Entrenando] {name}...")
        t0 = time.time()
        history = train_and_eval(model, train_loader, epochs=epochs, lr=1e-3, device=device)
        elapsed = time.time() - t0
        
        final_loss = history['loss'][-1]
        final_acc = history['accuracy'][-1]
        print(f" -> finalizado en {elapsed:.2f}s | Pérdida Final: {final_loss:.4f} | Precisión: {final_acc:.2f}%")
        
        results[name] = {
            'history': history,
            'final_acc': final_acc,
            'train_time': elapsed,
            'model': model
        }

    print("\n--- FASE 2: ESCALABILIDAD DE INFERENCIA EN SECUENCIAS LARGAS (Ms/Batch) ---")
    seq_lengths = [128, 256, 512, 1024]
    latency_table = {name: [] for name in models.keys()}
    
    for L in seq_lengths:
        print(f"\n[Testing Latencia] Longitud de Secuencia L = {L}")
        for name, data in results.items():
            lat_ms = measure_inference_latency(data['model'], seq_len=L, batch_size=16, runs=15, device=device)
            latency_table[name].append(lat_ms)
            print(f"   * {name:30s}: {lat_ms:7.2f} ms")

    print("\n" + "=" * 75)
    print("                 RESUMEN DE RESULTADOS EXPERIMENTO V341")
    print("=" * 75)
    print(f"{'Modelo':32s} | {'Acc (%)':8s} | {'128 (ms)':8s} | {'256 (ms)':8s} | {'512 (ms)':8s} | {'1024 (ms)':9s}")
    print("-" * 75)
    
    for name in models.keys():
        acc = results[name]['final_acc']
        lats = latency_table[name]
        print(f"{name:32s} | {acc:7.2f}% | {lats[0]:8.2f} | {lats[1]:8.2f} | {lats[2]:8.2f} | {lats[3]:9.2f}")
    
    print("=" * 75)

if __name__ == "__main__":
    run_experiment_v341()
