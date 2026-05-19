import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import json
import math

# Para la visualizacion
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# --- CONFIGURACION DE TIEMPO Y LOGGING ---
global_start_time = time.time()

def log_msg(msg):
    elapsed = time.time() - global_start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print(f"[{hours:02d}:{minutes:02d}:{seconds:02d}] {msg}")

# --- DETECCION DE DISPOSITIVO ---
device = torch.device('cpu')
log_msg("Detector de dispositivo: Ejecutando en CPU para optimizar latencia en red pequena")

# --- CAPA CONFORMAL LINEAR ---

class ConformalLinear(nn.Module):
    def __init__(self, in_features, out_features, num_coefficients=6, base_resolution=128, seed=42):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_coefficients = num_coefficients
        self.base_resolution = base_resolution
        
        # Generar base estática congelante (Textura geométrica aleatoria)
        # Usamos un generador local con semilla para que sea reproducible independientemente del orden de ejecución
        g = torch.Generator()
        g.manual_seed(seed)
        
        # Base de pesos 2D de alta resolución (congelada)
        base_tensor = torch.randn(1, 1, base_resolution, base_resolution, generator=g)
        # Inicialización de tipo Kaiming para texturas ricas
        nn.init.kaiming_normal_(base_tensor, a=math.sqrt(5))
        self.base_weights = nn.Parameter(base_tensor, requires_grad=False)
        
        # Coordenadas complejas base para el mapeo (línea regular en [-1, 1])
        z_cols = torch.linspace(-1.0, 1.0, steps=in_features)
        # Las tratamos como números complejos con parte imaginaria cero inicialmente
        self.register_buffer("z_cols", torch.complex(z_cols, torch.zeros_like(z_cols)))
        
        # Precomputar potencias de z para acelerar la multiplicación conformal: [num_coefficients, in_features]
        z_powers = []
        for n in range(1, num_coefficients + 1):
            z_powers.append(torch.pow(self.z_cols, n))
        self.register_buffer("z_powers", torch.stack(z_powers, dim=0))
        
        # Parámetros del mapa conformal: Coeficientes complejos por cada neurona de salida (a_n = alpha + i * beta)
        # Inicializados cerca de cero para que al principio el mapa sea casi la identidad: f(z) ~ z
        self.coeff_real = nn.Parameter(torch.randn(out_features, num_coefficients, generator=g) * 0.05)
        self.coeff_imag = nn.Parameter(torch.randn(out_features, num_coefficients, generator=g) * 0.05)
        
        # Parámetros de ganancia y sesgo por neurona (para ajustar dinámicamente la escala y el umbral)
        self.gain = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def get_weights(self):
        # 1. Construir coeficientes complejos: [out_features, num_coefficients]
        coeffs = torch.complex(self.coeff_real, self.coeff_imag)
        
        # 2. Mapeo Conformal Vectorizado (Producto matricial sobre plano complejo):
        # coeffs: [out_features, num_coefficients]
        # z_powers: [num_coefficients, in_features]
        # Producto da el término de perturbación: [out_features, in_features]
        perturbation = torch.matmul(coeffs, self.z_powers)
        
        # f(z) = z + perturbation
        w_complex = self.z_cols.unsqueeze(0) + perturbation  # [out_features, in_features]
        
        # 3. Mapeo Holomórfico de Frontera (Complex Tanh):
        # Aplicamos la tangente hiperbólica compleja para proyectar armónicamente sobre el disco/cuadrado unitario
        # Para estabilidad y compatibilidad con grid_sample, usamos tanh sobre la parte real e imaginaria de forma segura
        u_scaled = torch.tanh(w_complex.real)  # Eje vertical (filas)
        v_scaled = torch.tanh(w_complex.imag)  # Eje horizontal (columnas)
        
        # 4. Muestreo Tomográfico con grid_sample:
        # grid expects shape: [N, H_out, W_out, 2]
        # grid[..., 0] -> horizontal (columnas/eje v)
        # grid[..., 1] -> vertical (filas/eje u)
        grid = torch.stack([v_scaled, u_scaled], dim=-1).unsqueeze(0)  # [1, out_features, in_features, 2]
        
        # Muestreamos de la textura congelada usando interpolación bilineal y reflexión en los bordes
        sampled_weights = F.grid_sample(
            self.base_weights, 
            grid, 
            mode='bilinear', 
            padding_mode='reflection', 
            align_corners=True
        )  # [1, 1, out_features, in_features]
        
        weights = sampled_weights.squeeze(0).squeeze(0)  # [out_features, in_features]
        
        # 5. Escalamiento He/Kaiming adaptativo
        scale = math.sqrt(2.0 / self.in_features)
        weights = weights * scale * self.gain.unsqueeze(1)
        
        return weights

    def forward(self, x):
        # Obtener los pesos generados dinámicamente
        weights = self.get_weights()
        return F.linear(x, weights, self.bias)

# --- ARQUITECTURAS ---

class ConformalMLP(nn.Module):
    def __init__(self, hidden_size=128, num_coefficients=6, seed=42):
        super().__init__()
        # Usamos ConformalLinear para la capa grande de representación
        self.layer1 = ConformalLinear(784, hidden_size, num_coefficients=num_coefficients, seed=seed)
        # Capa de clasificación final estándar
        self.layer2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.layer1(x))
        return self.layer2(x)


class BaselineMLP(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.layer1 = nn.Linear(784, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

# --- BUCLE DE ENTRENAMIENTO INDIVIDUAL ---

def train_and_evaluate(model_type, hidden_size, train_loader, test_loader, epochs=10, lr=1.00e-03, seed=42, num_coefficients=6):
    # Fijar semillas para reproducibilidad
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if model_type == 'conformal':
        model = ConformalMLP(hidden_size=hidden_size, num_coefficients=num_coefficients, seed=seed).to(device)
    else:
        model = BaselineMLP(hidden_size=hidden_size).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    total_evals = 0
    net_forward_time = 0.0
    
    wall_start = time.time()
    
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Medir tiempo neto de forward
            f_start = time.time()
            output = model(data)
            loss = criterion(output, target)
            f_end = time.time()
            net_forward_time += (f_end - f_start)
            total_evals += 1
            
            loss.backward()
            optimizer.step()
            
            # REGLA DE SUPERVIVENCIA: Imprimir los primeros 5 batches de la época 1
            if epoch == 0 and batch_idx < 5:
                log_msg(f"  [EP 1] {model_type.upper()} hidden={hidden_size} Seed={seed} Batch {batch_idx+1}/5 | Loss: {loss.item():.4f}")
                
    wall_clock_time = time.time() - wall_start
    
    # Evaluar en Test
    model.eval()
    correct = 0
    total_test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_test_loss += criterion(output, target).item() * len(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_acc = correct / len(test_loader.dataset)
    test_loss = total_test_loss / len(test_loader.dataset)
    
    # Contar parámetros entrenables
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # PEI: Accuracy / log10(TotalParams + 1)
    pei = test_acc / math.log10(total_params + 1)
    
    return {
        "final_loss": test_loss,
        "accuracy": test_acc,
        "total_evaluations": total_evals,
        "wall_clock_time": wall_clock_time,
        "function_evaluation_time": net_forward_time,
        "internal_overhead_time": wall_clock_time - net_forward_time,
        "PEI": pei,
        "total_params": total_params,
        "model": model
    }

# --- VISUALIZACION GEOMETRICA ---

def plot_conformal_mapping(model, output_path):
    if plt is None:
        log_msg("Matplotlib no está disponible. Omitiendo gráfico.")
        return
        
    model.eval()
    with torch.no_grad():
        conformal_layer = model.layer1
        
        # 1. Obtener coeficientes y coordenadas conformes
        coeffs = torch.complex(conformal_layer.coeff_real, conformal_layer.coeff_imag)
        perturbation = torch.matmul(coeffs, conformal_layer.z_powers)
        w_complex = conformal_layer.z_cols.unsqueeze(0) + perturbation
        
        # Mapeamos a numpy
        w_np = w_complex.cpu().numpy()  # [out_features, in_features]
        z_np = conformal_layer.z_cols.cpu().numpy()  # [in_features]
        
        # Generar los pesos mapeados
        weights = conformal_layer.get_weights().cpu().numpy()  # [out_features, in_features]

    # Crear figura con 3 paneles
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Panel 1: Curvatura de Coordenadas en el Plano Complejo para algunas neuronas
    ax1 = axes[0]
    # Dibujar la línea de entrada original (eje real)
    ax1.plot(z_np.real, z_np.imag, color='black', linestyle='--', linewidth=2, label='Input Grid Original (1D)')
    
    # Dibujar la trayectoria transformada por el mapa conformal f(z) para las primeras 5 neuronas
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for idx in range(min(5, conformal_layer.out_features)):
        ax1.plot(w_np[idx].real, w_np[idx].imag, color=colors[idx], linewidth=1.8,
                 label=f'Neurona {idx+1} Mapeada')
        # Marcar los puntos extremos para ver el sentido del estiramiento
        ax1.scatter([w_np[idx, 0].real, w_np[idx, -1].real], [w_np[idx, 0].imag, w_np[idx, -1].imag], 
                    color=colors[idx], s=40, zorder=3)
        
    ax1.set_title("Deformación del Plano Complejo $z \\to f(z)$\n(Conformidad Local y Curvatura)", fontsize=12, weight='bold')
    ax1.set_xlabel("Re (Parte Real)")
    ax1.set_ylabel("Im (Parte Imaginaria)")
    ax1.grid(True, which='both', linestyle=':', alpha=0.5)
    ax1.legend(loc='upper right', frameon=True)
    
    # Panel 2: Distribución en la rejilla grid_sample con Tanh holomórfica
    ax2 = axes[1]
    # Círculo unitario de referencia
    theta = np.linspace(0, 2*np.pi, 200)
    ax2.plot(np.cos(theta), np.sin(theta), color='gray', linestyle=':', label='Frontera Unitario')
    
    for idx in range(min(5, conformal_layer.out_features)):
        u_scaled = np.tanh(w_np[idx].real)
        v_scaled = np.tanh(w_np[idx].imag)
        ax2.plot(v_scaled, u_scaled, color=colors[idx], linewidth=1.8, label=f'Trayectoria N{idx+1}')
        
    ax2.set_title("Coordenadas Finales en Rejilla $W_{base}$\n(Compresión por Tanh Holomórfica)", fontsize=12, weight='bold')
    ax2.set_xlabel("Eje Horizontal (v)")
    ax2.set_ylabel("Eje Vertical (u)")
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, which='both', linestyle=':', alpha=0.5)
    ax2.legend(loc='lower left', frameon=True)
    
    # Panel 3: Visualización de la Matriz de Pesos Generada final
    ax3 = axes[2]
    im = ax3.imshow(weights[:64, :128], cmap='RdBu', aspect='auto', interpolation='nearest')
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_title("Matriz de Pesos Proyectada $W_{conformal}$ (Zoom 64x128)\n(Patrones Continuos y Armónicos)", fontsize=12, weight='bold')
    ax3.set_xlabel("Inputs (Dimensión de Entrada)")
    ax3.set_ylabel("Outputs (Neuronas de Salida)")
    
    plt.suptitle("Óptica Conforme (v287): Proyecciones en el Plano Complejo para Generación de Pesos", fontsize=15, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    log_msg(f"saved: {output_path}")

# --- MAIN BENCHMARK SWEEP ---

def main():
    log_msg("=== INICIO EXPERIMENTO V287: ÓPTICA CONFORME ===")
    log_msg("Base Model File: scratch/prototype_v287_conformal_optics.py")
    log_msg(f"CPU Threads: {torch.get_num_threads()}")
    
    # Cargar Dataset MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    log_msg(f"Dataset cargado correctamente. Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Hiperparámetros de Ejecución
    HIDDEN_SIZE = 128
    EPOCHS = 5
    LR = 0.001
    NUM_COEFFS = 6
    seeds = [42, 43, 44, 45, 46]
    model_types = ['conformal', 'baseline']
    
    log_msg(f"Hiperparámetros de Ejecución:")
    log_msg(f"  Hidden Size: {HIDDEN_SIZE} | Architecture: 784 -> {HIDDEN_SIZE} -> 10")
    log_msg(f"  Coeficientes Conformes: {NUM_COEFFS}")
    log_msg(f"  Epocas: {EPOCHS} | Batch Size: 2048 | Learning Rate: {LR:.2e}")
    log_msg(f"  Semillas: {seeds}")
    log_msg("==================================================")
    
    results = {
        "metadata": {
            "hidden_size": HIDDEN_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "num_coefficients": NUM_COEFFS,
            "seeds": seeds
        },
        "runs": []
    }
    
    best_conformal_model = None
    best_conformal_acc = 0.0
    
    for model_type in model_types:
        log_msg(f"\n--- Evaluando Modelo: {model_type.upper()} ---")
        accs = []
        losses = []
        peis = []
        wall_times = []
        f_times = []
        params = 0
        
        for seed in seeds:
            res = train_and_evaluate(
                model_type=model_type,
                hidden_size=HIDDEN_SIZE,
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=EPOCHS,
                lr=LR,
                seed=seed,
                num_coefficients=NUM_COEFFS
            )
            
            results["runs"].append({
                "model_type": model_type,
                "seed": seed,
                "final_loss": res["final_loss"],
                "accuracy": res["accuracy"],
                "total_evaluations": res["total_evaluations"],
                "wall_clock_time": res["wall_clock_time"],
                "function_evaluation_time": res["function_evaluation_time"],
                "internal_overhead_time": res["internal_overhead_time"],
                "PEI": res["PEI"],
                "total_params": res["total_params"]
            })
            
            accs.append(res["accuracy"])
            losses.append(res["final_loss"])
            peis.append(res["PEI"])
            wall_times.append(res["wall_clock_time"])
            f_times.append(res["function_evaluation_time"])
            params = res["total_params"]
            
            # Guardar el mejor modelo conformal para graficar
            if model_type == 'conformal' and res["accuracy"] > best_conformal_acc:
                best_conformal_acc = res["accuracy"]
                best_conformal_model = res["model"]
                
        avg_acc = np.mean(accs)
        std_acc = np.std(accs)
        avg_loss = np.mean(losses)
        avg_pei = np.mean(peis)
        avg_wall = np.mean(wall_times)
        avg_forward = np.mean(f_times)
        avg_overhead = avg_wall - avg_forward
        
        log_msg(f"Resultado {model_type.upper()}:")
        log_msg(f"  Params Totales: {params:,}")
        log_msg(f"  Acc: {avg_acc*100:.2f}% (+/- {std_acc*100:.2f}%) | Loss: {avg_loss:.4f} | PEI: {avg_pei:.4f}")
        log_msg(f"  Tiempo Wall: {avg_wall:.2f}s | Forward: {avg_forward:.2f}s | Overhead: {avg_overhead:.2f}s")
        
    # Guardar resultados JSON
    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/summary", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    raw_path = "results/raw/v287_conformal_optics.json"
    with open(raw_path, 'w') as f:
        json.dump(results, f, indent=4)
    log_msg(f"saved: {raw_path}")
    
    # Resumen estadístico
    summary_results = {
        "metadata": results["metadata"],
        "summary": {}
    }
    for model_type in model_types:
        runs_filtered = [r for r in results["runs"] if r["model_type"] == model_type]
        accs = [r["accuracy"] for r in runs_filtered]
        losses = [r["final_loss"] for r in runs_filtered]
        peis = [r["PEI"] for r in runs_filtered]
        wall_times = [r["wall_clock_time"] for r in runs_filtered]
        f_times = [r["function_evaluation_time"] for r in runs_filtered]
        
        summary_results["summary"][model_type] = {
            "avg_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "avg_loss": float(np.mean(losses)),
            "avg_PEI": float(np.mean(peis)),
            "avg_wall_clock_time": float(np.mean(wall_times)),
            "avg_forward_time": float(np.mean(f_times)),
            "total_params": int(runs_filtered[0]["total_params"])
        }
        
    summary_path = "results/summary/v287_conformal_optics_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=4)
    log_msg(f"saved: {summary_path}")
    
    # Generar gráficos del mapeo para el mejor modelo conformal
    if best_conformal_model is not None:
        log_msg("\nGenerando gráfico de visualización de la Óptica Conforme...")
        fig_path = "results/figures/v287_conformal_optics.png"
        plot_conformal_mapping(best_conformal_model, fig_path)
        
    log_msg("=== PROCESO COMPLETADO EXITOSAMENTE ===")

if __name__ == "__main__":
    main()
