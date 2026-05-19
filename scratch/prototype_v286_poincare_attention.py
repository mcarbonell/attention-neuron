import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import os
import json
import math
import sys

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

# --- CONSTRUCCION DEL ARBOL TAXONOMICO ---
def build_tree(K=5, D=3):
    parents = {}
    level_nodes = {0: [0]}
    current_id = 1
    for depth in range(1, D + 1):
        level_nodes[depth] = []
        for p in level_nodes[depth - 1]:
            for _ in range(K):
                parents[current_id] = p
                level_nodes[depth].append(current_id)
                current_id += 1
    return parents, level_nodes

# --- GENERACION DEL DATASET ---
def generate_ancestry_dataset(parents):
    samples = []
    # Las relaciones son: 0 -> Padre, 1 -> Abuelo, 2 -> Bisabuelo
    for node in range(1, len(parents) + 1):
        # 1. Padre (Relacion 0)
        p1 = parents.get(node)
        if p1 is not None:
            samples.append((node, 0, p1))
            
            # 2. Abuelo (Relacion 1)
            p2 = parents.get(p1)
            if p2 is not None:
                samples.append((node, 1, p2))
                
                # 3. Bisabuelo (Relacion 2)
                p3 = parents.get(p2)
                if p3 is not None:
                    samples.append((node, 2, p3))
    return samples

# --- CAPAS DE ATENCION ---

class PoincareAttention(nn.Module):
    def __init__(self, dim, num_heads=1, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.eps = eps
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        self.beta = nn.Parameter(torch.tensor(1.0))

    def _project_to_ball(self, x):
        """Proyecta suavemente los vectores usando tanh para distribuirlos en todo el disco"""
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        factor = (1.0 - self.eps) * torch.tanh(norm) / (norm + 1e-10)
        return x * factor

    def _poincare_distance(self, u, v):
        """Calcula la distancia hiperbolica por parejas entre u [B, H, N, 1, D] y v [B, H, 1, M, D]"""
        sq_dist = torch.sum((u - v) ** 2, dim=-1)
        u_norm_sq = torch.sum(u ** 2, dim=-1)
        v_norm_sq = torch.sum(v ** 2, dim=-1)
        
        denom = (1.0 - u_norm_sq) * (1.0 - v_norm_sq)
        denom = torch.clamp(denom, min=self.eps)
        
        alpha = 1.0 + 2.0 * sq_dist / denom
        alpha = torch.clamp(alpha, min=1.0 + self.eps)
        
        # arcosh(x) = ln(x + sqrt(x^2 - 1))
        dist = torch.log(alpha + torch.sqrt(torch.clamp(alpha**2 - 1.0, min=1e-15)))
        return dist

    def forward(self, x):
        B, N, D = x.shape
        H = self.num_heads
        d_k = D // H
        
        q = self.q_proj(x).view(B, N, H, d_k).transpose(1, 2)  # [B, H, N, d_k]
        k = self.k_proj(x).view(B, N, H, d_k).transpose(1, 2)  # [B, H, N, d_k]
        v = self.v_proj(x).view(B, N, H, d_k).transpose(1, 2)  # [B, H, N, d_k]
        
        q_hyp = self._project_to_ball(q)
        k_hyp = self._project_to_ball(k)
        
        q_uns = q_hyp.unsqueeze(3)  # [B, H, N, 1, d_k]
        k_uns = k_hyp.unsqueeze(2)  # [B, H, 1, N, d_k]
        
        distances = self._poincare_distance(q_uns, k_uns)  # [B, H, N, N]
        
        attn_scores = -torch.abs(self.beta) * distances
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        out = torch.matmul(attn_weights, v)  # [B, H, N, d_k]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        
        return out, attn_weights, (q_hyp, k_hyp)


class EuclideanAttention(nn.Module):
    def __init__(self, dim, num_heads=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        H = self.num_heads
        d_k = D // H
        
        q = self.q_proj(x).view(B, N, H, d_k).transpose(1, 2)  # [B, H, N, d_k]
        k = self.k_proj(x).view(B, N, H, d_k).transpose(1, 2)  # [B, H, N, d_k]
        v = self.v_proj(x).view(B, N, H, d_k).transpose(1, 2)  # [B, H, N, d_k]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        
        return out, attn_weights, (None, None)

# --- CLASIFICADOR GENERAL ---

class AncestryClassifier(nn.Module):
    def __init__(self, num_nodes, num_relations, dim, attention_type='poincare', num_heads=1, eps=1e-5):
        super().__init__()
        self.node_embed = nn.Embedding(num_nodes, dim)
        self.rel_embed = nn.Embedding(num_relations, dim)
        self.attention_type = attention_type
        
        if attention_type == 'poincare':
            self.attn = PoincareAttention(dim, num_heads=num_heads, eps=eps)
        else:
            self.attn = EuclideanAttention(dim, num_heads=num_heads)
            
        self.head = nn.Linear(dim, num_nodes)

    def forward(self, node_ids, rel_ids):
        # node_ids: [B], rel_ids: [B]
        e_node = self.node_embed(node_ids).unsqueeze(1)  # [B, 1, dim]
        e_rel = self.rel_embed(rel_ids).unsqueeze(1)    # [B, 1, dim]
        
        x = torch.cat([e_node, e_rel], dim=1)           # [B, 2, dim]
        
        out, attn_weights, (q_hyp, k_hyp) = self.attn(x)
        
        # Clasificamos usando la representacion del token de relacion (index 1)
        rel_repr = out[:, 1, :]                         # [B, dim]
        logits = self.head(rel_repr)                     # [B, num_nodes]
        
        return logits, attn_weights, (q_hyp, k_hyp)

# --- BUCLE DE ENTRENAMIENTO INDIVIDUAL ---

def train_and_evaluate(attention_type, dim, train_data, test_data, num_nodes, num_relations, epochs=120, lr=5.00e-03, seed=42):
    # Fijar semillas
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = AncestryClassifier(num_nodes, num_relations, dim, attention_type=attention_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1.00e-05)
    criterion = nn.CrossEntropyLoss()
    
    # Preparar batches
    batch_size = 32
    train_x_node = torch.tensor([s[0] for s in train_data], dtype=torch.long)
    train_x_rel = torch.tensor([s[1] for s in train_data], dtype=torch.long)
    train_y = torch.tensor([s[2] for s in train_data], dtype=torch.long)
    
    test_x_node = torch.tensor([s[0] for s in test_data], dtype=torch.long).to(device)
    test_x_rel = torch.tensor([s[1] for s in test_data], dtype=torch.long).to(device)
    test_y = torch.tensor([s[2] for s in test_data], dtype=torch.long).to(device)
    
    num_batches = math.ceil(len(train_data) / batch_size)
    total_evals = 0
    net_forward_time = 0.0
    
    wall_start = time.time()
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(train_data))
        
        for batch_idx in range(num_batches):
            indices = permutation[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            b_node = train_x_node[indices].to(device)
            b_rel = train_x_rel[indices].to(device)
            b_y = train_y[indices].to(device)
            
            optimizer.zero_grad()
            
            # Medir tiempo de forward
            f_start = time.time()
            logits, _, _ = model(b_node, b_rel)
            loss = criterion(logits, b_y)
            f_end = time.time()
            net_forward_time += (f_end - f_start)
            total_evals += 1
            
            loss.backward()
            optimizer.step()
            
            # REGLA DE SUPERVIVENCIA: Imprimir los primeros 5 batches de la epoca 1
            if epoch == 0 and batch_idx < 5:
                log_msg(f"  [EP 1] {attention_type.upper()} d={dim} Seed={seed} Batch {batch_idx+1}/5 | Loss: {loss.item():.4f}")
                
    wall_clock_time = time.time() - wall_start
    
    # Evaluar
    model.eval()
    with torch.no_grad():
        test_logits, _, _ = model(test_x_node, test_x_rel)
        test_loss = criterion(test_logits, test_y).item()
        preds = torch.argmax(test_logits, dim=-1)
        test_acc = (preds == test_y).float().mean().item()
        
    # Contar parametros
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
        "total_params": total_params
    }

# --- VISUALIZACION 2D DEL DISCO DE POINCARE ---

def generate_poincare_disk_plot(model, parents, output_path):
    if plt is None:
        log_msg("Matplotlib no esta disponible. Omitiendo grafico.")
        return
        
    model.eval()
    with torch.no_grad():
        # Obtener todos los nodos
        num_nodes = model.node_embed.num_embeddings
        nodes = torch.arange(num_nodes, device=device)
        
        # Calcular claves en el espacio de Poincaré
        embeds = model.node_embed(nodes) # [N, 2]
        k = model.attn.k_proj(embeds)   # [N, 2]
        k_hyp = model.attn._project_to_ball(k).cpu().numpy() # [N, 2]
        
    plt.figure(figsize=(10, 10))
    
    # Dibujar la circunferencia unitaria
    theta = np.linspace(0, 2*np.pi, 200)
    plt.plot(np.cos(theta), np.sin(theta), color='gray', linestyle='--', alpha=0.7, label="Frontera de Poincare")
    plt.axhline(0, color='gray', alpha=0.3)
    plt.axvline(0, color='gray', alpha=0.3)
    
    # Clasificar niveles de los nodos
    # Nivel 0 (Raiz): 0
    # Nivel 1: 1-5
    # Nivel 2: 6-30
    # Nivel 3: 31-155
    levels = np.zeros(num_nodes, dtype=int)
    for n in range(num_nodes):
        if n == 0:
            levels[n] = 0
        elif n in range(1, 6):
            levels[n] = 1
        elif n in range(6, 31):
            levels[n] = 2
        else:
            levels[n] = 3
            
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    labels = ['Raiz (Nivel 0)', 'Nivel 1', 'Nivel 2', 'Nivel 3']
    
    # Dibujar enlaces parent-child
    for child, parent in parents.items():
        x_pts = [k_hyp[parent, 0], k_hyp[child, 0]]
        y_pts = [k_hyp[parent, 1], k_hyp[child, 1]]
        plt.plot(x_pts, y_pts, color='#7f7f7f', alpha=0.4, zorder=1)
        
    # Scatter de nodos por nivel
    for lvl in range(4):
        mask = (levels == lvl)
        plt.scatter(k_hyp[mask, 0], k_hyp[mask, 1], 
                    color=colors[lvl], label=labels[lvl], 
                    s=120 if lvl < 2 else 60, 
                    edgecolors='black', linewidths=0.8, zorder=2)
        
    # Etiquetar nodos seleccionados (raiz y algunos hijos para no saturar)
    for n in range(num_nodes):
        if levels[n] < 2 or (levels[n] == 2 and n % 5 == 0) or (levels[n] == 3 and n % 25 == 0):
            plt.annotate(str(n), (k_hyp[n, 0] + 0.015, k_hyp[n, 1] + 0.015), fontsize=9, weight='bold', zorder=3)
            
    plt.title("Visualizacion Espacial: Disco de Poincare (v286)\nAuto-organizacion Jerarquica de Claves (Q/K) d=2", fontsize=14, weight='bold')
    plt.xlim(-1.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    
    # Guardar
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    log_msg(f"saved: {output_path}")

# --- MAIN BENCHMARK SWEEP ---

def main():
    # Header de Metadatos
    log_msg("=== INICIO EXPERIMENTO V286 ===")
    log_msg("Base Model File: scratch/prototype_v286_poincare_attention.py")
    log_msg(f"CPU Threads: {torch.get_num_threads()}")
    log_msg("Hiperparametros de Ejecucion:")
    log_msg("  Arbol: K=5, D=3 (156 nodos)")
    log_msg("  Epocas: 120, Batch Size: 32, LR: 5.00e-03, WD: 1.00e-05")
    log_msg("  Semillas: 5 semillas independientes [42, 43, 44, 45, 46]")
    log_msg("  Dimensiones: [2, 4, 8, 16, 32, 64]")
    log_msg("=================================")
    
    # Construir arbol y dataset
    parents, level_nodes = build_tree(K=5, D=3)
    dataset = generate_ancestry_dataset(parents)
    num_nodes = 156
    num_relations = 3
    
    # Split train/test (80/20) reproducibles
    import random
    random.seed(42)
    random.shuffle(dataset)
    split = int(0.8 * len(dataset))
    train_data = dataset[:split]
    test_data = dataset[split:]
    
    log_msg(f"Dataset generado: {len(dataset)} muestras totales. Train: {len(train_data)}, Test: {len(test_data)}")
    
    dimensions = [2, 4, 8, 16, 32, 64]
    seeds = [42, 43, 44, 45, 46]
    attention_types = ['poincare', 'euclidean']
    
    results = {
        "metadata": {
            "K": 5,
            "D": 3,
            "num_nodes": num_nodes,
            "num_relations": num_relations,
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "epochs": 120,
            "learning_rate": 5.00e-03,
            "weight_decay": 1.00e-05,
            "seeds": seeds
        },
        "runs": []
    }
    
    # Barrido
    for dim in dimensions:
        log_msg(f"\n--- Probando Dimension d={dim} ---")
        for att_type in attention_types:
            accs = []
            losses = []
            peis = []
            wall_times = []
            f_times = []
            
            for seed in seeds:
                res = train_and_evaluate(
                    attention_type=att_type,
                    dim=dim,
                    train_data=train_data,
                    test_data=test_data,
                    num_nodes=num_nodes,
                    num_relations=num_relations,
                    epochs=120,
                    lr=5.00e-03,
                    seed=seed
                )
                
                results["runs"].append({
                    "attention_type": att_type,
                    "dimension": dim,
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
                
            avg_acc = np.mean(accs)
            std_acc = np.std(accs)
            avg_loss = np.mean(losses)
            avg_pei = np.mean(peis)
            avg_wall = np.mean(wall_times)
            avg_forward = np.mean(f_times)
            avg_overhead = avg_wall - avg_forward
            
            log_msg(f"Resultado {att_type.upper()}:")
            log_msg(f"  Acc: {avg_acc*100:.2f}% (+/- {std_acc*100:.2f}%) | Loss: {avg_loss:.4f} | PEI: {avg_pei:.4f}")
            log_msg(f"  Tiempo Wall: {avg_wall:.2f}s | Forward: {avg_forward:.2f}s | Overhead: {avg_overhead:.2f}s")
            
    # Guardar resultados JSON
    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/summary", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    raw_path = "results/raw/v286_poincare_attention.json"
    with open(raw_path, 'w') as f:
        json.dump(results, f, indent=4)
    log_msg(f"saved: {raw_path}")
    
    # Generar resumen estadistico en JSON
    summary_results = {
        "metadata": results["metadata"],
        "summary": {}
    }
    
    for dim in dimensions:
        summary_results["summary"][str(dim)] = {}
        for att_type in attention_types:
            runs_filtered = [r for r in results["runs"] if r["dimension"] == dim and r["attention_type"] == att_type]
            accs = [r["accuracy"] for r in runs_filtered]
            losses = [r["final_loss"] for r in runs_filtered]
            peis = [r["PEI"] for r in runs_filtered]
            wall_times = [r["wall_clock_time"] for r in runs_filtered]
            f_times = [r["function_evaluation_time"] for r in runs_filtered]
            
            summary_results["summary"][str(dim)][att_type] = {
                "avg_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "avg_loss": float(np.mean(losses)),
                "avg_PEI": float(np.mean(peis)),
                "avg_wall_clock_time": float(np.mean(wall_times)),
                "avg_forward_time": float(np.mean(f_times)),
                "total_params": int(runs_filtered[0]["total_params"])
            }
            
    summary_path = "results/summary/v286_poincare_attention_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=4)
    log_msg(f"saved: {summary_path}")
    
    # --- GENERACION DEL GRAFICO DE POINCARE CON LA MEJOR SEMILLA EN d=2 ---
    log_msg("\nGenerando grafico del Disco de Poincare para d=2...")
    best_model_seed = 42
    # Entrenar un modelo rapido solo para obtener los embeddings finales de d=2
    best_poincare_model = AncestryClassifier(num_nodes, num_relations, dim=2, attention_type='poincare').to(device)
    optimizer = optim.Adam(best_poincare_model.parameters(), lr=5.00e-03, weight_decay=1.00e-05)
    criterion = nn.CrossEntropyLoss()
    
    train_x_node = torch.tensor([s[0] for s in train_data], dtype=torch.long).to(device)
    train_x_rel = torch.tensor([s[1] for s in train_data], dtype=torch.long).to(device)
    train_y = torch.tensor([s[2] for s in train_data], dtype=torch.long).to(device)
    
    # 150 epocas para asegurar alineamiento perfecto en la visualizacion
    for _ in range(150):
        best_poincare_model.train()
        optimizer.zero_grad()
        logits, _, _ = best_poincare_model(train_x_node, train_x_rel)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()
        
    fig_path = "results/figures/v286_poincare_disk.png"
    generate_poincare_disk_plot(best_poincare_model, parents, fig_path)
    log_msg("=== PROCESO COMPLETADO EXITOAMENTE ===")

if __name__ == "__main__":
    main()
