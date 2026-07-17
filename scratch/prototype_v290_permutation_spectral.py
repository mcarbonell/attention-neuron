import torch
import torch.nn as nn
import numpy as np
import time
import os
import urllib.request
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuración de dispositivo
device = "cpu"  # CPU por defecto para consistencia y estabilidad en CPU local

# URL y path de Tiny Shakespeare
DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
DATA_PATH = 'scratch/data/tiny_shakespeare.txt'

def load_evaluation_data(tokenizer):
    """
    Descarga Tiny Shakespeare si no existe y extrae 20 secuencias de 512 tokens.
    """
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        print(f"Descargando Tiny Shakespeare de {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
        
    print("Tokenizando texto de Tiny Shakespeare...")
    tokens = tokenizer.encode(text)
    
    # 20 secuencias de 512 tokens = 10,240 tokens
    n_seq = 20
    seq_len = 512
    required_tokens = n_seq * seq_len
    
    if len(tokens) < required_tokens:
        raise ValueError(f"El texto tiene solo {len(tokens)} tokens, se necesitan {required_tokens}")
        
    input_ids = torch.tensor(tokens[:required_tokens]).view(n_seq, seq_len)
    return input_ids

# ══════════════════════════════════════════════════════════════════════
# ALGORITMOS DE ORDENACIÓN / PERMUTACIÓN
# ══════════════════════════════════════════════════════════════════════

def get_pca_permutation(W):
    """
    Ordena las columnas de W (tamaño d_model, d_mlp) proyectándolas en el 1er componente principal (PCA 1D).
    """
    # Centramos las columnas
    W_centered = W - W.mean(dim=1, keepdim=True)
    # SVD: W_centered = U @ diag(S) @ Vh
    # U: (768, 768), S: (768), Vh: (768, 3072)
    U, S, Vh = torch.linalg.svd(W_centered, full_matrices=False)
    # El primer autovector singular derecho (Vh[0, :]) representa las coordenadas en el 1er componente principal
    scores = Vh[0, :]
    perm = torch.argsort(scores)
    return perm

def get_greedy_tsp_permutation(W):
    """
    Heurística Greedy TSP (Vecino Más Cercano) O(N^2) para reordenar las columnas de W.
    """
    cols = W.t()  # (3072, 768)
    n = cols.size(0)
    
    # Distancia euclídea eficiente: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    norms = (cols ** 2).sum(dim=1)
    dist_mat = norms.unsqueeze(1) + norms.unsqueeze(0) - 2 * (cols @ cols.t())
    dist_mat = torch.clamp(dist_mat, min=0.0).sqrt()
    
    visited = torch.zeros(n, dtype=torch.bool, device=W.device)
    perm = torch.zeros(n, dtype=torch.long, device=W.device)
    
    # Empezamos en la columna de mayor norma
    current = torch.argmax(norms).item()
    perm[0] = current
    visited[current] = True
    
    for i in range(1, n):
        dists = dist_mat[current].clone()
        dists[visited] = float('inf')  # Ignorar nodos ya visitados
        next_node = torch.argmin(dists).item()
        perm[i] = next_node
        visited[next_node] = True
        current = next_node
        
    return perm

def get_fiedler_permutation(W):
    """
    Ordenación espectral usando el vector de Fiedler (segundo autovector más pequeño del Laplaciano del grafo).
    """
    cols = W.t()  # (3072, 768)
    norms = (cols ** 2).sum(dim=1)
    dist_sq = norms.unsqueeze(1) + norms.unsqueeze(0) - 2 * (cols @ cols.t())
    dist_sq = torch.clamp(dist_sq, min=0.0)
    
    # Escala Gaussiana sigma_sq (mediana de las distancias al cuadrado)
    sigma_sq = torch.median(dist_sq)
    if sigma_sq < 1e-6:
        sigma_sq = 1.0
        
    A = torch.exp(-dist_sq / (2.0 * sigma_sq))
    A.fill_diagonal_(0.0)  # Sin autolazos
    
    D = A.sum(dim=1)
    L = torch.diag(D) - A
    
    # Autovalores y autovectores (eigh es para matrices simétricas reales)
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    
    # Segundo autovector más pequeño
    fiedler_vector = eigenvectors[:, 1]
    perm = torch.argsort(fiedler_vector)
    return perm

# ══════════════════════════════════════════════════════════════════════
# TRANSFORMADA DCT-1D Y COMPRESIÓN
# ══════════════════════════════════════════════════════════════════════

def get_dct_matrix_1d(N, device):
    """Matriz DCT-II unidimensional ortogonal."""
    n = torch.arange(N, device=device).view(1, -1)
    k = torch.arange(N, device=device).view(-1, 1)
    M = torch.cos(np.pi * k * (2 * n + 1) / (2 * N))
    M[0, :] *= 1 / np.sqrt(2)
    M *= np.sqrt(2 / N)
    return M

def dct1d(W, dim):
    N = W.shape[dim]
    M = get_dct_matrix_1d(N, W.device)
    if dim == 1:
        return W @ M.t()
    else:  # dim == 0
        return M @ W

def idct1d(W_dct, dim):
    N = W_dct.shape[dim]
    M = get_dct_matrix_1d(N, W_dct.device)
    if dim == 1:
        return W_dct @ M
    else:  # dim == 0
        return M.t() @ W_dct

def compress_lowpass_1d(W, keep_ratio, dim):
    """Filtro Paso Bajo DCT-1D (JPG Slice)"""
    N = W.shape[dim]
    W_dct = dct1d(W, dim=dim)
    
    mask = torch.zeros_like(W_dct)
    limit = int(N * keep_ratio)
    
    if dim == 1:
        mask[:, :limit] = 1.0
    else:
        mask[:limit, :] = 1.0
        
    W_dct_compressed = W_dct * mask
    return idct1d(W_dct_compressed, dim=dim)

def compress_energy_1d(W, keep_ratio, dim):
    """Umbral de Energía DCT-1D (JPG Coefs)"""
    W_dct = dct1d(W, dim=dim)
    flat_dct = W_dct.flatten()
    k = int(flat_dct.numel() * keep_ratio)
    if k > 0:
        values, _ = torch.topk(torch.abs(flat_dct), k)
        threshold = values[-1]
        mask = torch.abs(W_dct) >= threshold
        W_dct_compressed = W_dct * mask
    else:
        W_dct_compressed = torch.zeros_like(W_dct)
        
    return idct1d(W_dct_compressed, dim=dim)

# ══════════════════════════════════════════════════════════════════════
# APLICACIÓN DE PERMUTACIONES Y COMPRESIÓN AL MODELO
# ══════════════════════════════════════════════════════════════════════

def permute_model_mlp(model, method="pca"):
    """
    Permuta localmente los pesos del MLP en todos los bloques de GPT-2.
    """
    print(f"Permutando capas MLP usando el método: {method.upper()}...")
    for i in range(len(model.transformer.h)):
        mlp = model.transformer.h[i].mlp
        
        # c_fc: (d_model, d_mlp) = (768, 3072)
        # Queremos ordenar las 3072 columnas de c_fc
        W_fc = mlp.c_fc.weight.data
        
        if method == "pca":
            perm = get_pca_permutation(W_fc)
        elif method == "tsp":
            perm = get_greedy_tsp_permutation(W_fc)
        elif method == "fiedler":
            perm = get_fiedler_permutation(W_fc)
        else:
            raise ValueError(f"Método desconocido: {method}")
            
        # Reordenar c_fc (columnas del peso y elementos del bias)
        mlp.c_fc.weight.data = mlp.c_fc.weight.data[:, perm]
        mlp.c_fc.bias.data = mlp.c_fc.bias.data[perm]
        
        # Reordenar c_proj (filas del peso para alinearse con la salida permutada de c_fc)
        mlp.c_proj.weight.data = mlp.c_proj.weight.data[perm, :]

def apply_spectral_compression_1d(model, keep_ratio, method="lowpass"):
    """
    Aplica compresión DCT-1D sobre las dimensiones ordenadas de MLP c_fc y c_proj.
    """
    for i in range(len(model.transformer.h)):
        mlp = model.transformer.h[i].mlp
        
        # Capa c_fc: weight es (768, 3072). La dimensión de los canales permutados es dim=1.
        W_fc = mlp.c_fc.weight.data.clone()
        orig_std_fc = W_fc.std()
        
        if method == "lowpass":
            W_fc_rec = compress_lowpass_1d(W_fc, keep_ratio, dim=1)
        elif method == "energy":
            W_fc_rec = compress_energy_1d(W_fc, keep_ratio, dim=1)
            
        # Variance Rescaling
        if W_fc_rec.std() > 0:
            W_fc_rec = W_fc_rec * (orig_std_fc / W_fc_rec.std())
        mlp.c_fc.weight.data = W_fc_rec
        
        # Capa c_proj: weight es (3072, 768). La dimensión de los canales permutados es dim=0.
        W_proj = mlp.c_proj.weight.data.clone()
        orig_std_proj = W_proj.std()
        
        if method == "lowpass":
            W_proj_rec = compress_lowpass_1d(W_proj, keep_ratio, dim=0)
        elif method == "energy":
            W_proj_rec = compress_energy_1d(W_proj, keep_ratio, dim=0)
            
        # Variance Rescaling
        if W_proj_rec.std() > 0:
            W_proj_rec = W_proj_rec * (orig_std_proj / W_proj_rec.std())
        mlp.c_proj.weight.data = W_proj_rec

# ══════════════════════════════════════════════════════════════════════
# BUCLUES DE EVALUACIÓN
# ══════════════════════════════════════════════════════════════════════

def evaluate_ppl(model, input_ids):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i in range(input_ids.size(0)):
            inputs = input_ids[i:i+1].to(device)
            outputs = model(inputs, labels=inputs)
            total_loss += outputs.loss.item()
    mean_loss = total_loss / input_ids.size(0)
    return np.exp(mean_loss)

def test_equivalence(input_ids):
    """
    Comprueba si las permutaciones son matemáticamente equivalentes al float32 original.
    """
    print("\n--- Verificación de Equivalencia Matemática ---")
    model_ref = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    ref_ppl = evaluate_ppl(model_ref, input_ids)
    print(f"PPL del modelo de referencia (original): {ref_ppl:.6f}")
    
    methods = ["pca", "tsp", "fiedler"]
    for m in methods:
        model_test = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        permute_model_mlp(model_test, method=m)
        test_ppl = evaluate_ppl(model_test, input_ids)
        delta = abs(test_ppl - ref_ppl)
        print(f"Método {m.upper()} | PPL: {test_ppl:.6f} | Delta: {delta:.2e} "
              f"{'[OK] EXCELENTE (Equivalente)' if delta < 1e-4 else '[ERROR] ERROR (Roto)'}")

def run_compression_benchmark(input_ids):
    """
    Evalúa la compresión espectral 1D con y sin permutación en diferentes keep_ratios.
    """
    print("\n--- Iniciando Benchmark de Compresión Espectral v290 ---")
    
    # Primero obtenemos la perplejidad base sin compresión
    model_baseline = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    base_ppl = evaluate_ppl(model_baseline, input_ids)
    print(f"Perplejidad Baseline (Float32): {base_ppl:.4f}\n")
    
    # Ratios de compresión a evaluar
    ratios = [0.9, 0.7, 0.5, 0.3, 0.1]
    
    # Escenarios
    # (Nombre escenario, método_ordenacion_previo, método_compresion_dct)
    scenarios = [
        ("Espacial / Frecuencial directo (Sin ordenar, Lowpass)", None, "lowpass"),
        ("Espectral PCA (Ordenado PCA, Lowpass)", "pca", "lowpass"),
        ("Espectral Greedy TSP (Ordenado TSP, Lowpass)", "tsp", "lowpass"),
        ("Espectral Fiedler (Ordenado Fiedler, Lowpass)", "fiedler", "lowpass"),
        ("Espacial / Frecuencial directo (Sin ordenar, Energy)", None, "energy"),
        ("Espectral PCA (Ordenado PCA, Energy)", "pca", "energy"),
        ("Espectral Fiedler (Ordenado Fiedler, Energy)", "fiedler", "energy"),
    ]
    
    # Tabla de resultados
    header = f"{'Escenario':<55} | " + " | ".join([f"Ratio {r:.1f}" for r in ratios])
    print(header)
    print("-" * len(header))
    
    for label, ord_method, comp_method in scenarios:
        row_str = f"{label:<55} | "
        ppl_values = []
        for r in ratios:
            # Cargar modelo limpio
            model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
            
            # Aplicar ordenación si corresponde
            if ord_method is not None:
                permute_model_mlp(model, method=ord_method)
                
            # Aplicar compresión
            apply_spectral_compression_1d(model, keep_ratio=r, method=comp_method)
            
            # Evaluar
            ppl = evaluate_ppl(model, input_ids)
            ppl_values.append(ppl)
            
        row_str += " | ".join([f"{p:9.2f}" if p < 10000 else f"{'Explosión':>9}" for p in ppl_values])
        print(row_str)

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimento v290: Reordenación de Pesos Espectral")
    parser.add_argument("--check_equivalence", action="store_true", help="Verifica si las permutaciones rompen el modelo")
    parser.add_argument("--run_benchmark", action="store_true", help="Corre la comparativa completa de compresión")
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = load_evaluation_data(tokenizer)
    
    # Si no se proveen argumentos, por defecto hacemos la validación básica y el benchmark
    if not args.check_equivalence and not args.run_benchmark:
        test_equivalence(input_ids)
        run_compression_benchmark(input_ids)
    else:
        if args.check_equivalence:
            test_equivalence(input_ids)
        if args.run_benchmark:
            run_compression_benchmark(input_ids)
