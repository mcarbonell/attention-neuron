import torch
import numpy as np
import collections
import heapq
import os
import json

# --- HUFFMAN CODING (Simulación de Compresión Sin Pérdida) ---

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(frequencies):
    heap = [HuffmanNode(char, freq) for char, freq in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)
    return heap[0]

def get_huffman_codes(node, prefix="", codes={}):
    if node:
        if node.char is not None:
            codes[node.char] = prefix
        get_huffman_codes(node.left, prefix + "0", codes)
        get_huffman_codes(node.right, prefix + "1", codes)
    return codes

# --- EXPERIMENTO V198 ---

def spectral_transform(x):
    # Walsh-Hadamard simplificado
    n = x.size(-1)
    if n == 1: return x
    x_e, x_o = x[..., 0::2], x[..., 1::2]
    return torch.cat([spectral_transform(x_e + x_o), spectral_transform(x_e - x_o)], dim=-1)

def run_entropy_compression_test():
    print(">>> V198: ENTROPY-SPECTRAL HYBRID COMPRESSION")
    
    # 1. Generar señal y transformarla
    n = 256
    t = torch.linspace(0, 1, n)
    x = torch.sin(2 * np.pi * 5 * t) + 0.3 * torch.randn(n)
    coeffs = spectral_transform(x)
    
    # 2. Compresión con Pérdida (Top-K)
    k = 32
    val, idx = torch.topk(torch.abs(coeffs), k)
    sparse_coeffs = torch.zeros_like(coeffs)
    sparse_coeffs[idx] = coeffs[idx]
    
    # --- PASO 2: COMPRESIÓN SIN PÉRDIDA (Sugerencia del Usuario) ---
    
    # Cuantizamos para poder aplicar Huffman (si no, hay infinitos valores)
    # Escalamos los coeficientes a enteros de 8 bits (0-255)
    c_min, c_max = sparse_coeffs.min(), sparse_coeffs.max()
    quant_coeffs = ((sparse_coeffs - c_min) / (c_max - c_min + 1e-8) * 255).long().numpy()
    
    # Calculamos frecuencias de los símbolos
    freqs = collections.Counter(quant_coeffs)
    huff_tree = build_huffman_tree(freqs)
    huff_codes = get_huffman_codes(huff_tree)
    
    # Cálculo de Bits
    # Sin comprimir: N * 32 bits (float) o N * 8 bits (int)
    bits_raw = n * 32
    bits_quant = n * 8
    
    # Con Huffman
    bits_huffman = sum(freqs[char] * len(code) for char, code in huff_codes.items())
    
    print(f"\nResultados (N={n}, Top-K={k}):")
    print(f"  Tamaño Original (Floats):  {bits_raw} bits")
    print(f"  Tamaño Cuantizado (8-bit): {bits_quant} bits")
    print(f"  Tamaño Huffman (Sin Pérdida sobre Cuantización): {bits_huffman} bits")
    
    ratio_huffman = bits_raw / bits_huffman
    print(f"\nRatio de Compresión Total: {ratio_huffman:.2f}x")
    
    # El error es solo por la cuantización y el Top-K, Huffman NO añade error.
    rec_coeffs = torch.from_numpy(quant_coeffs).float() / 255.0 * (c_max - c_min) + c_min
    x_rec = spectral_transform(rec_coeffs) / n
    mse = torch.mean((x - x_rec)**2).item()
    print(f"  MSE de Reconstrucción: {mse:.6e}")

if __name__ == "__main__":
    run_entropy_compression_test()
