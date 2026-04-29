import torch
import torch.nn.functional as F
import math
import os
import matplotlib.pyplot as plt

# --- Sequency-Ordered Walsh Matrix Generation ---
def get_walsh_matrix(N):
    """ Constructs a Sylvester Walsh-Hadamard matrix of size N x N. """
    if N == 1:
        return torch.tensor([[1.0]])
    H_prev = get_walsh_matrix(N // 2)
    top = torch.cat([H_prev, H_prev], dim=1)
    bottom = torch.cat([H_prev, -H_prev], dim=1)
    return torch.cat([top, bottom], dim=0)

def get_walsh_matrix_sequency(N):
    """ Constructs a Sequency-ordered (Walsh) matrix (ordered by frequency). """
    H = get_walsh_matrix(N)
    crossings = []
    for i in range(N):
        row = H[i]
        num_crossings = (row[:-1] * row[1:] < 0).sum().item()
        crossings.append((num_crossings, i))
    
    crossings.sort()
    indices = [idx for _, idx in crossings]
    return H[indices]

def walsh_2d_transform(image, H):
    """ Transform: W = H * I * H^T """
    return torch.matmul(H, torch.matmul(image, H.T))

def iwalsh_2d_transform(coeffs, H):
    """ Inverse: I = (1/N^2) * H^T * W * H """
    N = H.shape[0]
    return torch.matmul(H.T, torch.matmul(coeffs, H)) / (N * N)

def generate_synthetic_image(size=256):
    """ Genera una imagen de prueba 2D (círculos concéntricos y gradiente) """
    y, x = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing='ij')
    radius = torch.sqrt(x**2 + y**2)
    # Patrón suave ideal para ver la compresión
    img = torch.sin(5 * math.pi * radius) * torch.exp(-radius)
    # Normalizar a 0-1
    img = (img - img.min()) / (img.max() - img.min())
    return img

def run_experiment():
    print("=== Experimento V91b: Smooth Walsh con Ordenación por Frecuencia ===\n")
    
    N = 256
    img = generate_synthetic_image(N)
    
    # 1. Generar la matriz de Walsh ordenada por "Sequency" (Frecuencia real)
    print("1. Generando matriz de Walsh ordenada por Sequency...")
    H_256 = get_walsh_matrix_sequency(N)
    
    # 2. Transformada 2D
    print("2. Aplicando Transformada 2D (256x256)...")
    spectrum = walsh_2d_transform(img, H_256)
    
    # 3. Compresión (Filtro Paso Bajo Real)
    K = 32 # Comprimimos masivamente: de 256x256 a 32x32 (64x menos datos)
    print(f"3. Comprimiendo (Reteniendo solo frecuencias bajas {K}x{K})...")
    spectrum_compressed = torch.zeros_like(spectrum)
    # Ahora sí, la esquina superior izquierda contiene genuinamente las frecuencias más bajas
    spectrum_compressed[:K, :K] = spectrum[:K, :K] 
    
    # 4. Reconstrucción Pura (Blocky Walsh)
    print("4. Reconstruyendo imagen original (Blocky)...")
    img_blocky = iwalsh_2d_transform(spectrum_compressed, H_256)
    img_blocky = torch.clamp(img_blocky, 0, 1)
    
    # 5. Smooth Walsh (Filtro Bilineal / Shader)
    print("5. Aplicando Smooth Walsh (IFWHT KxK + Interpolación Bilineal)...")
    mini_spectrum = spectrum[:K, :K]
    H_32 = get_walsh_matrix_sequency(K)
    
    # Reconstruimos la miniatura real de 32x32
    # NOTA MATEMÁTICA: Los coeficientes de mini_spectrum se calcularon sobre N=256. 
    # Al hacer la inversa sobre K=32, los valores resultan (N/K)^2 veces más grandes.
    # Debemos escalar el resultado para mantener el rango de color [0, 1].
    img_mini = iwalsh_2d_transform(mini_spectrum, H_32) / ((N / K) ** 2)
    
    # Interpolamos de 32x32 a 256x256 usando el "Shader" Bilineal
    img_mini_tensor = img_mini.unsqueeze(0).unsqueeze(0)
    img_smooth = F.interpolate(img_mini_tensor, size=(N, N), mode='bilinear', align_corners=False)
    img_smooth = torch.clamp(img_smooth.squeeze(0).squeeze(0), 0, 1)
    
    # --- Métricas de Similitud y Compresión ---
    # Calculamos el Error Cuadrático Medio (MSE) y el PSNR (Peak Signal-to-Noise Ratio)
    mse_blocky = F.mse_loss(img_blocky, img).item()
    mse_smooth = F.mse_loss(img_smooth, img).item()
    
    # PSNR = 10 * log10(MAX^2 / MSE). Como los píxeles van de 0 a 1, MAX=1.
    psnr_blocky = -10 * math.log10(mse_blocky) if mse_blocky > 0 else float('inf')
    psnr_smooth = -10 * math.log10(mse_smooth) if mse_smooth > 0 else float('inf')
    
    # Cálculos de memoria (asumiendo float32 = 4 bytes por valor)
    bytes_original = N * N * 4
    bytes_compressed = K * K * 4
    compression_ratio = bytes_original / bytes_compressed
    
    print(f"\n--- Análisis de Compresión y Memoria ---")
    print(f"Tamaño Imagen Original ({N}x{N}):  {bytes_original / 1024:.1f} KB")
    print(f"Tamaño Coeficientes Walsh ({K}x{K}): {bytes_compressed / 1024:.1f} KB")
    print(f"Ratio de Compresión Logrado:      {compression_ratio:.1f}x")
    
    print(f"\n--- Métricas de Similitud (Fidelidad de Reconstrucción) ---")
    print(f"Walsh Puro (Blocky) -> MSE: {mse_blocky:.4f} | PSNR: {psnr_blocky:.2f} dB")
    print(f"Smooth Walsh (Lerp) -> MSE: {mse_smooth:.4f} | PSNR: {psnr_smooth:.2f} dB")
    
    if mse_smooth < mse_blocky:
        print(f"\n[VICTORIA MATEMÁTICA] El filtro Smooth Walsh reduce el error de reconstrucción en un {((mse_blocky - mse_smooth) / mse_blocky) * 100:.1f}%, manteniendo exactamente los mismos {bytes_compressed / 1024:.1f} KB de coste en memoria.")
    
    # --- Guardar Resultados Visuales ---
    os.makedirs("results/figures", exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(img.numpy(), cmap='gray')
    axs[0].set_title("Original (256x256)")
    axs[0].axis('off')
    
    axs[1].imshow(img_blocky.numpy(), cmap='gray')
    axs[1].set_title(f"Walsh Puro\n(Ceros en altas freq -> Pixel Art)")
    axs[1].axis('off')
    
    axs[2].imshow(img_smooth.numpy(), cmap='gray')
    axs[2].set_title(f"Smooth Walsh\n(Bilinear Lerp de la miniatura)")
    axs[2].axis('off')
    
    save_path = "results/figures/v91_smooth_walsh_sequency.png"
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"\n[ÉXITO] ¡Imagen comparativa corregida guardada en {save_path}!")

if __name__ == '__main__':
    run_experiment()
