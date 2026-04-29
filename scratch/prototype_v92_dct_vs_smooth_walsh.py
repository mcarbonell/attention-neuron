import torch
import torch.nn.functional as F
import math
import os
import time
import matplotlib.pyplot as plt

# --- Sequency-Ordered Walsh Matrix Generation ---
def get_walsh_matrix(N):
    if N == 1: return torch.tensor([[1.0]])
    H_prev = get_walsh_matrix(N // 2)
    top = torch.cat([H_prev, H_prev], dim=1)
    bottom = torch.cat([H_prev, -H_prev], dim=1)
    return torch.cat([top, bottom], dim=0)

def get_walsh_matrix_sequency(N):
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
    return torch.matmul(H, torch.matmul(image, H.T))

def iwalsh_2d_transform(coeffs, H):
    N = H.shape[0]
    return torch.matmul(H.T, torch.matmul(coeffs, H)) / (N * N)

# --- DCT Matrix Generation ---
def get_dct_matrix(N):
    """ Genera la matriz de la Transformada Discreta del Coseno (DCT-II) """
    D = torch.zeros(N, N)
    for k in range(N):
        for n in range(N):
            if k == 0:
                D[k, n] = 1.0 / math.sqrt(N)
            else:
                D[k, n] = math.sqrt(2.0 / N) * math.cos(math.pi * k * (2 * n + 1) / (2 * N))
    return D

def dct_2d_transform(image, D):
    return torch.matmul(D, torch.matmul(image, D.T))

def idct_2d_transform(coeffs, D):
    # La matriz DCT es ortogonal (D * D^T = I), por lo que la inversa es la transpuesta
    return torch.matmul(D.T, torch.matmul(coeffs, D))

def generate_synthetic_image(size=256):
    y, x = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing='ij')
    radius = torch.sqrt(x**2 + y**2)
    # Patrón geométrico suave
    img = torch.sin(5 * math.pi * radius) * torch.exp(-radius)
    img = (img - img.min()) / (img.max() - img.min())
    return img

def run_experiment():
    print("=== Experimento V92: Smooth Walsh vs DCT Clásica (JPEG) ===\n")
    N = 256
    K = 32
    img = generate_synthetic_image(N)
    
    print("Generando bases matemáticas...")
    H_256 = get_walsh_matrix_sequency(N)
    H_32 = get_walsh_matrix_sequency(K)
    D_256 = get_dct_matrix(N)
    
    print(f"\nComprimiendo imagen de {N}x{N} a {K}x{K} (Ratio 64x, Memoria: 4 KB para ambos)...\n")
    
    # --- PROCESO DCT (El estándar de JPEG) ---
    t0_dct = time.perf_counter()
    spectrum_dct = dct_2d_transform(img, D_256)
    
    # Comprimir (Quedarnos con la esquina KxK de bajas frecuencias)
    compressed_dct = torch.zeros_like(spectrum_dct)
    compressed_dct[:K, :K] = spectrum_dct[:K, :K]
    
    # Reconstruir DCT a resolución completa
    img_dct = idct_2d_transform(compressed_dct, D_256)
    img_dct = torch.clamp(img_dct, 0, 1)
    t_dct = time.perf_counter() - t0_dct
    
    # --- PROCESO SMOOTH WALSH (Nuestro algoritmo) ---
    t0_walsh = time.perf_counter()
    spectrum_walsh = walsh_2d_transform(img, H_256)
    
    # Comprimir
    mini_spectrum_walsh = spectrum_walsh[:K, :K]
    
    # Inversa en baja resolución (Ahorro computacional gigante)
    img_mini_walsh = iwalsh_2d_transform(mini_spectrum_walsh, H_32) / ((N / K) ** 2)
    
    # Interpolación Bilineal gratuita por hardware (Upscaling)
    img_mini_tensor = img_mini_walsh.unsqueeze(0).unsqueeze(0)
    img_smooth_walsh = F.interpolate(img_mini_tensor, size=(N, N), mode='bilinear', align_corners=False)
    img_smooth_walsh = torch.clamp(img_smooth_walsh.squeeze(0).squeeze(0), 0, 1)
    t_walsh = time.perf_counter() - t0_walsh
    
    # --- METRICAS ---
    mse_dct = F.mse_loss(img_dct, img).item()
    psnr_dct = -10 * math.log10(mse_dct) if mse_dct > 0 else float('inf')
    
    mse_walsh = F.mse_loss(img_smooth_walsh, img).item()
    psnr_walsh = -10 * math.log10(mse_walsh) if mse_walsh > 0 else float('inf')
    
    print("--- Resultados de Calidad (Fidelidad de Reconstrucción) ---")
    print(f"DCT Clásica  -> MSE: {mse_dct:.5f} | PSNR: {psnr_dct:.2f} dB")
    print(f"Smooth Walsh -> MSE: {mse_walsh:.5f} | PSNR: {psnr_walsh:.2f} dB")
    
    loss_of_quality = ((mse_walsh - mse_dct) / mse_dct) * 100
    print(f"-> Penalización de calidad del Smooth Walsh respecto al DCT: {loss_of_quality:.1f}% más de error.")
    
    print("\n--- Resultados de Velocidad (Multiplicación Matricial) ---")
    print(f"Tiempo DCT (Matemática de coma flotante trigonométrica): {t_dct*1000:.2f} ms")
    print(f"Tiempo Walsh (Matemática binaria + Upscaling):           {t_walsh*1000:.2f} ms")
    speedup = t_dct / t_walsh
    print(f"-> Aceleración: Smooth Walsh es {speedup:.1f}x más rápido.")
    
    # --- VISUALIZACION ---
    os.makedirs("results/figures", exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(img.numpy(), cmap='gray')
    axs[0].set_title("Original (256x256)")
    axs[0].axis('off')
    
    axs[1].imshow(img_dct.numpy(), cmap='gray')
    axs[1].set_title(f"DCT (Estilo JPEG)\nPSNR: {psnr_dct:.1f} dB")
    axs[1].axis('off')
    
    axs[2].imshow(img_smooth_walsh.numpy(), cmap='gray')
    axs[2].set_title(f"Smooth Walsh\nPSNR: {psnr_walsh:.1f} dB")
    axs[2].axis('off')
    
    save_path = "results/figures/v92_dct_vs_smooth_walsh.png"
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"\n[INFO] Gráfica comparativa guardada en {save_path}")
    print("Abre la imagen para comprobar si la pérdida matemática es perceptible por el ojo humano.")

if __name__ == '__main__':
    run_experiment()
