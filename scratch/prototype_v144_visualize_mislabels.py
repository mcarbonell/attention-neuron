import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import os

def visualize_mislabels():
    print("Preparando visualización de anomalías espectrales...")
    
    # Cargar dataset original (sin normalizar para ver los píxeles puros)
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    
    # Índices detectados como CRITICAL en V143
    # Formato: (Indice, Etiqueta Consenso Espectral)
    suspicious_cases = [
        (59915, 7), # El famoso 4 -> 7
        (32835, 1), # 7 -> 1 (Disfrazado)
        (8190, 6),  # 5 -> 6 (Muy curvo)
        (3432, 1),  # 8 -> 1 (Parece un palo)
        (47317, 1), # 8 -> 1
        (45143, 1), # 7 -> 1
        (8480, 1),  # 2 -> 1
        (42624, 9), # 8 -> 9
        (36446, 0), # 6 -> 0
        (46726, 8)  # 0 -> 8
    ]
    
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    fig.suptitle("V144: Anomalías Detectadas por la Memoria Holográfica (CSI Espectral)", fontsize=18, fontweight='bold')
    
    for i, (idx, spec_label) in enumerate(suspicious_cases):
        img, official_label = train_dataset[idx]
        ax = axes[i // 5, i % 5]
        
        # Usamos magma para que se vea bien el contraste de los trazos
        ax.imshow(img.squeeze(), cmap='magma')
        
        title = f"Idx: {idx}\nOfficial: {official_label}\nSpectral: {spec_label}"
        ax.set_title(title, 
                     color='yellow',
                     bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'),
                     fontsize=11)
        ax.axis('off')
        
        # Resaltar si el cambio es muy drástico
        if official_label != spec_label:
            ax.add_patch(plt.Rectangle((0,0), 27, 27, fill=False, edgecolor='red', linewidth=3))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs("results/figures", exist_ok=True)
    save_path = "results/figures/v144_mislabels.png"
    plt.savefig(save_path)
    print(f"\n[ÉXITO] Visualización generada.")
    print(f"Ruta: {save_path}")
    print("\nAnaliza las imágenes con borde rojo: son los casos donde el dataset dice una cosa y la 'física' de Walsh dice otra.")

if __name__ == "__main__":
    visualize_mislabels()
