import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

def get_haar_filters():
    # LL, HL (Vertical edge), LH (Horizontal edge), HH (Diagonal)
    kernel = torch.tensor([
        [[[1, 1], [1, 1]]],
        [[[1, -1], [1, -1]]],
        [[[1, 1], [-1, -1]]],
        [[[1, -1], [-1, 1]]]
    ], dtype=torch.float32) / 2.0
    return kernel

def haar_decomposition_2d(x, levels=2):
    # x: [1, 1, H, W]
    filters = get_haar_filters()
    current_ll = x
    results = []
    
    for i in range(levels):
        out = F.conv2d(current_ll, filters, stride=2)
        ll = out[:, 0:1]
        hl = out[:, 1:2]
        lh = out[:, 2:3]
        hh = out[:, 3:4]
        results.append((ll, hl, lh, hh))
        current_ll = ll
        
    return results

def visualize():
    # Cargar MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Elegir unos cuantos dígitos diferentes
    digits_to_show = [0, 1, 3, 4, 8]
    indices = []
    for d in digits_to_show:
        for i in range(len(dataset)):
            if dataset[i][1] == d:
                indices.append(i)
                break

    fig, axes = plt.subplots(len(digits_to_show), 5, figsize=(15, 3 * len(digits_to_show)))
    plt.subplots_adjust(hspace=0.4)

    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        img_input = img.unsqueeze(0) # [1, 1, 28, 28]
        # Pad a 32x32 para que sea potencia de 2
        img_padded = F.pad(img_input, (2, 2, 2, 2))
        
        # Obtener descomposición de Haar (Nivel 1 para visualizar bordes claros)
        decomp = haar_decomposition_2d(img_padded, levels=1)
        ll, hl, lh, hh = decomp[0]
        
        # Convertir a numpy para plotear
        orig = img_padded[0,0].numpy()
        v_edges = hl[0,0].abs().numpy() # Valor absoluto para ver energía
        h_edges = lh[0,0].abs().numpy()
        d_edges = hh[0,0].abs().numpy()
        low_res = ll[0,0].numpy()
        
        # Graficar
        axes[i, 0].imshow(orig, cmap='gray')
        axes[i, 0].set_title(f"Original (Dígito {label})")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(low_res, cmap='magma')
        axes[i, 1].set_title("Baja Res (LL)")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(v_edges, cmap='magma')
        axes[i, 2].set_title("Bordes Vert (HL)")
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(h_edges, cmap='magma')
        axes[i, 3].set_title("Bordes Horiz (LH)")
        axes[i, 3].axis('off')
        
        axes[i, 4].imshow(d_edges, cmap='magma')
        axes[i, 4].set_title("Esquinas/Diag (HH)")
        axes[i, 4].axis('off')

    plt.savefig('results/haar_visualization.png', bbox_inches='tight')
    print("Visualización guardada en 'results/haar_visualization.png'")
    # plt.show()

if __name__ == '__main__':
    visualize()
