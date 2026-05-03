
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
import os

def visualize_mnist_delta():
    # 1. Cargar una imagen de MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    try:
        dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    except:
        # Si falla por red, intentamos usar una imagen aleatoria si ya existe el directorio
        print("No se pudo descargar MNIST, asegúrate de tener conexión o el dataset en ./data")
        return

    # Coger un ejemplo aleatorio
    idx = torch.randint(0, len(dataset), (1,)).item()
    img_tensor, label = dataset[idx]
    img_np = img_tensor.squeeze().numpy() # 28x28
    
    # 2. Calcular la diferencia vertical (fila_i - fila_i-1)
    # Rellenamos la primera fila con la diferencia respecto a "negro" (0)
    diff_np = np.zeros_like(img_np)
    diff_np[0] = img_np[0]
    for i in range(1, 28):
        diff_np[i] = img_np[i] - img_np[i-1]
        
    # 3. Crear imagen comparativa
    # Queremos 3 paneles: Original, Diferencia (Gris), Diferencia (Color)
    # Tamaño: (28*3 + padding) x 28
    padding = 10
    canvas_w = 28 * 3 + padding * 2
    canvas_h = 28 + padding * 2
    
    combined = Image.new('RGB', (canvas_w * 4, canvas_h * 4), (30, 30, 30)) # Upscale 4x para ver mejor
    draw = ImageDraw.Draw(combined)
    
    def to_pixel(val): return int(val * 255)

    # Dibujar Original
    orig_img = Image.fromarray((img_np * 255).astype(np.uint8)).resize((28*4, 28*4), Image.NEAREST)
    combined.paste(orig_img, (padding, padding))
    
    # Dibujar Diferencial (Color)
    # Verde = Positivo (Entra en el trazo), Rojo = Negativo (Sale del trazo)
    diff_color = np.zeros((28, 28, 3), dtype=np.uint8)
    for r in range(28):
        for c in range(28):
            val = diff_np[r, c]
            if val > 0:
                diff_color[r, c] = [0, int(abs(val)*255), 0] # Verde
            elif val < 0:
                diff_color[r, c] = [int(abs(val)*255), 0, 0] # Rojo
    
    diff_img = Image.fromarray(diff_color).resize((28*4, 28*4), Image.NEAREST)
    combined.paste(diff_img, (28*4 + padding*2, padding))
    
    # Guardar
    output_path = "mnist_delta_comparison.png"
    combined.save(output_path)
    print(f"Visualización guardada en: {os.path.abspath(output_path)}")
    print(f"Etiqueta del número: {label}")

if __name__ == "__main__":
    visualize_mnist_delta()
