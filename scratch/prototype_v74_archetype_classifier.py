import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
import os
import sys

# Import the VectorCanvas from the previous script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prototype_v73_mnist_mean_vectorizer import VectorCanvas

def train_and_classify():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- V74: Archetype Nearest Centroid Classifier ---")
    print(f"Device: {device}")
    
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    print("\n1. Calculating Pixel Archetypes (Means) from Train Set...")
    pixel_archetypes = []
    for digit in range(10):
        imgs = [img.squeeze() for img, label in train_dataset if label == digit]
        pixel_archetypes.append(torch.stack(imgs).mean(dim=0).to(device))
    pixel_archetypes = torch.stack(pixel_archetypes) # (10, 28, 28)
    
    print("\n2. Training Vector Archetypes (15 strokes per digit)...")
    vector_archetypes = []
    for digit in range(10):
        print(f"   Vectorizing '{digit}'...", end='', flush=True)
        model = VectorCanvas(num_strokes=15, device=device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.5)
        target = pixel_archetypes[digit]
        
        for _ in range(200):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(), target)
            loss.backward()
            optimizer.step()
            
        vector_archetypes.append(model().detach())
        print(f" Loss: {loss.item():.4f}")
    vector_archetypes = torch.stack(vector_archetypes) # (10, 28, 28)
    
    print("\n3. Running Classification on Test Set (10,000 images)...")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    correct_pixel = 0
    correct_vector = 0
    total = 0
    
    # Pre-flatten archetypes for fast batch distance calculation
    flat_pixel = pixel_archetypes.view(10, 784)
    flat_vector = vector_archetypes.view(10, 784)
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            batch_size = data.size(0)
            flat_data = data.view(batch_size, 784) # (B, 784)
            
            # Distance to Pixel Archetypes: mean((x - a)^2)
            # Expand data: (B, 1, 784), archetypes: (1, 10, 784) -> (B, 10, 784)
            dist_pixel = torch.mean((flat_data.unsqueeze(1) - flat_pixel.unsqueeze(0))**2, dim=2) # (B, 10)
            pred_pixel = torch.argmin(dist_pixel, dim=1)
            correct_pixel += (pred_pixel == targets).sum().item()
            
            # Distance to Vector Archetypes
            dist_vector = torch.mean((flat_data.unsqueeze(1) - flat_vector.unsqueeze(0))**2, dim=2) # (B, 10)
            pred_vector = torch.argmin(dist_vector, dim=1)
            correct_vector += (pred_vector == targets).sum().item()
            
            total += batch_size
            
    acc_pixel = correct_pixel / total
    acc_vector = correct_vector / total
    
    print(f"\n=======================================")
    print(f"--- FINAL CLASSIFICATION RESULTS ---")
    print(f"=======================================")
    print(f"Template Matching via Pixel Archetypes:  {acc_pixel*100:.2f}% Accuracy")
    print(f"Template Matching via Vector Archetypes: {acc_vector*100:.2f}% Accuracy")
    print(f"=======================================")

if __name__ == "__main__":
    train_and_classify()
