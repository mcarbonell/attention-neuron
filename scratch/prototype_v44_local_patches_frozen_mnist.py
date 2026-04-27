import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import json
import os
import math
import random

class LocalPatchFrozenMLP(nn.Module):
    """
    V44: Local Patch Frozen Projection.
    Each of the 2048 neurons in the first layer only 'sees' a small 
    randomly placed local patch of the input image.
    """
    def __init__(self, input_size=784, hidden_size=2048, output_size=10, device='cpu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.frozen_layer = nn.Linear(input_size, hidden_size)
        
        print(f"Generating {hidden_size} Local Patch masks...")
        # Initialize weights to zero
        weights = torch.zeros(hidden_size, input_size)
        
        for i in range(hidden_size):
            # 1. Choose a random patch size between 6x6 and 14x14
            p_h = random.randint(6, 14)
            p_w = random.randint(6, 14)
            
            # 2. Choose a random top-left corner
            top = random.randint(0, 28 - p_h)
            left = random.randint(0, 28 - p_w)
            
            # 3. Create a mask for this patch
            mask = torch.zeros(28, 28)
            mask[top:top+p_h, left:left+p_w] = 1.0
            
            # 4. Generate random weights for this neuron
            # We use a normal distribution
            neuron_weights = torch.randn(784) * 0.1
            
            # 5. Apply mask (only the patch is kept)
            weights[i] = neuron_weights * mask.view(-1)
            
        # Normalize weights for healthy variance
        # Since each neuron has few active inputs, we adjust the scale
        # Active inputs per neuron varies, but let's use an average
        target_std = math.sqrt(2.0 / (10*10)) # approx 100 active inputs
        current_std = weights[weights != 0].std()
        weights = weights * (target_std / (current_std + 1e-8))
        
        self.frozen_layer.weight.data = weights.to(device)
        nn.init.zeros_(self.frozen_layer.bias)
        
        # Freeze
        self.frozen_layer.weight.requires_grad = False
        self.frozen_layer.bias.requires_grad = False
        
        # Trainable readout
        self.trainable_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        with torch.no_grad():
            x = self.frozen_layer(x)
            x = torch.relu(x)
        x = self.trainable_layer(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V44: LOCAL PATCH FROZEN PROJECTION ---")
    print(f"Device: {device}")

    # Hyperparameters
    HIDDEN_SIZE = 2048
    BATCH_SIZE = 256
    EPOCHS = 20
    LR = 0.001
    SEED = 42
    
    torch.manual_seed(SEED)
    random.seed(SEED)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1024)

    # Model
    model = LocalPatchFrozenMLP(hidden_size=HIDDEN_SIZE, device=device).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    metrics = {"history": []}
    t_start = time.time()

    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if epoch == 1 and batch_idx < 5:
                print(f"  > Batch {batch_idx} | Loss: {loss.item():.4f}")

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        acc = correct / 10000
        print(f"Epoch {epoch:2d} | Loss: {epoch_loss/len(train_loader):.4f} | Acc: {acc:.4f} | Time: {time.time()-t0:.1f}s")
        metrics["history"].append({"epoch": epoch, "acc": acc})

    t_end = time.time()
    print(f"\n🚀 Final Accuracy: {acc:.4f} | Total Time: {t_end - t_start:.1f}s")
    
    # Save results
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v44_local_patch_frozen.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
