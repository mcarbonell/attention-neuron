import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import json
import os
import math

# --- Configuration ---
N_BANK = 8  # Number of sinusoidal neurons per projection value
BATCH_SIZE = 256
EPOCHS = 15
LR = 0.002
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Layers ---

class ProjectionLayer(nn.Module):
    """
    Computes row and column sums. Output size: 56.
    """
    def forward(self, x):
        # x: (B, 1, 28, 28)
        row_sums = x.sum(dim=3).squeeze(1) # (B, 28)
        col_sums = x.sum(dim=2).squeeze(1) # (B, 28)
        return torch.cat([row_sums, col_sums], dim=1) # (B, 56)

class SinusoidalModulationLayer(nn.Module):
    """
    Learns frequency and phase for each projection value.
    Input: (B, 56)
    Output: (B, 56 * N_BANK)
    """
    def __init__(self, in_features, n_bank):
        super().__init__()
        self.in_features = in_features
        self.n_bank = n_bank
        
        # Initialize frequencies and phases
        # Frequency scale: spread out to capture different features
        self.omega = nn.Parameter(torch.randn(in_features, n_bank) * 0.5)
        self.phi = nn.Parameter(torch.randn(in_features, n_bank) * math.pi)

    def forward(self, x):
        # x: (B, 56)
        # Expand x to (B, 56, 1)
        x_expanded = x.unsqueeze(2)
        
        # Compute sin(omega * x + phi)
        # Broadcasting: (B, 56, 1) * (56, N_BANK) + (56, N_BANK) -> (B, 56, N_BANK)
        h = torch.sin(self.omega * x_expanded + self.phi)
        
        # Flatten to (B, 56 * N_BANK)
        return h.view(x.size(0), -1)

class SinusoidalProjectionNet(nn.Module):
    def __init__(self, n_bank=8):
        super().__init__()
        self.projection = ProjectionLayer()
        self.modulator = SinusoidalModulationLayer(56, n_bank)
        self.classifier = nn.Linear(56 * n_bank, 10)

    def forward(self, x):
        x = self.projection(x)
        x = self.modulator(x)
        return self.classifier(x)

# --- Main Logic ---

def main():
    print(f"\n>>> Running V121: Projection Sinusoids (N={N_BANK})")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    model = SinusoidalProjectionNet(n_bank=N_BANK).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    params = sum(p.numel() for p in model.parameters())
    print(f"Total Trainable Parameters: {params}")

    metrics = {
        'total_evaluations': 0,
        'wall_clock_time': 0,
        'function_evaluation_time': 0,
        'final_objective': 0,
        'test_acc': 0
    }

    start_wall = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            t0 = time.time()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            eval_time = time.time() - t0
            
            metrics['function_evaluation_time'] += eval_time
            metrics['total_evaluations'] += data.size(0)
            
            if epoch == 1 and batch_idx < 5:
                print(f"  [Fast Feedback] Epoch 1, Batch {batch_idx}: Loss = {loss.item():.4f}")

        model.eval()
        correct = 0
        total_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                total_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        acc = correct / len(test_dataset)
        avg_loss = total_loss / len(test_loader)
        print(f"  Epoch {epoch:2d} | Test Acc: {acc:.4f} | Loss: {avg_loss:.4f}")
        
    metrics['wall_clock_time'] = time.time() - start_wall
    metrics['internal_overhead_time'] = metrics['wall_clock_time'] - metrics['function_evaluation_time']
    metrics['final_objective'] = avg_loss
    metrics['test_acc'] = acc
    
    # Save results
    os.makedirs('results/raw', exist_ok=True)
    save_path = f'results/raw/v121_projection_sinusoids_n{N_BANK}.json'
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nFinal Test Accuracy: {acc*100:.2f}%")
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    main()
