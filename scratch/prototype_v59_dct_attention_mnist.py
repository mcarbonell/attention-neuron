import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time
import os
import json

# =============================================================================
# DCT BASIS GENERATION (Differentiable)
# =============================================================================

def get_dct_matrix(N, device='cpu'):
    """Generates the DCT-II basis matrix of size (N, N)."""
    dct_mat = torch.zeros(N, N, device=device)
    for i in range(N):
        for j in range(N):
            if i == 0:
                dct_mat[i, j] = math.sqrt(1/N)
            else:
                dct_mat[i, j] = math.sqrt(2/N) * math.cos((math.pi * i * (2*j + 1)) / (2*N))
    return dct_mat

# =============================================================================
# MODEL: DCT-ATTENTION NEURON
# =============================================================================

class DCTAttentionNet(nn.Module):
    """
    V59: DCT Attention Neuron Network.
    Instead of full 28x28 weights, each neuron learns a small KxK grid
    of DCT coefficients. This forces the model to focus on coherent
    frequency patterns, similar to JPEG compression logic.
    """
    def __init__(self, hidden_dim=256, k_size=8, device='cpu'):
        super().__init__()
        self.N = 28
        self.K = k_size
        self.hidden_dim = hidden_dim
        
        # Precompute DCT matrix
        self.register_buffer('D', get_dct_matrix(self.N, device=device))
        
        # Learnable DCT coefficients for each neuron
        # These are the ONLY 'weights' of the first layer
        # Initialization: Small random values to break symmetry
        self.dct_weights = nn.Parameter(torch.randn(hidden_dim, self.K, self.K) * 0.01)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc_final = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        # x: (B, 1, 28, 28)
        B = x.size(0)
        x = x.view(B, self.N, self.N)
        
        # 1. Transform image to DCT Domain: X_dct = D * X * D^T
        # We do this once for the whole batch
        # x: (B, 28, 28)
        x_dct = torch.matmul(self.D, x)           # (B, 28, 28)
        x_dct = torch.matmul(x_dct, self.D.t())   # (B, 28, 28)
        
        # 2. Extract the low-frequency quadrant (K x K)
        x_low = x_dct[:, :self.K, :self.K]        # (B, K, K)
        
        # 3. Compute neuron activations
        # We want: output_j = sum(x_low * dct_weights_j)
        # Using einstein summation for clarity and speed
        # b: batch, h: hidden_dim, i: k_row, j: k_col
        x_features = torch.einsum('bij,hij->bh', x_low, self.dct_weights)
        x_features = x_features + self.bias
        
        # 4. Standard MLP path
        x = self.bn1(x_features)
        x = F.relu(x)
        return self.fc_final(x)

# =============================================================================
# TRAINING LOOP
# =============================================================================

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V59: DCT ATTENTION NEURONS (MNIST) ---")
    print(f"Device: {device}")

    # Hyperparameters
    HIDDEN_DIM = 512
    K_SIZE = 8       # Each neuron only uses 8x8 = 64 parameters instead of 784!
    BATCH_SIZE = 128
    EPOCHS = 15
    LR = 0.005
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)

    # Model
    model = DCTAttentionNet(hidden_dim=HIDDEN_DIM, k_size=K_SIZE, device=device).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    weight_params = model.dct_weights.numel()
    compression_ratio = (HIDDEN_DIM * 784) / weight_params
    
    print(f"Hidden Dim: {HIDDEN_DIM}")
    print(f"DCT Kernel Size: {K_SIZE}x{K_SIZE}")
    print(f"Trainable Parameters: {total_params:,}")
    print(f"Weight Parameters: {weight_params:,} (vs {HIDDEN_DIM*784:,} dense)")
    print(f"Compression Ratio (Weights): {compression_ratio:.1f}x")

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    metrics = {
        "final_objective": 0,
        "total_evaluations": 0,
        "wall_clock_time": 0,
        "history": []
    }
    
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            metrics["total_evaluations"] += data.size(0)
            
            if epoch == 1 and batch_idx < 5:
                print(f"  > Batch {batch_idx} | Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()

        acc = correct / 10000
        epoch_time = time.time() - t0
        print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Acc: {acc:.4f} | Time: {epoch_time:.1f}s")
        
        metrics["history"].append({"epoch": epoch, "acc": acc, "loss": running_loss/len(train_loader)})

    t_end = time.time()
    metrics["wall_clock_time"] = t_end - t_start
    metrics["final_objective"] = acc
    
    print(f"\n🚀 Final Accuracy: {acc:.4f} | Total Time: {metrics['wall_clock_time']:.1f}s")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/v59_dct_k{K_SIZE}_h{HIDDEN_DIM}.pth")
    
    # Save findings
    os.makedirs("results/raw", exist_ok=True)
    filename = f"v59_dct_k{K_SIZE}_h{HIDDEN_DIM}.json"
    with open(f"results/raw/{filename}", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    train()
