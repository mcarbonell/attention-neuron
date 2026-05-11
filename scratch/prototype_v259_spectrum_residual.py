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
CONFIG = {
    "batch_size": 128,
    "epochs": 10,
    "gate_lr": 1e-3,
    "patch_size": 4,
    "hidden_dim": 1024, # Increased muscle
    "device": "cpu",
    "seed": 42
}

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_hadamard_matrix(n):
    if n == 1:
        return torch.tensor([[1.0]])
    h2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    h_prev = get_hadamard_matrix(n // 2)
    return torch.kron(h2, h_prev) / math.sqrt(2)

# --- Architecture Components ---
class TernaryLinearGated(nn.Module):
    def __init__(self, in_features, out_features, init_val=1.0):
        super().__init__()
        weights = torch.randint(-1, 2, (out_features, in_features)).float()
        self.register_buffer("weight", weights)
        self.gate = nn.Parameter(torch.full((out_features,), float(init_val)))
        
    def forward(self, x):
        res = torch.matmul(x, self.weight.t())
        return res * self.gate

class ResidualSpectrumBlock(nn.Module):
    def __init__(self, hidden_dim, hadamard_matrix):
        super().__init__()
        self.register_buffer("hadamard", hadamard_matrix)
        # We use a gated projection within the block
        self.proj = TernaryLinearGated(hidden_dim, hidden_dim, init_val=0.0) # Start closed
        self.silu = nn.SiLU()
        
    def forward(self, x):
        # x is (B, 64, D)
        residual = x
        
        # 1. Global Spectral Mixing
        x = self.hadamard @ x
        
        # 2. Gated Projection
        x = self.proj(x)
        x = self.silu(x)
        
        # 3. Residual connection
        return residual + x

class RSGTransformer(nn.Module):
    def __init__(self, patch_size, hidden_dim, num_classes=10):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        
        # 0. Precompute Hadamard
        h_mat = get_hadamard_matrix(64)
        
        # 1. Patch Embedding (Frozen Ternary)
        self.patch_embed = TernaryLinearGated(patch_size * patch_size, hidden_dim, init_val=1.0)
        self.silu_embed = nn.SiLU()
        
        # 2. Residual Blocks
        self.block1 = ResidualSpectrumBlock(hidden_dim, h_mat)
        self.block2 = ResidualSpectrumBlock(hidden_dim, h_mat)
        
        # 3. Final Head
        self.classifier = TernaryLinearGated(hidden_dim, num_classes, init_val=1.0)
        
    def forward(self, x):
        b, c, h, w = x.shape
        p = self.patch_size
        
        # Divide into patches
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(b, -1, p*p)
        
        # Step 1: Patch Embedding
        x = self.patch_embed(x) 
        x = self.silu_embed(x)
        
        # Step 2: Residual Mixing Blocks
        x = self.block1(x)
        x = self.block2(x)
        
        # Step 3: Global Average Pooling
        x = x.mean(dim=1)
        
        # Step 4: Classifier
        x = self.classifier(x)
        return x

    def print_architecture(self):
        print("\n--- Network Architecture (RESIDUAL Spectrum-Gated Transformer) ---")
        total_frozen = sum(p.numel() for p in self.buffers())
        total_learnable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Hidden Dim: {self.hidden_dim}")
        print(f"Blocks: 2 Residual Blocks")
        print(f"Frozen Params:    {total_frozen:,}")
        print(f"Learnable Gates:  {total_learnable:,}")
        print("---------------------------\n")

# --- Training / Eval ---
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data = torch.nn.functional.pad(data, (2, 2, 2, 2))
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(loader.dataset)

def train_model(model, train_loader, test_loader, config):
    model.print_architecture()
    optimizer = optim.Adam(model.parameters(), lr=config["gate_lr"])
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(config["epochs"]):
        model.train()
        epoch_start = time.time()
        for i, (data, target) in enumerate(train_loader):
            data = torch.nn.functional.pad(data, (2, 2, 2, 2))
            data, target = data.to(config["device"]), target.to(config["device"])
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if i < 5 and epoch == 0:
                print(f"  Batch {i}: Loss {loss.item():.4f}")
        
        acc = evaluate(model, test_loader, config["device"])
        print(f"Epoch {epoch+1}/{config['epochs']} | Acc: {acc:.2f}% | Time: {time.time()-epoch_start:.2f}s")
        
    return 

def main():
    set_seed(CONFIG["seed"])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(datasets.MNIST('data', train=False, transform=transform), batch_size=CONFIG["batch_size"], shuffle=False)
    
    model = RSGTransformer(CONFIG["patch_size"], CONFIG["hidden_dim"]).to(CONFIG["device"])
    train_model(model, train_loader, test_loader, CONFIG)

if __name__ == "__main__":
    main()
