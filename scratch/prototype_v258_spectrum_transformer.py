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
    "hidden_dim": 128,
    "device": "cpu",
    "seed": 42
}

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# --- Fast Walsh-Hadamard Transform (FWHT) ---
def fwht(x):
    """
    Fast Walsh-Hadamard Transform for tensors.
    x: (batch, seq_len, hidden_dim)
    We mix across the seq_len dimension.
    seq_len must be a power of 2.
    """
    b, n, d = x.shape
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x_j = x[:, j, :].clone()
                x_jh = x[:, j + h, :].clone()
                x[:, j, :] = x_j + x_jh
                x[:, j + h, :] = x_j - x_jh
        h *= 2
    return x / math.sqrt(n)

# --- Architecture Components ---
class TernaryLinearGated(nn.Module):
    def __init__(self, in_features, out_features, init_val=1.0):
        super().__init__()
        weights = torch.randint(-1, 2, (out_features, in_features)).float()
        self.register_buffer("weight", weights)
        self.gate = nn.Parameter(torch.full((out_features,), float(init_val)))
        
    def forward(self, x):
        # x can be (batch, seq, in) or (batch, in)
        res = torch.matmul(x, self.weight.t())
        return res * self.gate

class SpectrumGatedTransformer(nn.Module):
    def __init__(self, patch_size, hidden_dim, num_classes=10):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        
        # 1. Patch Embedding (Frozen Ternary)
        # 16 pixels -> hidden_dim
        self.patch_embed = TernaryLinearGated(patch_size * patch_size, hidden_dim, init_val=0.0)
        self.silu1 = nn.SiLU()
        
        # 2. Mixing Layer (Spectral + Gating)
        # After mixing, we apply another gated projection
        self.mixer_gate = TernaryLinearGated(hidden_dim, hidden_dim, init_val=1.0)
        self.silu2 = nn.SiLU()
        
        # 3. Final Head
        self.classifier = TernaryLinearGated(hidden_dim, num_classes, init_val=1.0)
        
    def forward(self, x):
        # x: (batch, 1, 32, 32) after padding
        b, c, h, w = x.shape
        p = self.patch_size
        
        # Divide into patches: (batch, 64, 16)
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(b, -1, p*p)
        
        # Step 1: Patch Embedding
        x = self.patch_embed(x) # (batch, 64, hidden_dim)
        x = self.silu1(x)
        
        # Step 2: Global Spectral Mixing (Cross-patch communication)
        # We mix the patches along the sequence dimension (64)
        x = fwht(x) 
        
        # Step 3: Mixer Projection
        x = self.mixer_gate(x)
        x = self.silu2(x)
        
        # Step 4: Global Average Pooling (across patches)
        x = x.mean(dim=1)
        
        # Step 5: Classifier
        x = self.classifier(x)
        return x

    def print_architecture(self):
        print("\n--- Network Architecture (Spectrum-Gated Transformer) ---")
        total_frozen = sum(p.numel() for p in self.buffers())
        total_learnable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Patches: 64 (4x4 pixels)")
        print(f"Hidden Dim: {self.hidden_dim}")
        print(f"Mixing: Fast Walsh-Hadamard (Global)")
        print(f"Frozen Params:    {total_frozen:,}")
        print(f"Learnable Gates:  {total_learnable:,}")
        print("---------------------------\n")

# --- Training / Eval ---
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            # Pad 28x28 to 32x32
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
    
    start_time = time.time()
    for epoch in range(config["epochs"]):
        model.train()
        epoch_start = time.time()
        for i, (data, target) in enumerate(train_loader):
            # Pad 28x28 to 32x32
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
        
    return time.time() - start_time

def main():
    set_seed(CONFIG["seed"])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(datasets.MNIST('data', train=False, transform=transform), batch_size=CONFIG["batch_size"], shuffle=False)
    
    model = SpectrumGatedTransformer(CONFIG["patch_size"], CONFIG["hidden_dim"]).to(CONFIG["device"])
    duration = train_model(model, train_loader, test_loader, CONFIG)
    
    # Save
    os.makedirs("results/raw", exist_ok=True)
    report = {"config": CONFIG, "duration": duration}
    with open("results/raw/v258_spectrum_transformer.json", "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nResults saved to results/raw/v258_spectrum_transformer.json")

if __name__ == "__main__":
    main()
