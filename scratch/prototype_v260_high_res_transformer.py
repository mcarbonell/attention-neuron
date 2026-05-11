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
    "epochs": 15,
    "max_lr": 1e-2,
    "patch_size": 2, # High resolution (2x2)
    "hidden_dim": 256,
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

def get_sinusoidal_embeddings(n_seq, d_model):
    """Fixed Sin/Cos Positional Encodings."""
    pe = torch.zeros(n_seq, d_model)
    position = torch.arange(0, n_seq, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

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
    def __init__(self, hidden_dim, seq_len, hadamard_matrix):
        super().__init__()
        self.register_buffer("hadamard", hadamard_matrix)
        self.proj = TernaryLinearGated(hidden_dim, hidden_dim, init_val=0.0)
        self.silu = nn.SiLU()
        
    def forward(self, x):
        residual = x
        x = self.hadamard @ x
        x = self.proj(x)
        x = self.silu(x)
        return residual + x

class HighResPSGT(nn.Module):
    def __init__(self, patch_size, hidden_dim, num_classes=10):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        
        # 0. Constants (2x2 patches in 32x32 image -> 256 patches)
        n_patches = (32 // patch_size) ** 2 # 256
        self.register_buffer("hadamard", get_hadamard_matrix(n_patches))
        self.register_buffer("pos_encoding", get_sinusoidal_embeddings(n_patches, hidden_dim))
        
        # 1. Patch Embedding (4 pixels -> hidden_dim)
        self.patch_embed = TernaryLinearGated(patch_size * patch_size, hidden_dim, init_val=1.0)
        self.silu_embed = nn.SiLU()
        
        # 2. Blocks (Deepened to 4 blocks)
        self.block1 = ResidualSpectrumBlock(hidden_dim, n_patches, self.hadamard)
        self.block2 = ResidualSpectrumBlock(hidden_dim, n_patches, self.hadamard)
        self.block3 = ResidualSpectrumBlock(hidden_dim, n_patches, self.hadamard)
        self.block4 = ResidualSpectrumBlock(hidden_dim, n_patches, self.hadamard)
        
        # 3. Head
        self.classifier = TernaryLinearGated(hidden_dim, num_classes, init_val=1.0)
        
    def forward(self, x):
        b, c, h, w = x.shape
        p = self.patch_size
        
        # Patching: (batch, 256, 4)
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(b, -1, p*p)
        
        # Step 1: Embed + Positional
        x = self.patch_embed(x)
        x = x + self.pos_encoding
        x = self.silu_embed(x)
        
        # Step 2: Blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Step 3: GAP
        x = x.mean(dim=1)
        
        # Step 4: Classifier
        return self.classifier(x)

    def print_architecture(self):
        print("\n--- Network Architecture (High-Res PSGT v2) ---")
        total_frozen = sum(p.numel() for p in self.buffers())
        total_learnable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Patches: 256 (2x2 pixels)")
        print(f"Blocks: 4 Residual Blocks")
        print(f"Learnable Gates: {total_learnable:,}")
        print(f"Frozen Params:   {total_frozen:,}")
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
    optimizer = optim.Adam(model.parameters(), lr=config["max_lr"]/10)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config["max_lr"], 
        steps_per_epoch=len(train_loader), 
        epochs=config["epochs"]
    )
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
            scheduler.step()
            
            if i < 5 and epoch == 0:
                print(f"  Batch {i}: Loss {loss.item():.4f}")
        
        acc = evaluate(model, test_loader, config["device"])
        print(f"Epoch {epoch+1}/{config['epochs']} | Acc: {acc:.2f}% | Time: {time.time()-epoch_start:.2f}s")

def main():
    set_seed(CONFIG["seed"])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(datasets.MNIST('data', train=False, transform=transform), batch_size=CONFIG["batch_size"], shuffle=False)
    
    model = HighResPSGT(CONFIG["patch_size"], CONFIG["hidden_dim"]).to(CONFIG["device"])
    train_model(model, train_loader, test_loader, CONFIG)

if __name__ == "__main__":
    main()
