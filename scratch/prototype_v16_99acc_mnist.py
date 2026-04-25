import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time
import json

class AttentionNeuronLayer(nn.Module):
    """
    V16: Over-Parametrized Attention Neuron Layer.
    W_evolved = W_init * (1 + delta_in_m @ delta_out_m) + (delta_in_a @ delta_out_a)
    W_init is fixed and random.
    """
    def __init__(self, in_features, out_features, rank=32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Fixed base weights (Kaiming Normal)
        std = math.sqrt(2.0 / in_features)
        self.register_buffer('w_init', torch.randn(out_features, in_features) * std)
        
        # Rank-based scaling for stability
        rank_scale = 1.0 / math.sqrt(rank)
        
        # Multiplicative modulation: initialized such that product is small but non-zero
        self.delta_in_m = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        
        # Additive modulation: initialized at zero
        self.delta_in_a = nn.Parameter(torch.zeros(out_features, rank))
        self.delta_out_a = nn.Parameter(torch.zeros(rank, in_features))
        
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Optional: LayerNorm to stabilize activations in deep networks
        self.ln = nn.LayerNorm(out_features)

    def forward(self, x):
        # Multiplicative part: (1 + rank_r_update)
        # Using 1.0 + matmul allows the network to start as the random baseline and "tune" it.
        w_m = torch.matmul(self.delta_in_m, self.delta_out_m)
        w_a = torch.matmul(self.delta_in_a, self.delta_out_a)
        
        w_evolved = self.w_init * (1.0 + w_m) + w_a
        
        y = torch.matmul(x, w_evolved.t()) + self.bias
        return self.ln(y)

class AttentionNeuronNetV16(nn.Module):
    def __init__(self, rank=32):
        super().__init__()
        self.layer1 = AttentionNeuronLayer(784, 1024, rank=rank)
        self.layer2 = AttentionNeuronLayer(1024, 1024, rank=rank)
        self.layer3 = AttentionNeuronLayer(1024, 10, rank=rank)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training V16 (Rank 32 Attention Neuron) on: {device}")
    
    # Hyperparameters
    BATCH_SIZE = 128
    EPOCHS = 30
    RANK = 32
    MAX_LR = 0.005
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    model = AttentionNeuronNetV16(rank=RANK).to(device)
    params = count_parameters(model)
    print(f"Trainable Parameters: {params:,}")
    
    # MLP Equivalent parameters calculation
    # (784*1024 + 1024) + (1024*1024 + 1024) + (1024*10 + 10) = 803,840 + 1,049,600 + 10,250 = 1,863,690
    print(f"MLP Equivalent: ~1.86M parameters (AN uses {params/1863690:.1%})")
    
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR / 10, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, steps_per_epoch=len(train_loader), epochs=EPOCHS
    )
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    t_start = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        acc = correct / 10000
        if acc > best_acc:
            best_acc = acc
            
        print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Acc: {acc:.4f} | Best: {best_acc:.4f}")
        
    t_total = time.time() - t_start
    print(f"\nTraining finished in {t_total:.1f}s")
    print(f"Final Best Test Accuracy: {best_acc:.4f}")
    
    # Save results
    results = {
        "model": "V16_Attention_Neuron",
        "rank": RANK,
        "trainable_params": params,
        "best_acc": best_acc,
        "epochs": EPOCHS,
        "wall_clock_time": t_total,
        "dataset": "MNIST"
    }
    
    with open("results/raw/v16_99target_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
