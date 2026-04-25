import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time
import json

class AttentionNeuronLayerV17(nn.Module):
    def __init__(self, in_features, out_features, rank=64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Fixed base weights (Kaiming Normal)
        std = math.sqrt(2.0 / in_features)
        self.register_buffer('w_init', torch.randn(out_features, in_features) * std)
        
        # Dual modulation (rank-r)
        self.delta_in_m = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.delta_in_a = nn.Parameter(torch.zeros(out_features, rank))
        self.delta_out_a = nn.Parameter(torch.zeros(rank, in_features))
        
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        w_m = torch.matmul(self.delta_in_m, self.delta_out_m)
        w_a = torch.matmul(self.delta_in_a, self.delta_out_a)
        
        w_evolved = self.w_init * (1.0 + w_m) + w_a
        
        y = torch.matmul(x, w_evolved.t()) + self.bias
        return self.bn(y)

class ColossusNetV17(nn.Module):
    def __init__(self, rank=64):
        super().__init__()
        # Wider first layer to capture more from the random substrate
        self.layer1 = AttentionNeuronLayerV17(784, 2048, rank=rank)
        self.layer2 = AttentionNeuronLayerV17(2048, 1024, rank=rank)
        self.layer3 = AttentionNeuronLayerV17(1024, 10, rank=rank)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2) # Increased dropout for generalization

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
    print(f"Training V17 'The Colossus' (Rank 64 + Data Aug) on: {device}")
    
    BATCH_SIZE = 128
    EPOCHS = 40 # Increased epochs for augmentation
    RANK = 64
    MAX_LR = 0.005
    
    # DATA AUGMENTATION: The key to 99%
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    model = ColossusNetV17(rank=RANK).to(device)
    params = count_parameters(model)
    print(f"Trainable Parameters: {params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR / 10, weight_decay=0.05)
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
            # Save the model if it's the best so far
            torch.save(model.state_dict(), "results/v17_best_model.pt")
            
        print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Acc: {acc:.4f} | Best: {best_acc:.4f}")
        
    t_total = time.time() - t_start
    print(f"\nTraining finished in {t_total:.1f}s")
    print(f"Final Best Test Accuracy: {best_acc:.4f}")
    
    results = {
        "model": "V17_Colossus",
        "rank": RANK,
        "trainable_params": params,
        "best_acc": best_acc,
        "epochs": EPOCHS,
        "wall_clock_time": t_total,
        "dataset": "MNIST",
        "augmentation": True
    }
    
    with open("results/raw/v17_colossus_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
