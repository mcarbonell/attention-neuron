import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time
import json
import os

class RosettaLayer(nn.Module):
    """
    V22: The Rosetta Layer (Multi-Substrate Library)
    A neuron can mix K different random substrates using a softmax attention.
    """
    def __init__(self, in_features, out_features, rank=32, num_substrates=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.num_substrates = num_substrates
        
        # K Fixed random substrates
        std = math.sqrt(2.0 / in_features)
        for k in range(num_substrates):
            self.register_buffer(f'w_init_{k}', torch.randn(out_features, in_features) * std)
        
        # Shared modulation parameters for this layer
        self.delta_in_m = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.delta_in_a = nn.Parameter(torch.zeros(out_features, rank))
        self.delta_out_a = nn.Parameter(torch.zeros(rank, in_features))
        
        # Library Attention: softmax logits for mixing the K substrates per neuron
        self.library_logits = nn.Parameter(torch.zeros(out_features, num_substrates))
        
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        # 1. Calculate mixing weights per neuron
        # Shape: (out_features, num_substrates, 1)
        mix_weights = torch.softmax(self.library_logits, dim=1).unsqueeze(2)
        
        # 2. Compute the mixed substrate
        # This is the "Alchemist" logic scaled to K substrates
        w_mixed = 0
        for k in range(self.num_substrates):
            w_init_k = getattr(self, f'w_init_{k}')
            w_mixed += mix_weights[:, k, :] * w_init_k
            
        # 3. Apply low-rank modulation
        w_m = torch.matmul(self.delta_in_m, self.delta_out_m)
        w_a = torch.matmul(self.delta_in_a, self.delta_out_a)
        
        w_evolved = w_mixed * (1.0 + w_m) + w_a
        
        y = torch.matmul(x, w_evolved.t()) + self.bias
        return self.bn(y)

class RosettaStoneNetV22(nn.Module):
    def __init__(self, rank=32, num_substrates=4):
        super().__init__()
        # Flattened CIFAR-10: 3072
        self.layer1 = RosettaLayer(3072, 2048, rank=rank, num_substrates=num_substrates)
        self.layer2 = RosettaLayer(2048, 1024, rank=rank, num_substrates=num_substrates)
        self.layer3 = RosettaLayer(1024, 10, rank=rank, num_substrates=num_substrates)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 3072)
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training V22 'THE ROSETTA STONE' (MLP for CIFAR-10) on: {device}")
    
    BATCH_SIZE = 128
    EPOCHS = 50
    RANK = 32
    NUM_SUBSTRATES = 4
    MAX_LR = 0.003
    
    # We use some data augmentation even for MLP to help it generalize
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=2)
    
    model = RosettaStoneNetV22(rank=RANK, num_substrates=NUM_SUBSTRATES).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR / 10, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, total_steps=len(train_loader)*EPOCHS, 
        pct_start=0.2, anneal_strategy='cos'
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
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

    # Library Usage Analysis
    print("\n--- Rosetta Stone Analysis: Library Usage ---")
    for i, layer in enumerate([model.layer1, model.layer2, model.layer3]):
        # Library weights shape: (out_features, num_substrates)
        weights = torch.softmax(layer.library_logits, dim=1).detach().cpu()
        mean_usage = weights.mean(dim=0).numpy()
        print(f"Layer {i+1} Mean Substrate Usage: {mean_usage}")

    t_total = time.time() - t_start
    print(f"\nROSETTA STONE finished in {t_total:.1f}s")
    print(f"Final Best Test Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
