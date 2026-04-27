import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import math
import json
import os

class LineNeuronLayer(nn.Module):
    def __init__(self, num_neurons=256):
        super().__init__()
        self.num_neurons = num_neurons
        self.size = 28
        
        # FIXED INITIALIZATION: All neurons start in the center (v55 style)
        init_points = torch.zeros(num_neurons, 2, 2)
        init_points[:, 0, :] = torch.tensor([14.0, 7.0])
        init_points[:, 1, :] = torch.tensor([14.0, 21.0])
        self.points = nn.Parameter(init_points)
        
        # Trainable sigmas, but we will apply a multiplier on top
        self.log_sigma_pos = nn.Parameter(torch.full((num_neurons,), math.log(1.2)))
        self.log_sigma_neg = nn.Parameter(torch.full((num_neurons,), math.log(2.5)))
        
        y, x = torch.meshgrid(torch.linspace(0, 27, 28), torch.linspace(0, 27, 28), indexing='ij')
        self.register_buffer("grid", torch.stack([x, y], dim=-1).view(1, 784, 2))

    def get_masks(self, sigma_mult=1.0):
        t = torch.linspace(0, 1, 10, device=self.points.device).view(1, 10, 1)
        p0, p1 = self.points[:, 0:1, :], self.points[:, 1:2, :]
        line_points = (1-t) * p0 + t * p1
        
        diff = self.grid.unsqueeze(0) - line_points.unsqueeze(2)
        dist_sq = torch.sum(diff**2, dim=-1)
        min_dist_sq, _ = torch.min(dist_sq, dim=1)
        
        # Apply the scheduler multiplier
        sigma_pos = torch.exp(self.log_sigma_pos).view(-1, 1) * sigma_mult
        sigma_neg = torch.exp(self.log_sigma_neg).view(-1, 1) * sigma_mult
        
        stroke = torch.exp(-min_dist_sq / (2 * sigma_pos**2))
        surround = torch.exp(-min_dist_sq / (2 * sigma_neg**2))
        
        return stroke - 0.5 * surround

    def forward(self, x, sigma_mult=1.0):
        B = x.size(0)
        masks = self.get_masks(sigma_mult=sigma_mult)
        return torch.matmul(x.view(B, 784), masks.t())

class LineNet(nn.Module):
    def __init__(self, num_neurons=256):
        super().__init__()
        self.line_layer = LineNeuronLayer(num_neurons=num_neurons)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_neurons, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x, sigma_mult=1.0):
        h = self.line_layer(x, sigma_mult=sigma_mult)
        return self.classifier(h)

def train():
    device = torch.device('cpu')
    print(f"--- V56: COARSE-TO-FINE SCHEDULER (CENTER INIT) ---")
    
    NUM_NEURONS = 256
    EPOCHS = 15
    LR = 0.002
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=256, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1024)

    model = LineNet(num_neurons=NUM_NEURONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    history = []
    
    # Sigma scheduler parameters
    start_mult = 5.0
    end_mult = 1.0
    warmup_epochs = 5

    for epoch in range(1, EPOCHS + 1):
        model.train()
        # Linear decay of sigma multiplier
        if epoch <= warmup_epochs:
            current_mult = start_mult - (start_mult - end_mult) * ((epoch - 1) / (warmup_epochs - 1))
        else:
            current_mult = end_mult
            
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data, sigma_mult=current_mult)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            with torch.no_grad(): model.line_layer.points.clamp_(0, 27.9)
            epoch_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data, sigma_mult=current_mult).argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        acc = correct / 10000
        history.append(acc)
        print(f"Epoch {epoch:2d} | Mult: {current_mult:.2f} | Loss: {epoch_loss/len(train_loader):.4f} | Acc: {acc:.4f}")

    os.makedirs("results/raw", exist_ok=True)
    metrics = {
        "experiment": "v56_sigma_scheduler",
        "best_objective": max(history),
        "history_acc": history
    }
    with open("results/raw/v56_sigma_scheduler.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    torch.save(model.state_dict(), "v56_scheduler.pth")
    print("✅ Model saved as v56_scheduler.pth")

if __name__ == "__main__":
    train()
