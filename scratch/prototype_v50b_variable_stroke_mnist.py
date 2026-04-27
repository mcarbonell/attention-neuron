import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import math
import random
import os
import json

class StrokeNeuronLayer(nn.Module):
    def __init__(self, num_neurons=256, device='cpu'):
        super().__init__()
        self.num_neurons = num_neurons
        self.size = 28
        
        # Trainable Bezier Points: (N, 3, 2)
        self.points = nn.Parameter(torch.rand(num_neurons, 3, 2) * 28.0)
        
        # Trainable Stroke Widths (one per neuron)
        # Using log space to ensure sigma > 0
        self.log_sigma_pos = nn.Parameter(torch.full((num_neurons,), math.log(1.5)))
        self.log_sigma_neg = nn.Parameter(torch.full((num_neurons,), math.log(3.0)))
        
        y, x = torch.meshgrid(torch.linspace(0, 27, 28), torch.linspace(0, 27, 28), indexing='ij')
        self.register_buffer("grid", torch.stack([x, y], dim=-1).view(1, 784, 2))

    def get_masks(self):
        t = torch.linspace(0, 1, 12, device=self.points.device).view(1, 12, 1)
        p0, p1, p2 = self.points[:, 0:1, :], self.points[:, 1:2, :], self.points[:, 2:3, :]
        bezier_points = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2 # (N, 12, 2)
        
        diff = self.grid.unsqueeze(0) - bezier_points.unsqueeze(2) # (N, 12, 784, 2)
        dist_sq = torch.sum(diff**2, dim=-1) # (N, 12, 784)
        min_dist_sq, _ = torch.min(dist_sq, dim=1) # (N, 784)
        
        # Sigmas (N, 1)
        sigma_pos = torch.exp(self.log_sigma_pos).view(-1, 1)
        sigma_neg = torch.exp(self.log_sigma_neg).view(-1, 1)
        
        stroke = torch.exp(-min_dist_sq / (2 * sigma_pos**2))
        surround = torch.exp(-min_dist_sq / (2 * sigma_neg**2))
        
        return stroke - 0.6 * surround

    def forward(self, x):
        B = x.size(0)
        masks = self.get_masks()
        return torch.matmul(x.view(B, 784), masks.t())

class StrokeNet(nn.Module):
    def __init__(self, num_neurons=256):
        super().__init__()
        self.stroke_layer = StrokeNeuronLayer(num_neurons=num_neurons)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_neurons, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        h = self.stroke_layer(x)
        return self.classifier(h)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V50b: VARIABLE STROKE NEURONS ---")
    
    NUM_NEURONS = 256
    EPOCHS = 15
    LR = 0.005
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=256, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1024)

    model = StrokeNet(num_neurons=NUM_NEURONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            with torch.no_grad(): model.stroke_layer.points.clamp_(0, 28)

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        print(f"Epoch {epoch:2d} | Acc: {correct/10000:.4f}")

    # Save weights
    torch.save(model.state_dict(), "v50b_strokes.pth")
    print("✅ Model saved as v50b_strokes.pth")

if __name__ == "__main__":
    main()
