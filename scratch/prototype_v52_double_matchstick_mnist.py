import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import math
import json
import os

class DoubleLineNeuronLayer(nn.Module):
    def __init__(self, num_neurons=256, device='cpu'):
        super().__init__()
        self.num_neurons = num_neurons
        self.size = 28
        
        # Trainable Line Endpoints: (N, 4, 2) -> [p0, p1, p2, p3]
        # p0-p1 forms Line 1, p2-p3 forms Line 2
        self.points = nn.Parameter(torch.rand(num_neurons, 4, 2) * 28.0)
        
        # Trainable Stroke Widths (one pair per neuron for both lines)
        self.log_sigma_pos = nn.Parameter(torch.full((num_neurons,), math.log(1.2)))
        self.log_sigma_neg = nn.Parameter(torch.full((num_neurons,), math.log(2.5)))
        
        y, x = torch.meshgrid(torch.linspace(0, 27, 28), torch.linspace(0, 27, 28), indexing='ij')
        self.register_buffer("grid", torch.stack([x, y], dim=-1).view(1, 784, 2))

    def get_masks(self):
        # Sample points along both line segments
        t = torch.linspace(0, 1, 8, device=self.points.device).view(1, 8, 1)
        
        # Line 1
        p0, p1 = self.points[:, 0:1, :], self.points[:, 1:2, :]
        line1_pts = (1-t) * p0 + t * p1 # (N, 8, 2)
        
        # Line 2
        p2, p3 = self.points[:, 2:3, :], self.points[:, 3:4, :]
        line2_pts = (1-t) * p2 + t * p3 # (N, 8, 2)
        
        # Combine sampled points: (N, 16, 2)
        all_pts = torch.cat([line1_pts, line2_pts], dim=1)
        
        # Calculate distance to nearest point among all 16 sampled points
        diff = self.grid.unsqueeze(0) - all_pts.unsqueeze(2) # (N, 16, 784, 2)
        dist_sq = torch.sum(diff**2, dim=-1) # (N, 16, 784)
        min_dist_sq, _ = torch.min(dist_sq, dim=1) # (N, 784)
        
        sigma_pos = torch.exp(self.log_sigma_pos).view(-1, 1)
        sigma_neg = torch.exp(self.log_sigma_neg).view(-1, 1)
        
        stroke = torch.exp(-min_dist_sq / (2 * sigma_pos**2))
        surround = torch.exp(-min_dist_sq / (2 * sigma_neg**2))
        
        return stroke - 0.5 * surround

    def forward(self, x):
        B = x.size(0)
        masks = self.get_masks()
        return torch.matmul(x.view(B, 784), masks.t())

class DoubleLineNet(nn.Module):
    def __init__(self, num_neurons=256):
        super().__init__()
        self.line_layer = DoubleLineNeuronLayer(num_neurons=num_neurons)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_neurons, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        h = self.line_layer(x)
        return self.classifier(h)

def train_and_eval():
    device = torch.device('cpu') 
    print(f"--- V52: DOUBLE MATCHSTICK NEURONS ---")
    
    NUM_NEURONS = 256
    EPOCHS = 10
    LR = 0.005
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1024)

    model = DoubleLineNet(num_neurons=NUM_NEURONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    start_wall = time.time()
    eval_time = 0
    history = []

    print(f"Training on {device}...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            fwd_start = time.time()
            output = model(data)
            eval_time += (time.time() - fwd_start)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                model.line_layer.points.clamp_(0, 27.9)
            
            epoch_loss += loss.item()
            if batch_idx == 0:
                print(f"  Epoch {epoch} | Initial Batch Loss: {loss.item():.4f}")

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        acc = correct / 10000
        history.append(acc)
        print(f"Epoch {epoch:2d} | Loss: {epoch_loss/len(train_loader):.4f} | Acc: {acc:.4f} | Time: {time.time()-epoch_start:.2f}s")

    total_time = time.time() - start_wall
    os.makedirs("results/raw", exist_ok=True)
    
    metrics = {
        "experiment": "v52_double_matchstick",
        "final_objective": history[-1],
        "best_objective": max(history),
        "total_evaluations": len(train_loader) * EPOCHS,
        "wall_clock_time": total_time,
        "function_evaluation_time": eval_time,
        "internal_overhead_time": total_time - eval_time,
        "num_neurons": NUM_NEURONS,
        "params_per_neuron": 8 + 2, # 4 points (8 coords) + 2 sigmas
        "history_acc": history
    }
    
    with open("results/raw/v52_double_matchstick.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    torch.save(model.state_dict(), "v52_double_matchsticks.pth")
    print("✅ Model saved as v52_double_matchsticks.pth")
    print(f"✅ Final Accuracy: {history[-1]:.4f} | Best: {max(history):.4f}")

if __name__ == "__main__":
    train_and_eval()
