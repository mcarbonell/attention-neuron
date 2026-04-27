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
    def __init__(self, num_neurons=256, device='cpu'):
        super().__init__()
        self.num_neurons = num_neurons
        self.size = 28
        
        # Trainable Line Endpoints: (N, 2, 2) -> [p0, p1]
        self.points = nn.Parameter(torch.rand(num_neurons, 2, 2) * 28.0)
        
        # Trainable Stroke Widths (one per neuron)
        # log_sigma_pos for the excitatory center, log_sigma_neg for inhibitory surround
        self.log_sigma_pos = nn.Parameter(torch.full((num_neurons,), math.log(1.2)))
        self.log_sigma_neg = nn.Parameter(torch.full((num_neurons,), math.log(2.5)))
        
        y, x = torch.meshgrid(torch.linspace(0, 27, 28), torch.linspace(0, 27, 28), indexing='ij')
        self.register_buffer("grid", torch.stack([x, y], dim=-1).view(1, 784, 2))

    def get_masks(self):
        # Sample 10 points along the line segment for distance approximation
        t = torch.linspace(0, 1, 10, device=self.points.device).view(1, 10, 1)
        p0, p1 = self.points[:, 0:1, :], self.points[:, 1:2, :]
        line_points = (1-t) * p0 + t * p1 # Linear interpolation: (N, 10, 2)
        
        # grid: (1, 784, 2), line_points: (N, 10, 2)
        diff = self.grid.unsqueeze(0) - line_points.unsqueeze(2) # (N, 10, 784, 2)
        dist_sq = torch.sum(diff**2, dim=-1) # (N, 10, 784)
        min_dist_sq, _ = torch.min(dist_sq, dim=1) # (N, 784)
        
        sigma_pos = torch.exp(self.log_sigma_pos).view(-1, 1)
        sigma_neg = torch.exp(self.log_sigma_neg).view(-1, 1)
        
        # Difference of Gaussians (DoG) along the line
        stroke = torch.exp(-min_dist_sq / (2 * sigma_pos**2))
        surround = torch.exp(-min_dist_sq / (2 * sigma_neg**2))
        
        return stroke - 0.5 * surround

    def forward(self, x):
        B = x.size(0)
        masks = self.get_masks()
        # x: (B, 1, 28, 28) -> (B, 784)
        # masks: (N, 784)
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

    def forward(self, x):
        h = self.line_layer(x)
        return self.classifier(h)

def train_and_eval():
    # Use CPU for small models as it is often faster due to low overhead
    device = torch.device('cpu') 
    print(f"--- V51: LINE NEURONS (MATCHSTICK) ---")
    
    NUM_NEURONS = 256
    EPOCHS = 10
    LR = 0.0005
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Check if data exists, if not download
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1024)

    model = LineNet(num_neurons=NUM_NEURONS).to(device)
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
            
            # Constraint: Keep points within the image bounds
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
    
    # Ensure results directory exists
    os.makedirs("results/raw", exist_ok=True)
    
    metrics = {
        "experiment": "v51_line_neurons",
        "final_objective": history[-1],
        "total_evaluations": len(train_loader) * EPOCHS,
        "wall_clock_time": total_time,
        "function_evaluation_time": eval_time,
        "internal_overhead_time": total_time - eval_time,
        "num_neurons": NUM_NEURONS,
        "params_per_neuron": 4 + 2, # x1,y1,x2,y2 + 2 sigmas
        "history_acc": history
    }
    
    with open("results/raw/v51_line_neurons.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Save weights
    torch.save(model.state_dict(), "v51_matchsticks.pth")
    print("✅ Model weights saved as v51_matchsticks.pth")
    
    print(f"\n✅ Training Complete.")
    print(f"✅ Final Accuracy: {history[-1]:.4f}")
    print(f"✅ Results saved in results/raw/v51_line_neurons.json")

if __name__ == "__main__":
    train_and_eval()
