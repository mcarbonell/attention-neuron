import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import os

class DoubleLineNeuronLayer(nn.Module):
    def __init__(self, num_neurons=256):
        super().__init__()
        self.num_neurons = num_neurons
        self.points = nn.Parameter(torch.rand(num_neurons, 4, 2) * 28.0)
        self.log_sigma_pos = nn.Parameter(torch.full((num_neurons,), math.log(1.2)))
        self.log_sigma_neg = nn.Parameter(torch.full((num_neurons,), math.log(2.5)))
        y, x = torch.meshgrid(torch.linspace(0, 27, 28), torch.linspace(0, 27, 28), indexing='ij')
        self.register_buffer("grid", torch.stack([x, y], dim=-1).view(1, 784, 2))

    def get_masks(self):
        t = torch.linspace(0, 1, 8, device=self.points.device).view(1, 8, 1)
        p0, p1 = self.points[:, 0:1, :], self.points[:, 1:2, :]
        line1_pts = (1-t) * p0 + t * p1
        p2, p3 = self.points[:, 2:3, :], self.points[:, 3:4, :]
        line2_pts = (1-t) * p2 + t * p3
        all_pts = torch.cat([line1_pts, line2_pts], dim=1)
        diff = self.grid.unsqueeze(0) - all_pts.unsqueeze(2)
        dist_sq = torch.sum(diff**2, dim=-1)
        min_dist_sq, _ = torch.min(dist_sq, dim=1)
        sigma_pos = torch.exp(self.log_sigma_pos).view(-1, 1)
        sigma_neg = torch.exp(self.log_sigma_neg).view(-1, 1)
        stroke = torch.exp(-min_dist_sq / (2 * sigma_pos**2))
        surround = torch.exp(-min_dist_sq / (2 * sigma_neg**2))
        return stroke - 0.5 * surround

class DoubleLineNet(nn.Module):
    def __init__(self, num_neurons=256):
        super().__init__()
        self.line_layer = DoubleLineNeuronLayer(num_neurons=num_neurons)
        self.classifier = nn.Sequential(nn.ReLU(), nn.Linear(num_neurons, 128), nn.ReLU(), nn.Linear(128, 10))

def visualize():
    model = DoubleLineNet(num_neurons=256)
    weights_path = "v52_double_matchsticks.pth"
    if not os.path.exists(weights_path):
        print(f"❌ Error: {weights_path} not found.")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        masks = model.line_layer.get_masks()
        masks = masks.view(256, 28, 28).cpu().numpy()
        
    fig, axes = plt.subplots(16, 16, figsize=(14, 14))
    fig.suptitle("Double Matchstick Neurons (v52)", fontsize=16)
    for i in range(256):
        ax = axes[i // 16, i % 16]
        ax.imshow(masks[i], cmap='RdBu', vmin=-1, vmax=1)
        ax.axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("v52_double_matchsticks_gallery.png", dpi=150)
    print("🎨 Gallery saved as v52_double_matchsticks_gallery.png")
    plt.show()

if __name__ == "__main__":
    visualize()
