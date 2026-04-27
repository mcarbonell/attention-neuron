import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import os
import numpy as np

class RGBMatchstickLayer(nn.Module):
    def __init__(self, num_neurons=512):
        super().__init__()
        self.num_neurons = num_neurons
        self.points = nn.Parameter(torch.rand(num_neurons, 2, 2) * 31.0)
        self.color_weights = nn.Parameter(torch.randn(num_neurons, 3))
        self.log_sigma_pos = nn.Parameter(torch.full((num_neurons,), math.log(1.5)))
        self.log_sigma_neg = nn.Parameter(torch.full((num_neurons,), math.log(3.5)))
        y, x = torch.meshgrid(torch.linspace(0, 31, 32), torch.linspace(0, 31, 32), indexing='ij')
        self.register_buffer("grid", torch.stack([x, y], dim=-1).view(1, 1024, 2))

    def get_masks(self):
        t = torch.linspace(0, 1, 10, device=self.points.device).view(1, 10, 1)
        p0, p1 = self.points[:, 0:1, :], self.points[:, 1:2, :]
        line_points = (1-t) * p0 + t * p1
        diff = self.grid.unsqueeze(0) - line_points.unsqueeze(2)
        dist_sq = torch.sum(diff**2, dim=-1)
        min_dist_sq, _ = torch.min(dist_sq, dim=1)
        sigma_pos = torch.exp(self.log_sigma_pos).view(-1, 1)
        sigma_neg = torch.exp(self.log_sigma_neg).view(-1, 1)
        stroke = torch.exp(-min_dist_sq / (2 * sigma_pos**2))
        surround = torch.exp(-min_dist_sq / (2 * sigma_neg**2))
        return stroke - 0.4 * surround

class CIFARMatchNet(nn.Module):
    def __init__(self, num_neurons=512):
        super().__init__()
        self.match_layer = RGBMatchstickLayer(num_neurons=num_neurons)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10)
        )

def visualize():
    # We'll visualize the first 256 neurons for a clean 16x16 grid
    num_to_viz = 256
    model = CIFARMatchNet(num_neurons=512)
    
    # We need to save the model in the trainer first if we haven't
    weights_path = "v54_matchsticks_cifar.pth"
    if not os.path.exists(weights_path):
        print(f"❌ Error: {weights_path} not found. Please train the model and ensure it saves the weights.")
        return
        
    print(f"Loading weights from {weights_path}...")
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        masks = model.match_layer.get_masks()[:num_to_viz] # (256, 1024)
        colors = model.match_layer.color_weights[:num_to_viz] # (256, 3)
        
        # Normalize masks for display [0, 1]
        masks = (masks - masks.min()) / (masks.max() - masks.min() + 1e-8)
        masks = masks.view(num_to_viz, 32, 32)
        
        # Convert color weights to actual RGB colors for visualization
        # We'll use a sigmoid to squash them into [0, 1] for display purposes
        viz_colors = torch.sigmoid(colors).cpu().numpy()
        
    fig, axes = plt.subplots(16, 16, figsize=(16, 16))
    fig.suptitle("RGB Matchstick Neurons (v54) - CIFAR-10 Feature Detectors", fontsize=20)
    
    for i in range(num_to_viz):
        ax = axes[i // 16, i % 16]
        
        # Create an RGB image for this neuron
        # Mask is (32, 32), Color is (3,)
        mask_np = masks[i].cpu().numpy()
        color_np = viz_colors[i]
        
        # Multiplicative color blending: mask * color
        neuron_rgb = np.zeros((32, 32, 3))
        for c in range(3):
            neuron_rgb[:, :, c] = mask_np * color_np[c]
            
        ax.imshow(neuron_rgb)
        ax.axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_file = "v54_cifar_gallery.png"
    plt.savefig(output_file, dpi=150)
    print(f"🎨 RGB Gallery saved as {output_file}")
    plt.show()

if __name__ == "__main__":
    visualize()
