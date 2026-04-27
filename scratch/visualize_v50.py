import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math

# Use the same class definition as the prototype
class StrokeNeuronLayer(nn.Module):
    def __init__(self, num_neurons=256):
        super().__init__()
        self.num_neurons = num_neurons
        self.points = nn.Parameter(torch.rand(num_neurons, 3, 2) * 28.0)
        self.log_sigma_pos = nn.Parameter(torch.full((num_neurons,), math.log(1.5)))
        self.log_sigma_neg = nn.Parameter(torch.full((num_neurons,), math.log(3.0)))
        y, x = torch.meshgrid(torch.linspace(0, 27, 28), torch.linspace(0, 27, 28), indexing='ij')
        self.register_buffer("grid", torch.stack([x, y], dim=-1).view(1, 784, 2))

    def get_masks(self):
        t = torch.linspace(0, 1, 12, device=self.points.device).view(1, 12, 1)
        p0, p1, p2 = self.points[:, 0:1, :], self.points[:, 1:2, :], self.points[:, 2:3, :]
        bezier_points = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
        diff = self.grid.unsqueeze(0) - bezier_points.unsqueeze(2)
        dist_sq = torch.sum(diff**2, dim=-1)
        min_dist_sq, _ = torch.min(dist_sq, dim=1)
        sigma_pos = torch.exp(self.log_sigma_pos).view(-1, 1)
        sigma_neg = torch.exp(self.log_sigma_neg).view(-1, 1)
        stroke = torch.exp(-min_dist_sq / (2 * sigma_pos**2))
        surround = torch.exp(-min_dist_sq / (2 * sigma_neg**2))
        return stroke - 0.6 * surround

class StrokeNet(nn.Module):
    def __init__(self, num_neurons=256):
        super().__init__()
        self.stroke_layer = StrokeNeuronLayer(num_neurons=num_neurons)
        self.classifier = nn.Sequential(nn.ReLU(), nn.Linear(num_neurons, 128), nn.ReLU(), nn.Linear(128, 10))

def visualize():
    model = StrokeNet(num_neurons=256)
    if not torch.os.path.exists("v50b_strokes.pth"):
        print("❌ Error: v50b_strokes.pth not found. Please train the model first.")
        return
        
    model.load_state_dict(torch.load("v50b_strokes.pth", map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        masks = model.stroke_layer.get_masks() # (256, 784)
        masks = masks.view(256, 28, 28).cpu().numpy()
        
    fig, axes = plt.subplots(16, 16, figsize=(12, 12))
    for i in range(256):
        ax = axes[i // 16, i % 16]
        ax.imshow(masks[i], cmap='RdBu', vmin=-1, vmax=1)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig("v50b_gallery.png")
    plt.show()
    print("🎨 Gallery saved as v50b_gallery.png")

if __name__ == "__main__":
    visualize()
