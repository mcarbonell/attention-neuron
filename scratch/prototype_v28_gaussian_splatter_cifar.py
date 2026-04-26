import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import json
import os
import math

class GaussianSplatLayer2D(nn.Module):
    """
    V28 Component: 2D Gaussian Splatting Layer.
    Instead of discrete convolution kernels, it learns continuous Gaussian ovals
    (Splats) defined by center (x,y), spread (sigma_x, sigma_y), rotation (rho), and amplitude (A).
    """
    def __init__(self, in_channels, out_channels, height=32, width=32, splats_per_filter=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.H = height
        self.W = width
        self.splats_per_filter = splats_per_filter
        
        # Grid of coordinates [-1, 1] for spatial stability
        y = torch.linspace(-1, 1, self.H)
        x = torch.linspace(-1, 1, self.W)
        self.register_buffer('grid_y', y.view(self.H, 1).expand(self.H, self.W))
        self.register_buffer('grid_x', x.view(1, self.W).expand(self.H, self.W))
        
        # Splat Parameters: Shape (out_channels, in_channels, splats_per_filter)
        # Initialize centers randomly across the image [-0.8, 0.8]
        self.cx = nn.Parameter(torch.rand(out_channels, in_channels, splats_per_filter) * 1.6 - 0.8)
        self.cy = nn.Parameter(torch.rand(out_channels, in_channels, splats_per_filter) * 1.6 - 0.8)
        
        # Initialize spreads (log-scale to ensure positivity via exp)
        # Start with relatively small, localized splats
        self.log_sig_x = nn.Parameter(torch.ones(out_channels, in_channels, splats_per_filter) * -2.0)
        self.log_sig_y = nn.Parameter(torch.ones(out_channels, in_channels, splats_per_filter) * -2.0)
        
        # Rotation parameter (tanh will constrain it to [-1, 1] for valid correlation)
        self.rho_raw = nn.Parameter(torch.zeros(out_channels, in_channels, splats_per_filter))
        
        # Amplitude (Weight of the splat)
        std = math.sqrt(2.0 / (in_channels * splats_per_filter))
        self.amplitude = nn.Parameter(torch.randn(out_channels, in_channels, splats_per_filter) * std)
        
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def _generate_filters(self):
        """
        Renders the continuous Gaussian parameters into discrete spatial filters (H x W).
        Returns tensor of shape (out_channels, in_channels, H, W)
        """
        # Ensure positive variances and valid correlation
        sig_x = torch.exp(self.log_sig_x) + 1e-4 # (out, in, splats)
        sig_y = torch.exp(self.log_sig_y) + 1e-4
        rho = torch.tanh(self.rho_raw) * 0.95 # Constrain strictly between (-1, 1)
        
        # Reshape grid for broadcasting: (1, 1, 1, H, W)
        gx = self.grid_x.view(1, 1, 1, self.H, self.W)
        gy = self.grid_y.view(1, 1, 1, self.H, self.W)
        
        # Reshape params for broadcasting: (out, in, splats, 1, 1)
        cx = self.cx.view(self.out_channels, self.in_channels, self.splats_per_filter, 1, 1)
        cy = self.cy.view(self.out_channels, self.in_channels, self.splats_per_filter, 1, 1)
        sx = sig_x.view(self.out_channels, self.in_channels, self.splats_per_filter, 1, 1)
        sy = sig_y.view(self.out_channels, self.in_channels, self.splats_per_filter, 1, 1)
        r = rho.view(self.out_channels, self.in_channels, self.splats_per_filter, 1, 1)
        amp = self.amplitude.view(self.out_channels, self.in_channels, self.splats_per_filter, 1, 1)
        
        # Calculate Gaussian exponent (Mahalanobis distance)
        dx = (gx - cx) / sx
        dy = (gy - cy) / sy
        z = (dx**2 - 2*r*dx*dy + dy**2) / (1 - r**2)
        
        # Gaussian function
        # Shape: (out, in, splats, H, W)
        gaussians = amp * torch.exp(-0.5 * z)
        
        # Sum over the splats to get the final spatial filter
        # Shape: (out, in, H, W)
        filters = torch.sum(gaussians, dim=2)
        return filters

    def forward(self, x):
        """
        x shape: (batch, in_channels, H, W)
        Since the splats cover the whole image (or a large receptive field), 
        we apply it as a fully connected spatial operation or a depthwise spatial contraction.
        Here we use it as a global spatial attention mask (equivalent to a 32x32 convolution without padding).
        """
        B, C, H, W = x.shape
        # Generate the dynamic spatial filters
        # filters shape: (out_channels, in_channels, H, W)
        filters = self._generate_filters()
        
        # Apply the filters over the entire spatial dimensions
        # x is (B, in_C, H, W). We want output (B, out_C)
        # Expand x to match filters: (B, 1, in_C, H, W)
        # Expand filters to match x: (1, out_C, in_C, H, W)
        # Element-wise multiply and sum over in_C, H, W
        
        # More efficient way using einsum or viewing:
        x_flat = x.view(B, C, -1) # (B, in_C, H*W)
        filters_flat = filters.view(self.out_channels, self.in_channels, -1) # (out_C, in_C, H*W)
        
        # Batched matrix multiplication:
        # y_bc = sum_{i, h, w} x_{b, i, h, w} * w_{c, i, h, w}
        y = torch.einsum('bil,cil->bc', x_flat, filters_flat) + self.bias
        
        # We output a 1x1 spatial tensor to fit into standard CNN flows if needed, 
        # or just treat it as a flattened output.
        return y.view(B, self.out_channels, 1, 1)

class GaussianSplatterNet(nn.Module):
    def __init__(self, splats_per_filter=4):
        super().__init__()
        # Since splatting looks at the whole image, we can do a very shallow network.
        # It's essentially a highly parameter-efficient MLP.
        self.splat1 = GaussianSplatLayer2D(in_channels=3, out_channels=128, height=32, width=32, splats_per_filter=splats_per_filter)
        self.bn1 = nn.BatchNorm1d(128)
        
        # The output of splat1 is (B, 128, 1, 1). 
        # We can just use standard dense layers from here.
        self.fc1 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 10)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Splat layer extracts 128 global features
        x = self.splat1(x)
        x = x.view(x.size(0), -1) # Flatten the 1x1
        
        x = self.relu(self.bn1(x))
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.fc1(x)))
        x = self.fc2(x)
        return x

def main():
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training V28 'THE GAUSSIAN SPLATTER 2D' on: {device}")
    
    BATCH_SIZE = 128
    EPOCHS = 50
    SPLATS = 8 # 8 ovals per feature
    MAX_LR = 0.01 # Splats might need a higher LR to move around quickly
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # We avoid RandomCrop initially because the splats learn absolute spatial positions.
        # If we crop, the center shifts. We'll start with pure spatial learning.
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
    
    os.makedirs("results", exist_ok=True)
    
    model = GaussianSplatterNet(splats_per_filter=SPLATS).to(device)
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
        t_epoch_start = time.time()
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
            torch.save(model.state_dict(), "results/v28_splatter_best.pt")
            
        t_now = time.time()
        elapsed = t_now - t_start
        epoch_time = t_now - t_epoch_start
        eta = (elapsed / epoch) * (EPOCHS - epoch)
        
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] Epoch {epoch:2d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Acc: {acc:.4f} | Best: {best_acc:.4f} | Time: {epoch_time:.1f}s | ETA: {eta/60:.1f}m")

    t_total = time.time() - t_start
    print(f"\nGAUSSIAN SPLATTER finished in {t_total:.1f}s")
    print(f"Final Best Test Accuracy: {best_acc:.4f}")
    
    results = {
        "model": "V28_Gaussian_Splatter",
        "splats_per_filter": SPLATS,
        "trainable_params": params,
        "best_acc": best_acc,
        "epochs": EPOCHS,
        "wall_clock_time": t_total,
        "dataset": "CIFAR-10"
    }
    
    with open("results/raw/v28_splatter_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
