import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import math
import os
import json

# --- 1. SPECTRAL COMPONENTS (From V124) ---

def get_walsh_matrix_sequency(N):
    def get_walsh(n):
        if n == 1: return torch.tensor([[1.0]])
        h_prev = get_walsh(n // 2)
        return torch.cat([torch.cat([h_prev, h_prev], dim=1),
                          torch.cat([h_prev, -h_prev], dim=1)], dim=0)
    H = get_walsh(N)
    crossings = [( (H[i, :-1] * H[i, 1:] < 0).sum().item(), i) for i in range(N)]
    crossings.sort()
    return H[[idx for _, idx in crossings]]

def iwalsh_2d(coeffs, H):
    N = H.shape[0]
    return torch.matmul(H.t(), torch.matmul(coeffs, H)) / (N * N)

class SmoothWalshLayer(nn.Module):
    def __init__(self, in_size, out_features, K=8, N=32):
        super().__init__()
        self.out_features = out_features
        self.K = K
        self.N = N # Resolution of the weight field (e.g. 32x32)
        # Parameters are SPECTRAL COEFFICIENTS
        self.spectral_core = nn.Parameter(torch.randn(out_features, K, K) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('H_K', get_walsh_matrix_sequency(K))

    def get_weights(self):
        # Inverse Walsh to get KxK block
        w_mini = iwalsh_2d(self.spectral_core, self.H_K).unsqueeze(1) # [Out, 1, K, K]
        # Smooth upsample to NxN field
        w_N = F.interpolate(w_mini, size=(self.N, self.N), mode='bilinear', align_corners=True)
        return w_N.view(self.out_features, -1)

    def forward(self, x):
        return F.linear(x, self.get_weights(), self.bias)

# --- 2. SPECTRAL OPTIMIZER (SWO V125) ---

class SmoothSpectralAdam(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 k_ratio=0.25, mode='bilinear'):
        defaults = dict(lr=lr, betas=betas, eps=eps, k_ratio=k_ratio, mode=mode)
        super(SmoothSpectralAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    if p.dim() >= 2:
                        s = p.shape
                        kh = max(2, int(s[-2] * group['k_ratio']))
                        kw = max(2, int(s[-1] * group['k_ratio']))
                        state['k_size'] = (kh, kw)
                        # Always store as 4D: [N, C, kh, kw]
                        if p.dim() == 2:
                            state['exp_avg'] = torch.zeros((1, 1, kh, kw), device=p.device)
                            state['exp_avg_sq'] = torch.zeros((1, 1, kh, kw), device=p.device)
                        else:
                            state['exp_avg'] = torch.zeros((s[0], 1, kh, kw), device=p.device)
                            state['exp_avg_sq'] = torch.zeros((s[0], 1, kh, kw), device=p.device)
                    else:
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                beta1, beta2 = group['betas']
                
                if p.dim() >= 2:
                    s = p.shape
                    kh, kw = state['k_size']
                    
                    # Reshape to 4D for interpolation [N, C, H, W]
                    if p.dim() == 2:
                        g_view = grad.view(1, 1, s[0], s[1])
                    else: # 3D or more
                        g_view = grad.unsqueeze(1) if p.dim() == 3 else grad.view(-1, 1, s[-2], s[-1])
                    
                    g_small = F.interpolate(g_view, size=(kh, kw), mode=group['mode'], align_corners=True)
                    
                    state['exp_avg'].mul_(beta1).add_(g_small, alpha=1 - beta1)
                    state['exp_avg_sq'].mul_(beta2).addcmul_(g_small, g_small, value=1 - beta2)
                    
                    bc1 = 1 - beta1 ** state['step']
                    bc2 = 1 - beta2 ** state['step']
                    
                    # Reconstruct and reshape back
                    m_rec_view = F.interpolate(state['exp_avg'], size=s[-2:], mode=group['mode'], align_corners=True)
                    v_rec_view = F.interpolate(state['exp_avg_sq'], size=s[-2:], mode=group['mode'], align_corners=True)
                    
                    m_rec = m_rec_view.view(s)
                    v_rec = v_rec_view.view(s)
                    v_rec.clamp_(min=0.0) 
                    
                    denom = (v_rec.sqrt() / math.sqrt(bc2)).add_(group['eps'])
                    step_size = group['lr'] / bc1
                    p.addcdiv_(m_rec, denom, value=-step_size)
                else:
                    state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
                    state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    bc1 = 1 - beta1 ** state['step']
                    bc2 = 1 - beta2 ** state['step']
                    denom = (state['exp_avg_sq'].sqrt() / math.sqrt(bc2)).add_(group['eps'])
                    step_size = group['lr'] / bc1
                    p.addcdiv_(state['exp_avg'], denom, value=-step_size)

        return loss

# --- 3. THE EXPERIMENT ---

class TotalSpectralModel(nn.Module):
    def __init__(self, h1=512, h2=256, k=8):
        super().__init__()
        # Input 28x28 -> Pad to 32x32 -> 1024
        self.layer1 = SmoothWalshLayer(1024, h1, K=k, N=32)
        self.bn1 = nn.BatchNorm1d(h1)
        # Hidden layer 2 (treating 1D output as 1D sequence for Walsh?) 
        # Actually for simplicity, we use standard Linear here or another Walsh if we reshape
        # Let's use a 1D Walsh-like projection for the hidden layer
        self.layer2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc_out = nn.Linear(h2, 10)

    def forward(self, x):
        x = F.pad(x, (2, 2, 2, 2)).view(x.size(0), -1)
        x = F.relu(self.bn1(self.layer1(x)))
        x = F.relu(self.bn2(self.layer2(x)))
        return self.fc_out(x)

def main():
    device = torch.device("cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=256, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)

    model = TotalSpectralModel(k=8).to(device)
    
    # We apply SWO to the WHOLE model
    # Weights are already small, and Adam states will be EVEN smaller
    optimizer = SmoothSpectralAdam(model.parameters(), lr=0.002, k_ratio=0.25)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params:,}")
    # Estimated State RAM:
    # Layer 1 is compressed 16x. Layer 2 & 3 are compressed 16x in the optimizer.
    # Standard Adam for this model would be total_params * 8 bytes.
    # SWO Adam will be roughly total_params * 8 * 0.0625 bytes.
    print(f"Estimated Optimizer RAM: {total_params * 2 * 4 * 0.0625 / 1024:.2f} KB")

    print("\nStarting Training (v126: Total Spectral Entropy)...")
    start_time = time.time()
    for epoch in range(5):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch} [{batch_idx*len(data):5d}/60000] Loss: {loss.item():.4f}")

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                correct += output.argmax(dim=1).eq(target).sum().item()
        print(f"  Epoch {epoch} Test Acc: {100. * correct / 10000:.2f}%")

    print(f"\nTotal Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
