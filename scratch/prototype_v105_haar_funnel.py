import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import time

def get_haar_filters(device, dtype):
    kernel = torch.tensor([
        [[[1, 1], [1, 1]]],
        [[[1, -1], [1, -1]]],
        [[[1, 1], [-1, -1]]],
        [[[1, -1], [-1, 1]]]
    ], dtype=dtype, device=device) / 2.0
    return kernel

class HaarTransform2D(nn.Module):
    def __init__(self, levels=4):
        super().__init__()
        self.levels = levels

    def forward(self, x):
        # x: [B, 1, 32, 32]
        all_coeffs = []
        current_ll = x
        filters = get_haar_filters(x.device, x.dtype)
        
        for i in range(self.levels):
            out = F.conv2d(current_ll, filters, stride=2)
            current_ll = out[:, 0:1] # LL
            # Flatten spatial dims and add to list
            details = out[:, 1:].reshape(x.shape[0], -1)
            all_coeffs.append(details)
            
        all_coeffs.append(current_ll.reshape(x.shape[0], -1)) # Final DC
        return torch.cat(all_coeffs, dim=1)

class HaarFunnelNet(nn.Module):
    def __init__(self, hidden_dim=3):
        super().__init__()
        self.transform = HaarTransform2D(levels=4)
        self.input_dim = 1024
        
        # El "Embudo": de 1024 bordes a hidden_dim neuronas
        self.funnel = nn.Linear(self.input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = F.pad(x, (2, 2, 2, 2))
        coeffs = self.transform(x)
        
        # Usamos el valor absoluto para detectar energía de borde pura
        # Aplicamos BatchNorm para estabilizar el rango de los coeficientes Haar
        # (algunos niveles tienen mucha más energía que otros)
        feat = self.funnel(coeffs.abs())
        feat = F.relu(feat)
        return self.classifier(feat)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        import torch_directml
        if torch_directml.is_available() and device.type == 'cpu':
            device = torch_directml.device()
    except ImportError:
        pass
        
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Parámetros:
    # funnel: 1024 * 3 + 3 = 3075
    # classifier: 3 * 10 + 10 = 40
    # Total: 3115 parámetros
    model = HaarFunnelNet(hidden_dim=3).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parámetros entrenables: {total_params}")
    
    optimizer = optim.Adam(model.parameters(), lr=0.005) # Subimos un poco el LR para el embudo estrecho
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 11):
        start_time = time.time()
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        acc = 100. * correct / len(test_loader.dataset)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:02d} | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {acc:.2f}% | Time: {elapsed:.1f}s")

if __name__ == '__main__':
    train()
