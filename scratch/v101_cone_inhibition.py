import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import time

class ConeAttention2DLayer(nn.Module):
    def __init__(self, in_height, in_width, out_features):
        super().__init__()
        self.in_height = in_height
        self.in_width = in_width
        self.in_features = in_height * in_width
        self.out_features = out_features
        
        # Parámetros por neurona: centro (X e Y), amplitud, radio y bias
        self.center_x = nn.Parameter(torch.rand(out_features))
        self.center_y = nn.Parameter(torch.rand(out_features))
        
        # ¡INHIBICIÓN!: Inicializamos la amplitud entre -1 y 1
        # Así tenemos conos "positivos" (excitatorios) y "negativos" (inhibitorios) desde el inicio
        self.amplitude = nn.Parameter(torch.empty(out_features).uniform_(-1.0, 1.0))
        
        self.radius = nn.Parameter(torch.ones(out_features) * 0.2)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Crear cuadrícula de coordenadas (X, Y) normalizadas [0, 1]
        y_coords = torch.linspace(0, 1, in_height)
        x_coords = torch.linspace(0, 1, in_width)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        self.register_buffer('grid_positions', grid)

    def get_weights(self):
        pos = self.grid_positions.unsqueeze(0)
        cx = self.center_x.unsqueeze(1)
        cy = self.center_y.unsqueeze(1)
        centers = torch.cat([cx, cy], dim=1).unsqueeze(1)
        
        dist = torch.norm(pos - centers, p=2, dim=2)
        r = self.radius.unsqueeze(1).abs() + 1e-4
        a = self.amplitude.unsqueeze(1)
        
        base_weight = F.relu(1.0 - (dist / r))
        weights = base_weight * a
        return weights

    def forward(self, x):
        weights = self.get_weights()
        return F.linear(x, weights, self.bias)

class ConeNetInhibition(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.flatten = nn.Flatten()
        # Capa 1: Cono 2D (Extracción de características espaciales con inhibición)
        self.layer1 = ConeAttention2DLayer(28, 28, hidden_size)
        # Capa 2: Densa estándar
        self.layer2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.flatten(x)
        # Usamos ReLU. Si un cono inhibitorio ve blanco, la suma final bajará.
        # Si la suma es muy negativa, ReLU la dejará en 0, lo cual es correcto
        # biológicamente (tasa de disparo no puede ser negativa).
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

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

    model = ConeNetInhibition(hidden_size=256).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parámetros entrenables: {total_params} !!!")
    
    # Bajamos el learning rate a 0.001 como sugeriste
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
