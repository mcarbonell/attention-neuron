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
        # Inicializamos los centros esparcidos al azar por la imagen [0, 1]
        self.center_x = nn.Parameter(torch.rand(out_features))
        self.center_y = nn.Parameter(torch.rand(out_features))
        self.amplitude = nn.Parameter(torch.ones(out_features))
        self.radius = nn.Parameter(torch.ones(out_features) * 0.2) # 20% del tamaño por defecto
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Crear cuadrícula de coordenadas (X, Y) normalizadas [0, 1] para cada píxel
        y_coords = torch.linspace(0, 1, in_height)
        x_coords = torch.linspace(0, 1, in_width)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Aplanamos la cuadrícula: shape (784, 2)
        grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        self.register_buffer('grid_positions', grid)

    def get_weights(self):
        # grid_positions: (in_features, 2) -> (1, in_features, 2)
        pos = self.grid_positions.unsqueeze(0)
        
        # centers: (out_features, 2) -> (out_features, 1, 2)
        cx = self.center_x.unsqueeze(1)
        cy = self.center_y.unsqueeze(1)
        centers = torch.cat([cx, cy], dim=1).unsqueeze(1) 
        # Espera, cat: cx es (out, 1), cy es (out, 1). cat dim=1 es (out, 2). Luego unsqueeze(1) -> (out, 1, 2)
        centers = torch.cat([cx, cy], dim=1).unsqueeze(1)
        
        # Calculamos distancia euclidiana de cada centro a todos los píxeles
        # dist shape: (out_features, in_features)
        dist = torch.norm(pos - centers, p=2, dim=2)
        
        r = self.radius.unsqueeze(1).abs() + 1e-4
        a = self.amplitude.unsqueeze(1)
        
        # Cono 2D: max(0, 1 - distancia / radio) * amplitud
        base_weight = F.relu(1.0 - (dist / r))
        weights = base_weight * a
        return weights

    def forward(self, x):
        weights = self.get_weights() # (out_features, in_features)
        return F.linear(x, weights, self.bias)

class TriangleAttention1DLayer(nn.Module):
    # Reutilizamos el triángulo 1D para la capa de salida (capa 2)
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.center = nn.Parameter(torch.linspace(0.1, 0.9, out_features))
        self.amplitude = nn.Parameter(torch.ones(out_features))
        self.half_width = nn.Parameter(torch.ones(out_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('input_positions', torch.linspace(0, 1, in_features))

    def get_weights(self):
        pos = self.input_positions.unsqueeze(0)
        c = self.center.unsqueeze(1)
        w = self.half_width.unsqueeze(1).abs() + 1e-4 
        a = self.amplitude.unsqueeze(1)
        dist = torch.abs(pos - c)
        base_weight = F.relu(1.0 - (dist / w))
        weights = base_weight * a
        return weights

    def forward(self, x):
        weights = self.get_weights()
        return F.linear(x, weights, self.bias)

class ConeNet(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.flatten = nn.Flatten()
        # Capa 1: de imagen 2D (28x28) a neuronas ocultas usando el Cono
        self.layer1 = ConeAttention2DLayer(28, 28, hidden_size)
        # Capa 2: de las neuronas ocultas a las 10 clases usando Triángulo 1D
        self.layer2 = TriangleAttention1DLayer(hidden_size, 10)

    def forward(self, x):
        x = self.flatten(x)
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

    model = ConeNet(hidden_size=2560).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parámetros entrenables: {total_params} !!!")
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
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
