import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import time

class TriangleAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Parámetros por neurona: centro, amplitud, ancho (half_width) y bias
        # Inicializamos los centros repartidos uniformemente por el espacio de entrada [0, 1]
        self.center = nn.Parameter(torch.linspace(0.1, 0.9, out_features))
        self.amplitude = nn.Parameter(torch.ones(out_features))
        self.half_width = nn.Parameter(torch.ones(out_features) * 0.1) # 10% del espacio
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Posiciones fijas para las entradas, normalizadas entre 0 y 1
        self.register_buffer('input_positions', torch.linspace(0, 1, in_features))

    def get_weights(self):
        # input_positions: (1, in_features)
        # center: (out_features, 1)
        pos = self.input_positions.unsqueeze(0)
        c = self.center.unsqueeze(1)
        
        # Aseguramos que el ancho sea positivo para evitar divisiones por cero o negativas
        w = self.half_width.unsqueeze(1).abs() + 1e-4 
        a = self.amplitude.unsqueeze(1)
        
        # Calculamos la forma del triángulo: max(0, 1 - |pos - centro| / ancho) * amplitud
        dist = torch.abs(pos - c)
        base_weight = F.relu(1.0 - (dist / w))
        weights = base_weight * a
        return weights

    def forward(self, x):
        # x: (batch_size, in_features)
        # Generamos los pesos al vuelo en cada forward pass
        weights = self.get_weights() # (out_features, in_features)
        
        # Multiplicación matricial tradicional
        return F.linear(x, weights, self.bias)

class TriangleNet(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.flatten = nn.Flatten()
        # Capa 1: de los 784 píxeles a las neuronas ocultas
        self.layer1 = TriangleAttentionLayer(28*28, hidden_size)
        # Capa 2: de las neuronas ocultas a las 10 clases de salida
        self.layer2 = TriangleAttentionLayer(hidden_size, 10)

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

    model = TriangleNet(hidden_size=2560).to(device)
    
    # Calcular cantidad de parámetros
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parámetros entrenables: {total_params} !!!")
    
    # Red tradicional: 784*256 + 256 + 256*10 + 10 = 203,530
    # Nuestra red: (256*4) + (10*4) = 1,064

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
