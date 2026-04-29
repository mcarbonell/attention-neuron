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
            # out[:, 1:] contains HL, LH, HH
            details = out[:, 1:].reshape(x.shape[0], -1)
            all_coeffs.append(details)
            
        all_coeffs.append(current_ll.reshape(x.shape[0], -1)) # Final DC
        return torch.cat(all_coeffs, dim=1)

class SelectiveHaarLayer(nn.Module):
    def __init__(self, input_dim, out_features, selection_rank=4):
        super().__init__()
        self.input_dim = input_dim
        self.out_features = out_features
        self.selection_rank = selection_rank
        
        # En lugar de un peso por cada coeficiente ( input_dim * out_features )
        # Aprendemos una máscara de selección de bajo rango.
        # Esto obliga a las neuronas a compartir "puntos de interés" en el espectro Haar.
        self.u = nn.Parameter(torch.randn(out_features, selection_rank) * 0.1)
        self.v = nn.Parameter(torch.randn(selection_rank, input_dim) * 0.1)
        
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def get_weights(self):
        # Reconstruye la matriz de pesos: [out_features, input_dim]
        # Esto es equivalente a que cada neurona sea una combinación lineal de 'selection_rank' prototipos
        return torch.matmul(self.u, self.v)

    def forward(self, x):
        weights = self.get_weights()
        return F.linear(x, weights, self.bias)

class SelectiveHaarNet(nn.Module):
    def __init__(self, hidden_dim=64, selection_rank=8):
        super().__init__()
        self.transform = HaarTransform2D(levels=4)
        
        # Input dim calculation for 32x32:
        # L1: 3 * (16*16) = 768
        # L2: 3 * (8*8)   = 192
        # L3: 3 * (4*4)   = 48
        # L4: 3 * (2*2)   = 12
        # DC: 2*2         = 4
        # Total = 1024
        self.input_dim = 1024
        
        # Capa selectiva: en lugar de 1024*hidden_dim parámetros,
        # usa (hidden_dim*rank + rank*1024)
        self.selective_layer = SelectiveHaarLayer(self.input_dim, hidden_dim, selection_rank=selection_rank)
        
        self.classifier = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = F.pad(x, (2, 2, 2, 2))
        coeffs = self.transform(x)
        # Usamos el valor absoluto de los coeficientes para detectar "energía de borde"
        # independientemente de si es un borde blanco-negro o negro-blanco
        features = F.relu(self.selective_layer(coeffs.abs()))
        return self.classifier(features)

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
    # selective_layer: (64*8 + 8*1024) = 512 + 8192 = 8704 (un poco alto)
    # Vamos a bajar el rank a 2 y hidden a 48 para competir con los 3.8k de la V101
    # (48*2 + 2*1024) = 96 + 2048 = 2144
    # classifier: 48*10 = 480
    # Total = ~2624 + biases = ~2700 parámetros.
    model = SelectiveHaarNet(hidden_dim=48, selection_rank=2).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parámetros entrenables: {total_params}")
    
    optimizer = optim.Adam(model.parameters(), lr=0.002)
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
