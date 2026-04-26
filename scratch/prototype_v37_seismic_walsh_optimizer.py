import torch
import torch.nn as nn
import torch.optim as optim
import math
import time

# --- FWHT Core para el Optimizador ---
def fwht(x):
    # Soporta (N,) o (B, N)
    original_shape = x.shape
    if len(original_shape) == 1:
        x = x.unsqueeze(0)
    
    B, N = x.shape
    h = 1
    while h < N:
        x = x.view(B, N // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = torch.stack([a + b, a - b], dim=2)
        h *= 2
    
    res = x.view(B, N)
    return res if len(original_shape) > 1 else res.squeeze(0)

def ifwht(x):
    N = x.shape[-1]
    return fwht(x) / N

# --- Seismic Walsh Optimizer ---

class SeismicWalshOptimizer:
    """
    V37: THE SEISMIC WALSH OPTIMIZER.
    Implements the 'Move the Ground' philosophy from Seismic Descent
    using Walsh-structured noise to escape local minima.
    """
    def __init__(self, parameters, base_lr=0.001, a0=0.01, freq0=0.1):
        self.params = list(parameters)
        self.lr = base_lr
        self.a0 = a0           # Amplitud máxima del terremoto
        self.freq0 = freq0     # Frecuencia base de oscilación
        self.t = 0             # Tiempo interno (pasos)
        
        # Almacenamos vectores de energía sísmica por cada parámetro
        self.seismic_energies = []
        for p in self.params:
            # La FWHT requiere potencia de 2. Buscamos la potencia de 2 superior.
            n = p.numel()
            n_pow2 = 2**math.ceil(math.log2(n))
            # Energía sísmica inicial en el dominio de Walsh
            energy = torch.randn(n_pow2, device=p.device) * 0.1
            self.seismic_energies.append(energy)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self.t += 1
        
        # 1. Calcular amplitud actual del terremoto (Seismic Schedule)
        # A(t) = A0 * sin(t * freq)
        # La frecuencia decae lentamente para estabilizar al final
        current_freq = self.freq0 / (1 + self.t * 0.001)
        amplitude = self.a0 * math.sin(self.t * current_freq)
        
        # 2. Aplicar la actualización + la deformación del terreno
        with torch.no_grad():
            for p, energy in zip(self.params, self.seismic_energies):
                if p.grad is None:
                    continue
                
                # A. Descenso de gradiente estándar
                p.data.add_(p.grad, alpha=-self.lr)
                
                # B. Deformación Sísmica Estructurada (Walsh)
                # Generamos el ruido sísmico proyectando la energía 
                # del dominio Walsh al espacio del parámetro.
                n = p.numel()
                n_pow2 = energy.shape[0]
                
                # La vibración es una onda de Walsh "viva"
                vibration_walsh = energy * math.cos(self.t * current_freq * 0.5)
                vibration_spatial = ifwht(vibration_walsh)
                
                # Recortamos a la medida real del parámetro
                seismic_kick = vibration_spatial[:n].view(p.shape)
                
                # Movemos el suelo!
                p.data.add_(seismic_kick, alpha=amplitude)
                
                # C. Evolución de la energía sísmica (Drift)
                # La energía sísmica cambia lentamente para que el siguiente 
                # "terremoto" tenga una estructura ligeramente distinta.
                energy.add_(torch.randn_like(energy) * 0.01)
                energy.div_(energy.norm() + 1e-8).mul_(math.sqrt(n_pow2)) # Normalizar

        return loss

# --- Script de Prueba (MNIST con Accuracy) ---

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training V37 'SEISMIC WALSH DESCENT' on MNIST: {device}")
    
    # Un modelo ultra-simple para ver el efecto
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 10)
    ).to(device)
    
    # Cargamos MNIST
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000, shuffle=False)
    
    # Inicializamos nuestro Optimizador Sísmico
    optimizer = SeismicWalshOptimizer(model.parameters(), base_lr=0.0001, a0=0.001, freq0=0.1)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    t0 = time.time()
    for epoch in range(1, 11):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(-1, 784).to(device)
            target = target.to(device)
            
            optimizer.step(closure=lambda: criterion(model(data), target).backward() or True)
            
        # Evaluación al final de la época
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.view(-1, 784).to(device), target.to(device)
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        acc = correct / 10000
        if acc > best_acc: best_acc = acc
        
        current_freq = optimizer.freq0 / (1 + optimizer.t * 0.001)
        amp = optimizer.a0 * math.sin(optimizer.t * current_freq)
        print(f"Epoch {epoch:2d}/10 | Acc: {acc:.4f} | Best: {best_acc:.4f} | Seismic Amp: {amp:+.4f} | Time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
