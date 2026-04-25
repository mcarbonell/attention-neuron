import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time

class PolymorphicAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=2, mask_prob=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        std = math.sqrt(2.0 / in_features)
        self.register_buffer('w_init', torch.randn(out_features, in_features) * std)
        
        # Modulación Residual base (V1)
        self.delta_in_m = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        
        self.delta_in_a = nn.Parameter(torch.zeros(out_features, rank))
        self.delta_out_a = nn.Parameter(torch.zeros(rank, in_features))
        self.theta_bias = nn.Parameter(torch.zeros(out_features))
        
        # EL DIAL POLIMÓRFICO (Alpha):
        # Inicializamos en 0.0 logit -> sigmoid(0) = 0.5. 
        # Empieza prestando 50% de atención a Suma y 50% a Max.
        self.alpha_logits = nn.Parameter(torch.zeros(out_features))
        
        self.mask_prob = mask_prob

    def forward(self, x):
        w_m = torch.matmul(self.delta_in_m, self.delta_out_m)
        w_a = torch.matmul(self.delta_in_a, self.delta_out_a)
        
        if self.training and self.mask_prob < 1.0:
            mask = torch.bernoulli(torch.full(self.w_init.shape, self.mask_prob, device=self.w_init.device))
            w_evolved = torch.where(mask > 0, self.w_init + self.w_init * w_m + w_a, self.w_init)
        else:
            m_eff = self.mask_prob * w_m
            a_eff = self.mask_prob * w_a
            w_evolved = self.w_init + self.w_init * m_eff + a_eff
            
        # --- LÓGICA POLIMÓRFICA (SUM vs MAX) ---
        
        # 1. Calculamos la Suma clásica (rápida, multiplicando matrices)
        y_sum = torch.matmul(x, w_evolved.t())
        
        # 2. Calculamos el Max exacto 
        # Expandimos X: (Batch, 1, In) y W: (1, Out, In) para multiplicar elemento a elemento
        # Esto genera un tensor (Batch, Out, In) con todas las conexiones individuales
        z_elements = x.unsqueeze(1) * w_evolved.unsqueeze(0)
        # Buscamos la conexión que más se ha activado para cada neurona de salida
        y_max, _ = torch.max(z_elements, dim=2)
        
        # 3. Mezclamos usando el dial Alpha por neurona
        alpha = torch.sigmoid(self.alpha_logits) # Rango [0, 1]
        
        # alpha está en (Out,). Hacemos broadcast a (Batch, Out)
        y_mixed = alpha * y_sum + (1.0 - alpha) * y_max
            
        return y_mixed + torch.sin(self.theta_bias)

class PolymorphicAttentionMLP(nn.Module):
    def __init__(self, mask_prob=0.5):
        super().__init__()
        self.layer1 = PolymorphicAttentionLayer(784, 512, rank=2, mask_prob=mask_prob)
        self.layer2 = PolymorphicAttentionLayer(512, 10, rank=2, mask_prob=mask_prob)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking V13 (Polymorphic Attention: Sum vs Max) with ADAM on: {device}")

    BATCH_SIZE = 256
    EPOCHS = 10
    LR = 0.001

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    model = PolymorphicAttentionMLP(mask_prob=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        acc = correct / 10000
        print(f"Epoch {epoch} | Test Acc: {acc:.4f} | Time: {time.time() - t0:.1f}s")
        
    # Análisis post-entrenamiento de los diales:
    print("\n--- Análisis de la Identidad Neuronal ---")
    for name, layer in [("Capa Oculta (512)", model.layer1), ("Capa Salida (10)", model.layer2)]:
        alphas = torch.sigmoid(layer.alpha_logits).detach().cpu().numpy()
        sum_neurons = sum(1 for a in alphas if a > 0.6)
        max_neurons = sum(1 for a in alphas if a < 0.4)
        hybrid_neurons = len(alphas) - sum_neurons - max_neurons
        
        print(f"{name}:")
        print(f"  - Preferencia SUM (alpha > 0.6): {sum_neurons} neuronas")
        print(f"  - Preferencia MAX (alpha < 0.4): {max_neurons} neuronas")
        print(f"  - Híbridas (0.4 < alpha < 0.6): {hybrid_neurons} neuronas")

if __name__ == "__main__":
    main()
