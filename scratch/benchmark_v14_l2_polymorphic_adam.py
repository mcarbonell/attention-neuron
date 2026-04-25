import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time

class L2PolymorphicLayer(nn.Module):
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
        
        # Dial Polimórfico (SUM vs L2)
        # Inicializamos en 0.0 logit -> sigmoid(0) = 0.5 (50% Sum, 50% L2)
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
            
        # --- LÓGICA POLIMÓRFICA OPTIMIZADA (SUM vs L2 Norm) ---
        
        # 1. Agregación Lineal (SUM)
        y_sum = torch.matmul(x, w_evolved.t())
        
        # 2. Agregación Energética (L2 Norm aproximando el MAX)
        # sqrt( X^2 @ (W^2).T + epsilon )
        # Usamos abs() o elevamos al cuadrado. Elevar al cuadrado amplifica más los rasgos dominantes.
        x_sq = torch.square(x)
        w_sq = torch.square(w_evolved)
        y_l2 = torch.sqrt(torch.matmul(x_sq, w_sq.t()) + 1e-8)
        
        # 3. Mezclamos usando el dial Alpha por neurona
        alpha = torch.sigmoid(self.alpha_logits)
        
        # Mezcla final
        y_mixed = alpha * y_sum + (1.0 - alpha) * y_l2
            
        return y_mixed + torch.sin(self.theta_bias)

class L2PolymorphicMLP(nn.Module):
    def __init__(self, mask_prob=0.5):
        super().__init__()
        self.layer1 = L2PolymorphicLayer(784, 512, rank=2, mask_prob=mask_prob)
        self.layer2 = L2PolymorphicLayer(512, 10, rank=2, mask_prob=mask_prob)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking V14 (L2 Polymorphic Attention: Sum vs L2) with ADAM on: {device}")

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

    model = L2PolymorphicMLP(mask_prob=0.5).to(device)
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
    print("\n--- Análisis de la Identidad Neuronal (SUM vs L2) ---")
    for name, layer in [("Capa Oculta (512)", model.layer1), ("Capa Salida (10)", model.layer2)]:
        alphas = torch.sigmoid(layer.alpha_logits).detach().cpu().numpy()
        sum_neurons = sum(1 for a in alphas if a > 0.6)
        l2_neurons = sum(1 for a in alphas if a < 0.4)
        hybrid_neurons = len(alphas) - sum_neurons - l2_neurons
        
        print(f"{name}:")
        print(f"  - Preferencia SUM (alpha > 0.6): {sum_neurons} neuronas")
        print(f"  - Preferencia L2 (alpha < 0.4): {l2_neurons} neuronas")
        print(f"  - Híbridas (0.4 < alpha < 0.6): {hybrid_neurons} neuronas")

if __name__ == "__main__":
    main()
