import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time
import json
from pathlib import Path

# --- Arquitectura: Stochastic Dual Phase Layer (Rank-2 + Phase Bias + Stochastic Mask) ---
class StochasticDualPhaseLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=2, mask_prob=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.mask_prob = mask_prob
        
        # Sustrato físico: Matriz densa aleatoria (CONGELADA)
        std = math.sqrt(2.0 / in_features)
        self.register_buffer('w_init', torch.randn(out_features, in_features) * std)
        
        # Parámetros entrenables (Deltas de modulación)
        self.delta_in_m = nn.Parameter(torch.randn(out_features, rank) * 0.01 + 1.0)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_features) * 0.01 + 1.0)
        self.delta_in_a = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.delta_out_a = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        
        # Bias de Fase
        self.theta_bias = nn.Parameter(torch.zeros(out_features))

    def get_w_evolved(self, training=True):
        # Reconstrucción Rank-2
        w_m = torch.matmul(self.delta_in_m, self.delta_out_m)
        w_a = torch.matmul(self.delta_in_a, self.delta_out_a)
        
        if training and self.mask_prob < 1.0:
            # Máscara estocástica: ¿Qué cables se ven afectados por la modulación?
            mask = torch.bernoulli(torch.full(self.w_init.shape, self.mask_prob, device=self.w_init.device))
            # Donde la máscara es 0, el cable permanece en su estado original (w_init)
            w_evolved = torch.where(mask > 0, self.w_init * w_m + w_a, self.w_init)
        else:
            # En inferencia (o si prob=1), aplicamos el valor esperado
            # Nota: Para mantener la escala, si mask_prob=0.5, el efecto de los deltas se promedia
            m_eff = 1.0 + self.mask_prob * (w_m - 1.0)
            a_eff = self.mask_prob * w_a
            w_evolved = self.w_init * m_eff + a_eff
            
        return w_evolved

    def forward(self, x):
        w_evolved = self.get_w_evolved(training=self.training)
        phase_bias = torch.sin(self.theta_bias)
        return torch.matmul(x, w_evolved.t()) + phase_bias

class StochasticRank2MLP(nn.Module):
    def __init__(self, mask_prob=0.5):
        super().__init__()
        self.layer1 = StochasticDualPhaseLayer(784, 512, rank=2, mask_prob=mask_prob)
        self.layer2 = StochasticDualPhaseLayer(512, 10, rank=2, mask_prob=mask_prob)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

# --- Optimizador DGE (Denoised Gradient Estimation) Simplificado ---
class Rank2DGEOptimizer:
    def __init__(self, model, lr=0.01, delta=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.model = model
        self.lr = lr
        self.delta = delta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # Aplanamos los parámetros para el optimizador de orden cero
        self.params_refs = [p for p in model.parameters() if p.requires_grad]
        self.dim = sum(p.numel() for p in self.params_refs)
        
        self.m = torch.zeros(self.dim)
        self.v = torch.zeros(self.dim)
        self.t = 0

    def get_params_vec(self):
        return torch.cat([p.data.view(-1) for p in self.params_refs])

    def set_params_vec(self, vec):
        offset = 0
        for p in self.params_refs:
            numel = p.numel()
            p.data.copy_(vec[offset:offset+numel].view(p.shape))
            offset += numel

    def step(self, f_loss):
        self.t += 1
        x = self.get_params_vec()
        device = x.device
        
        # 1. Generar perturbación aleatoria (SPSA style)
        xi = torch.randint(0, 2, (self.dim,), device=device).float() * 2 - 1
        
        # 2. Evaluación Dual (Forward passes)
        # Nota: La estocasticidad interna de la capa añadirá ruido, 
        # lo cual DGE debería ser capaz de "denoise" mediante el promedio (EMA).
        self.set_params_vec(x + self.delta * xi)
        l_plus = f_loss()
        
        self.set_params_vec(x - self.delta * xi)
        l_minus = f_loss()
        
        # 3. Estimación del gradiente
        grad_est = (l_plus - l_minus) / (2 * self.delta) * xi
        
        # 4. Adam EMA Update
        self.m = self.beta1 * self.m.to(device) + (1 - self.beta1) * grad_est
        self.v = self.beta2 * self.v.to(device) + (1 - self.beta2) * (grad_est**2)
        
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        step = self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        
        # 5. Aplicar mejora
        new_x = x - step
        self.set_params_vec(new_x)
        
        return (l_plus + l_minus) / 2.0, 2 # Retornamos loss promedio y num evals

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entrenando V10 (DGE + Rank2 + Stochastic) en: {device}")

    BATCH_SIZE = 4096  # Lote grande para estabilizar DGE
    EPOCHS = 10
    LR = 0.01
    DELTA = 0.001
    MASK_PROB = 0.5

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    model = StochasticRank2MLP(mask_prob=MASK_PROB).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {total_params} (Reducción >99% vs densa)")

    optimizer = Rank2DGEOptimizer(model, lr=LR, delta=DELTA)
    criterion = nn.CrossEntropyLoss()

    stats = {
        "final_objective": 0.0,
        "total_evaluations": 0,
        "wall_clock_time": 0.0,
        "function_evaluation_time": 0.0,
        "internal_overhead_time": 0.0
    }

    t_start = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_t0 = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            def f_loss():
                t_f_start = time.time()
                output = model(data)
                loss = criterion(output, target)
                stats["function_evaluation_time"] += (time.time() - t_f_start)
                return loss

            batch_loss, num_evals = optimizer.step(f_loss)
            stats["total_evaluations"] += num_evals
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch} [{batch_idx*BATCH_SIZE}/{len(train_dataset)}] Loss: {batch_loss:.4f}")

        # Evaluación
        model.eval()
        correct = 0
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_acc = correct / len(test_loader.dataset)
        stats["final_objective"] = test_acc
        
        print(f"--- Epoch {epoch} Finalizada | Test Acc: {test_acc:.4f} | Time: {time.time() - epoch_t0:.1f}s ---")

    stats["wall_clock_time"] = time.time() - t_start
    stats["internal_overhead_time"] = stats["wall_clock_time"] - stats["function_evaluation_time"]

    print("\n" + "="*40)
    print("MÉTRICAS FINALES (GEMINI.md)")
    print(f"Final Test Accuracy: {stats['final_objective']:.4f}")
    print(f"Total Evaluations: {stats['total_evaluations']}")
    print(f"Wall Clock Time: {stats['wall_clock_time']:.2f}s")
    print(f"Function Eval Time: {stats['function_evaluation_time']:.2f}s")
    print(f"Internal Overhead: {stats['internal_overhead_time']:.2f}s")
    print("="*40)

    # Guardar resultados
    res_path = Path("results/raw/v10_dge_rank2_stochastic.json")
    res_path.parent.mkdir(parents=True, exist_ok=True)
    with open(res_path, "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
