import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import math

# --- Fast Walsh-Hadamard Transform (Vectorized) ---
def fwht(x):
    """
    Computes the Fast Walsh-Hadamard Transform of a batch of vectors.
    Input x: (B, N) where N must be a power of 2.
    """
    B, N = x.shape
    h = 1
    while h < N:
        x = x.view(B, N // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = torch.stack([a + b, a - b], dim=2)
        h *= 2
    return x.view(B, N)

# --- Cerebelo Espectral (Sistema 1 - Rápido) ---
class SpectralCerebellum(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # Para MNIST, pad 784 to 1024 for FWHT power of 2
        self.pad_dim = 1024 
        
        # Filtro Espectral (Solo 1024 parámetros)
        self.spectral_filter = nn.Parameter(torch.ones(self.pad_dim))
        
        # Clasificador ultra-rápido (1024 x 10)
        self.classifier = nn.Linear(self.pad_dim, num_classes)

    def forward(self, x):
        # x is (B, 784). Pad to 1024
        B = x.shape[0]
        x_pad = F.pad(x, (0, self.pad_dim - 784))
        
        # 1. Transformada al dominio de frecuencias (O(N log N))
        f_x = fwht(x_pad)
        
        # 2. Modulación Espectral (Eq) - O(N)
        f_mod = f_x * self.spectral_filter
        
        # 3. Clasificación Directa (Cortocircuito)
        logits = self.classifier(f_mod)
        return logits

# --- Córtex Profundo (Sistema 2 - Lento) ---
class DeepCortex(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Heavy computation (O(N^2) matricial multiplications)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        logits = self.classifier(x)
        return logits

# --- Red Dual (Cognición Completa) ---
class DualRoutingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cerebellum = SpectralCerebellum(784, 10)
        self.cortex = DeepCortex(784, 512, 10)
        self.entropy_threshold = 0.5 # Umbral de duda

    def calculate_entropy(self, logits):
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1)
        return entropy

    def forward_train(self, x):
        # Durante el entrenamiento, calculamos ambas vías
        logits_fast = self.cerebellum(x)
        logits_slow = self.cortex(x)
        return logits_fast, logits_slow

    def forward_infer(self, x):
        # Durante la inferencia (Batch size = 1 for dynamic routing simulation)
        logits_fast = self.cerebellum(x)
        entropy = self.calculate_entropy(logits_fast)
        
        if entropy.item() < self.entropy_threshold:
            # EARLY EXIT: El Cerebelo está seguro. Abortar Córtex.
            return logits_fast, "cerebellum", entropy.item()
        else:
            # FALLBACK: El Cerebelo duda. Enrutar al Córtex.
            logits_slow = self.cortex(x)
            return logits_slow, "cortex", entropy.item()

# --- Entrenamiento y Evaluación ---
def train_and_evaluate():
    print("=== Experimento V89: Cerebelo Espectral y Early-Exit ===\n")
    
    # --- Datos ---
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # BS=1 for dynamic routing

    model = DualRoutingNetwork()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 3
    print(f"Entrenando Red Dual por {epochs} épocas...")
    
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            
            logits_fast, logits_slow = model.forward_train(data)
            
            # Loss conjunto: Queremos que ambos sean precisos independientemente
            loss_fast = F.cross_entropy(logits_fast, target)
            loss_slow = F.cross_entropy(logits_slow, target)
            loss = loss_fast + loss_slow
            
            loss.backward()
            optimizer.step()
            
            if batch_idx == 4 and epoch == 0:
                print(f"[Epoch {epoch+1}] Fast Feedback: Loss Fast={loss_fast.item():.4f}, Loss Slow={loss_slow.item():.4f}")
        
        print(f"Epoch {epoch+1} completada en {time.time()-start_time:.2f}s")

    # --- Evaluación (Inferencia Dinámica) ---
    print("\nIniciando Evaluación de Inferencia Dinámica (Early-Exit)...")
    model.eval()
    
    correct_total = 0
    cerebellum_exits = 0
    cortex_fallbacks = 0
    correct_cerebellum = 0
    correct_cortex = 0
    
    time_cerebellum_total = 0
    time_cortex_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1)
            
            # Medimos el tiempo del Cerebelo puro
            t0 = time.perf_counter()
            logits_fast = model.cerebellum(data)
            t_fast = time.perf_counter() - t0
            
            entropy = model.calculate_entropy(logits_fast).item()
            
            if entropy < model.entropy_threshold:
                # EARLY EXIT
                cerebellum_exits += 1
                time_cerebellum_total += t_fast
                pred = logits_fast.argmax(dim=1, keepdim=True)
                if pred.eq(target.view_as(pred)).item():
                    correct_total += 1
                    correct_cerebellum += 1
            else:
                # CORTEX FALLBACK
                cortex_fallbacks += 1
                # Medimos el tiempo del Córtex puro
                t0_c = time.perf_counter()
                logits_slow = model.cortex(data)
                t_slow = time.perf_counter() - t0_c
                
                time_cortex_total += (t_fast + t_slow) # El tiempo total incluye la duda inicial
                
                pred = logits_slow.argmax(dim=1, keepdim=True)
                if pred.eq(target.view_as(pred)).item():
                    correct_total += 1
                    correct_cortex += 1

    total_samples = len(test_loader.dataset)
    accuracy = 100. * correct_total / total_samples
    exit_rate = 100. * cerebellum_exits / total_samples
    
    print(f"\n--- Resultados Finales de Enrutamiento V89 ---")
    print(f"Accuracy Global: {accuracy:.2f}%")
    print(f"Resolución Rápida (Cerebelo): {cerebellum_exits} muestras ({exit_rate:.1f}%)")
    print(f"Resolución Lenta (Córtex):    {cortex_fallbacks} muestras ({100-exit_rate:.1f}%)")
    
    if cerebellum_exits > 0:
        acc_cer = 100. * correct_cerebellum / cerebellum_exits
        avg_time_cer = time_cerebellum_total / cerebellum_exits * 1000
        print(f"  -> Precisión del Cerebelo: {acc_cer:.2f}% (Tiempo prom: {avg_time_cer:.3f} ms)")
        
    if cortex_fallbacks > 0:
        acc_cor = 100. * correct_cortex / cortex_fallbacks
        avg_time_cor = time_cortex_total / cortex_fallbacks * 1000
        print(f"  -> Precisión del Córtex:   {acc_cor:.2f}% (Tiempo prom: {avg_time_cor:.3f} ms)")

    print("\nConclusión de Eficiencia:")
    if cerebellum_exits > 0 and cortex_fallbacks > 0:
        speedup = avg_time_cor / avg_time_cer
        print(f"¡El Cerebelo es {speedup:.1f}x más rápido que el Córtex!")
        print(f"Al delegar el {exit_rate:.1f}% del trabajo al Cerebelo, hemos ahorrado un cálculo inmenso.")

if __name__ == '__main__':
    train_and_evaluate()
