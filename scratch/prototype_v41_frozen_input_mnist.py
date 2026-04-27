import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import json
import os
import math

class FrozenInputMLP(nn.Module):
    """
    V41: Frozen Input Projection Experiment.
    Architecture: 784 -> 2048 (FROZEN) -> 10 (TRAINABLE)
    The first layer is a random projection that acts as a fixed feature extractor.
    """
    def __init__(self, input_size=784, hidden_size=2048, output_size=10):
        super().__init__()
        # First layer: Random and Frozen
        self.frozen_layer = nn.Linear(input_size, hidden_size)
        # Initialize and freeze
        nn.init.kaiming_uniform_(self.frozen_layer.weight, a=math.sqrt(5))
        self.frozen_layer.weight.requires_grad = False
        self.frozen_layer.bias.requires_grad = False
        
        # Second layer: Trainable readout
        self.trainable_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        # Pass through frozen projection
        with torch.no_grad():
            x = self.frozen_layer(x)
            x = torch.relu(x)
        # Pass through trainable readout
        x = self.trainable_layer(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V41: FROZEN INPUT PROJECTION ---")
    print(f"Device: {device}")

    # Hyperparameters
    HIDDEN_SIZE = 2048
    BATCH_SIZE = 256
    EPOCHS = 20
    LR = 0.001
    SEED = 42
    
    torch.manual_seed(SEED)

    # Data Loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # Model initialization
    model = FrozenInputMLP(hidden_size=HIDDEN_SIZE).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    print(f"Frozen Parameters: {frozen_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Reduction Ratio: {frozen_params/trainable_params:.1f}x")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Metrics tracking
    metrics = {
        "final_objective": 0,
        "total_evaluations": 0,
        "wall_clock_time": 0,
        "function_evaluation_time": 0,
        "internal_overhead_time": 0,
        "history": []
    }

    t_start = time.time()
    total_evals = 0
    eval_time = 0

    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        epoch_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            t_eval_start = time.time()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            eval_time += (time.time() - t_eval_start)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            total_evals += len(data)
            
            if epoch == 1 and batch_idx < 5:
                print(f"  > Batch {batch_idx} | Loss: {loss.item():.4f}")

        # Evaluation
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        acc = correct / 10000
        t_epoch = time.time() - t0
        print(f"Epoch {epoch:2d} | Loss: {epoch_loss/len(train_loader):.4f} | Acc: {acc:.4f} | Time: {t_epoch:.1f}s")
        metrics["history"].append({"epoch": epoch, "acc": acc, "loss": epoch_loss/len(train_loader)})

    t_end = time.time()
    metrics["final_objective"] = acc
    metrics["total_evaluations"] = total_evals
    metrics["wall_clock_time"] = t_end - t_start
    metrics["function_evaluation_time"] = eval_time
    metrics["internal_overhead_time"] = (t_end - t_start) - eval_time

    # Save results
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v41_frozen_projection.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n🚀 Final Accuracy: {acc:.4f}")
    print(f"⏱️ Total Time: {t_end - t_start:.1f}s")
    print(f"📊 Trainable Params: {trainable_params}")

if __name__ == "__main__":
    main()
