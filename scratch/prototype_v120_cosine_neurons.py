import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import json
import os
import math

# --- Configuration ---
HIDDEN_SIZE = 64
BATCH_SIZE = 256
EPOCHS = 10
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model Variants ---

class CosineActivation(nn.Module):
    def forward(self, x):
        return torch.cos(x)

class SineActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class CosineReLUActivation(nn.Module):
    def forward(self, x):
        return torch.relu(torch.cos(x))

class VariantMLP(nn.Module):
    def __init__(self, hidden_size, activation_type='relu'):
        super().__init__()
        self.layer1 = nn.Linear(784, hidden_size)
        
        if activation_type == 'relu':
            self.activation = nn.ReLU()
        elif activation_type == 'cosine':
            self.activation = CosineActivation()
        elif activation_type == 'sine':
            self.activation = SineActivation()
        elif activation_type == 'cosine_relu':
            self.activation = CosineReLUActivation()
        else:
            raise ValueError("Unknown activation type")
            
        self.layer2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.activation(self.layer1(x))
        return self.layer2(x)

# --- Training and Evaluation ---

def train_variant(variant_name, activation_type):
    print(f"\n>>> Running Variant: {variant_name} ({activation_type})")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    model = VariantMLP(HIDDEN_SIZE, activation_type).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    metrics = {
        'variant': variant_name,
        'activation': activation_type,
        'total_evaluations': 0,
        'wall_clock_time': 0,
        'function_evaluation_time': 0,
        'internal_overhead_time': 0,
        'final_objective': 0,
        'test_acc': 0
    }

    start_wall = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            t0 = time.time()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            eval_time = time.time() - t0
            
            metrics['function_evaluation_time'] += eval_time
            metrics['total_evaluations'] += data.size(0)
            
            # RULE: Fast Feedback (first 5 batches)
            if epoch == 1 and batch_idx < 5:
                print(f"  [Fast Feedback] Epoch 1, Batch {batch_idx}: Loss = {loss.item():.4f}")

        # Evaluation
        model.eval()
        correct = 0
        total_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                total_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        acc = correct / len(test_dataset)
        avg_loss = total_loss / len(test_loader)
        print(f"  Epoch {epoch:2d} | Test Acc: {acc:.4f} | Loss: {avg_loss:.4f}")
        
    metrics['wall_clock_time'] = time.time() - start_wall
    metrics['internal_overhead_time'] = metrics['wall_clock_time'] - metrics['function_evaluation_time']
    metrics['final_objective'] = avg_loss
    metrics['test_acc'] = acc
    
    return metrics

def main():
    variants = [
        ('Baseline', 'relu'),
        ('PureCosine', 'cosine'),
        ('PureSine', 'sine'),
        ('CosineReLU', 'cosine_relu')
    ]
    
    all_results = []
    
    for name, act in variants:
        result = train_variant(name, act)
        all_results.append(result)
        
    # Save results
    os.makedirs('results/raw', exist_ok=True)
    save_path = 'results/raw/v120_cosine_experiment.json'
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nResults saved to {save_path}")
    
    # Summary Table
    print("\nSummary Results:")
    print(f"{'Variant':<15} | {'Acc':<8} | {'Loss':<8} | {'Eval Time':<10} | {'Overhead':<10}")
    print("-" * 65)
    for r in all_results:
        print(f"{r['variant']:<15} | {r['test_acc']:<8.4f} | {r['final_objective']:<8.4f} | {r['function_evaluation_time']:<10.2f} | {r['internal_overhead_time']:<10.2f}")

if __name__ == "__main__":
    main()
