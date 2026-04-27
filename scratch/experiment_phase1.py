import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time
import json
import os
import argparse
import numpy as np

# =============================================================================
# MODULAR ATTENTION NEURON LAYER
# =============================================================================
class AttentionNeuronLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=2, 
                 use_multiplicative=True, use_additive=True, use_phase_bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.use_multiplicative = use_multiplicative
        self.use_additive = use_additive
        self.use_phase_bias = use_phase_bias

        # Fixed base weights (Kaiming Normal)
        std = math.sqrt(2.0 / in_features)
        self.register_buffer('w_init', torch.randn(out_features, in_features) * std)

        # Multiplicative modulation
        if self.use_multiplicative:
            self.delta_in_m = nn.Parameter(torch.randn(out_features, rank) * 0.01)
            self.delta_out_m = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        
        # Additive modulation
        if self.use_additive:
            self.delta_in_a = nn.Parameter(torch.randn(out_features, rank) * 0.01)
            self.delta_out_a = nn.Parameter(torch.randn(rank, in_features) * 0.01)

        # Bias
        if self.use_phase_bias:
            self.theta_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # Start with the base substrate
        w_evolved = self.w_init.clone()

        # Apply multiplicative modulation: W = W_init * (1 + delta_m)
        if self.use_multiplicative:
            w_m = torch.matmul(self.delta_in_m, self.delta_out_m)
            w_evolved = w_evolved * (1.0 + w_m)

        # Apply additive modulation: W = W + delta_a
        if self.use_additive:
            w_a = torch.matmul(self.delta_in_a, self.delta_out_a)
            w_evolved = w_evolved + w_a

        # Compute output
        y = torch.matmul(x, w_evolved.t())

        # Apply bias
        if self.use_phase_bias:
            y = y + torch.sin(self.theta_bias)
        else:
            y = y + self.bias

        return y

class AttentionNeuronMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Architecture: 784 -> 512 -> 10
        self.layer1 = AttentionNeuronLayer(
            784, 512, 
            rank=config['rank'], 
            use_multiplicative=config['use_multiplicative'], 
            use_additive=config['use_additive'], 
            use_phase_bias=config['use_phase_bias']
        )
        self.layer2 = AttentionNeuronLayer(
            512, 10, 
            rank=config['rank'], 
            use_multiplicative=config['use_multiplicative'], 
            use_additive=config['use_additive'], 
            use_phase_bias=config['use_phase_bias']
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# =============================================================================
# TRAINING ENGINE
# =============================================================================
def train_model(config, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    model = AttentionNeuronMLP(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    best_acc = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for data, target in train_loader:
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
        
        acc = correct / len(test_loader.dataset)
        best_acc = max(best_acc, acc)

    return best_acc

# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, required=True, help='ID of the experiment (e.g., A1)')
    parser.add_argument('--rank', type=int, default=2)
    parser.add_argument('--use_multiplicative', action='store_true', default=True)
    parser.add_argument('--no_multiplicative', action='store_false', dest='use_multiplicative')
    parser.add_argument('--use_additive', action='store_true', default=True)
    parser.add_argument('--no_additive', action='store_false', dest='use_additive')
    parser.add_argument('--use_phase_bias', action='store_true', default=True)
    parser.add_argument('--no_phase_bias', action='store_false', dest='use_phase_bias')
    parser.add_argument('--seeds', type=int, default=1)
    args = parser.parse_args()

    config = {
        'rank': args.rank,
        'use_multiplicative': args.use_multiplicative,
        'use_additive': args.use_additive,
        'use_phase_bias': args.use_phase_bias
    }

    print(f"Running Experiment {args.exp_id} with config: {config} | Seeds: {args.seeds}")
    
    results = []
    for s in range(args.seeds):
        seed = 42 + s
        acc = train_model(config, seed=seed)
        results.append(acc)
        print(f"Seed {seed} | Acc: {acc:.4f}")

    final_result = {
        'exp_id': args.exp_id,
        'config': config,
        'results': results,
        'mean': np.mean(results),
        'std': np.std(results)
    }

    os.makedirs("results/phase1", exist_ok=True)
    with open(f"results/phase1/{args.exp_id}.json", "w") as f:
        json.dump(final_result, f, indent=4)

    print(f"Experiment {args.exp_id} finished. Mean Acc: {final_result['mean']:.4f}")

if __name__ == "__main__":
    main()
