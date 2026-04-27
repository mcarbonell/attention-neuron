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
# LAYERS IMPLEMENTATIONS
# =============================================================================

class StandardLinearLayer(nn.Module):
    """Standard Dense Linear Layer."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

class FrozenBiasLayer(nn.Module):
    """Frozen weights, only trainable bias."""
    def __init__(self, in_features, out_features):
        super().__init__()
        std = math.sqrt(2.0 / in_features)
        w_init = torch.randn(out_features, in_features) * std
        self.register_buffer('w_init', w_init)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return torch.matmul(x, self.w_init.t()) + self.bias

class LowRankAdditiveLayer(nn.Module):
    """Frozen weights + Low-Rank Additive (LoRA-like). W = W_init + (A @ B)"""
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        std = math.sqrt(2.0 / in_features)
        w_init = torch.randn(out_features, in_features) * std
        self.register_buffer('w_init', w_init)
        
        self.rank = rank
        self.A = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        w_evolved = self.w_init + torch.matmul(self.A, self.B)
        return torch.matmul(x, w_evolved.t()) + self.bias

class AttentionNeuronLayer(nn.Module):
    """The 'Golden Baseline' Attention Neuron: Rank-k, Mult, Add, Phase."""
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        std = math.sqrt(2.0 / in_features)
        self.register_buffer('w_init', torch.randn(out_features, in_features) * std)
        
        self.rank = rank
        self.delta_in_m = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.delta_in_a = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.delta_out_a = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.theta_bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        w_m = torch.matmul(self.delta_in_m, self.delta_out_m)
        w_a = torch.matmul(self.delta_in_a, self.delta_out_a)
        w_evolved = self.w_init * (1.0 + w_m) + w_a
        return torch.matmul(x, w_evolved.t()) + torch.sin(self.theta_bias)

# =============================================================================
# MODEL WRAPPER
# =============================================================================
class ComparisonMLP(nn.Module):
    def __init__(self, layer_type, config):
        super().__init__()
        self.layer_type = layer_type
        
        if layer_type == 'dense':
            self.layer1 = StandardLinearLayer(784, 512)
            self.layer2 = StandardLinearLayer(512, 10)
        elif layer_type == 'frozen_bias':
            self.layer1 = FrozenBiasLayer(784, 512)
            self.layer2 = FrozenBiasLayer(512, 10)
        elif layer_type == 'low_rank_add':
            self.layer1 = LowRankAdditiveLayer(784, 512, rank=config['rank'])
            self.layer2 = LowRankAdditiveLayer(512, 10, rank=config['rank'])
        elif layer_type == 'attention_neuron':
            self.layer1 = AttentionNeuronLayer(784, 512, rank=config['rank'])
            self.layer2 = AttentionNeuronLayer(512, 10, rank=config['rank'])
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# =============================================================================
# TRAINING ENGINE
# =============================================================================
def train_model(layer_type, config, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    model = ComparisonMLP(layer_type, config).to(device)
    
    # Count trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
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

    return best_acc, trainable_params

# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, required=True)
    parser.add_argument('--layer_type', type=str, required=True, choices=['dense', 'frozen_bias', 'low_rank_add', 'attention_neuron'])
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--seeds', type=int, default=1)
    args = parser.parse_args()

    config = {'rank': args.rank}
    print(f"Running Experiment {args.exp_id} | Layer: {args.layer_type} | Rank: {args.rank} | Seeds: {args.seeds}")
    
    results = []
    params_list = []
    for s in range(args.seeds):
        seed = 42 + s
        acc, params = train_model(args.layer_type, config, seed=seed)
        results.append(acc)
        params_list.append(params)
        print(f"Seed {seed} | Acc: {acc:.4f} | Params: {params:,}")

    final_result = {
        'exp_id': args.exp_id,
        'layer_type': args.layer_type,
        'config': config,
        'results': results,
        'mean_acc': np.mean(results),
        'std_acc': np.std(results),
        'trainable_params': int(np.mean(params_list))
    }

    os.makedirs("results/phase2", exist_ok=True)
    with open(f"results/phase2/{args.exp_id}.json", "w") as f:
        json.dump(final_result, f, indent=4)

    print(f"Experiment {args.exp_id} finished. Mean Acc: {final_result['mean_acc']:.4f} | Params: {final_result['trainable_params']:,}")

if __name__ == "__main__":
    main()
