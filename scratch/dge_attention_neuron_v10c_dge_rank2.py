import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time
import json
import sys
import os
from pathlib import Path

# Añadimos el repo de dge-optimizer al path
DGE_REPO_PATH = r"C:\Users\mrcm_\Local\proj\algorithms\dge-optimizer"
if DGE_REPO_PATH not in sys.path:
    sys.path.append(DGE_REPO_PATH)

try:
    from dge.torch_optimizer import TorchDGEOptimizer
except ImportError:
    print("ERROR: No se encontró dge.torch_optimizer.")
    sys.exit(1)

class FlatStochasticRank2MLP(nn.Module):
    def __init__(self, arch, rank=2, mask_prob=0.8):
        super().__init__()
        self.arch = arch
        self.rank = rank
        self.mask_prob = mask_prob
        self.layer_info = []
        total_params = 0
        
        for i, (in_f, out_f) in enumerate(zip(arch[:-1], arch[1:])):
            std = math.sqrt(2.0 / in_f)
            w_init = torch.randn(out_f, in_f) * std
            self.register_buffer(f'w_init_{i}', w_init)
            counts = [out_f * rank, rank * in_f, out_f * rank, rank * in_f, out_f]
            self.layer_info.append({'in': in_f, 'out': out_f, 'counts': counts, 'offset': total_params})
            total_params += sum(counts)
            
        self.dim = total_params
        self.params = nn.Parameter(torch.zeros(total_params))
        self.init_params()

    def init_params(self):
        with torch.no_grad():
            for info in self.layer_info:
                off = info['offset']
                c = info['counts']
                rank_scale = 1.0 / math.sqrt(self.rank)
                self.params[off : off + c[0]].fill_(rank_scale + 0.01)
                self.params[off + c[0] : off + c[0] + c[1]].fill_(rank_scale)
                self.params[off + c[0] + c[1] : off + sum(c)].fill_(0.0)

    def forward_with_params(self, x, p_vec, training=True, masks=None):
        P = p_vec.shape[0]
        if x.dim() > 2: x = x.view(x.size(0), -1)
        h = x.unsqueeze(0).expand(P, -1, -1) 
        
        for i, info in enumerate(self.layer_info):
            w_init = getattr(self, f'w_init_{i}')
            off = info['offset']
            c = info['counts']
            in_f, out_f = info['in'], info['out']
            
            d_in_m = p_vec[:, off : off + c[0]].view(P, out_f, self.rank)
            d_out_m = p_vec[:, off + c[0] : off + c[0] + c[1]].view(P, self.rank, in_f)
            d_in_a = p_vec[:, off + c[0] + c[1] : off + c[0] + c[1] + c[2]].view(P, out_f, self.rank)
            d_out_a = p_vec[:, off + c[0] + c[1] + c[2] : off + sum(c) - out_f].view(P, self.rank, in_f)
            theta = p_vec[:, off + sum(c) - out_f : off + sum(c)].view(P, 1, out_f)
            
            w_m = torch.bmm(d_in_m, d_out_m) 
            w_a = torch.bmm(d_in_a, d_out_a) 
            
            if training and self.mask_prob < 1.0:
                if masks is not None and i < len(masks): mask = masks[i]
                else: mask = torch.bernoulli(torch.full((out_f, in_f), self.mask_prob, device=x.device))
                w_evolved = torch.where(mask > 0, w_init * w_m + w_a, w_init)
            else:
                m_eff = 1.0 + self.mask_prob * (w_m - 1.0)
                a_eff = self.mask_prob * w_a
                w_evolved = w_init * m_eff + a_eff
            
            h = torch.bmm(h, w_evolved.transpose(1, 2)) + torch.sin(theta)
            if i < len(self.layer_info) - 1: h = torch.relu(h)
        return h

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entrenando V10c (Objetivo 90%) en: {device}")

    ARCH = (784, 512, 10)
    BATCH_SIZE = 4096 # Lote grande para máxima estabilidad
    EPOCHS = 20
    LR = 0.005        
    DELTA = 1e-3
    MASK_PROB = 0.8   # Suavizamos la máscara para facilitar la convergencia
    K_BLOCKS = [512, 128]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1024, shuffle=False)

    model = FlatStochasticRank2MLP(ARCH, mask_prob=MASK_PROB).to(device)
    layer_sizes = [sum(info['counts']) for info in model.layer_info]
    
    optimizer = TorchDGEOptimizer(
        dim=model.dim, layer_sizes=layer_sizes, k_blocks=K_BLOCKS,
        lr=LR, delta=DELTA, total_steps=len(train_loader) * EPOCHS,
        device=device, chunk_size=128
    )

    criterion = nn.CrossEntropyLoss(reduction='none')
    stats = {"final_objective": 0.0, "total_evaluations": 0, "wall_clock_time": 0.0, "function_evaluation_time": 0.0, "internal_overhead_time": 0.0}

    t_start = time.time()
    params_vec = model.params.data.clone()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_t0 = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            step_masks = []
            for info in model.layer_info:
                m = torch.bernoulli(torch.full((info['out'], info['in']), MASK_PROB, device=device))
                step_masks.append(m)

            def f_batched(p_batch):
                t_f_start = time.time()
                with torch.no_grad():
                    logits = model.forward_with_params(data, p_batch, training=True, masks=step_masks)
                    P, B, C = logits.shape
                    loss = criterion(logits.view(P*B, C), target.repeat(P)).view(P, B).mean(dim=1)
                stats["function_evaluation_time"] += (time.time() - t_f_start)
                return loss

            params_vec, num_evals = optimizer.step(f_batched, params_vec)
            stats["total_evaluations"] += num_evals
            
            if batch_idx % 5 == 0:
                with torch.no_grad(): l0 = f_batched(params_vec.unsqueeze(0)).item()
                print(f"Epoch {epoch} [{batch_idx*BATCH_SIZE}/60000] Loss: {l0:.4f}")

        # Evaluación
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model.forward_with_params(data, params_vec.unsqueeze(0), training=False).squeeze(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_acc = correct / 10000
        stats["final_objective"] = test_acc
        print(f"--- Epoch {epoch} | Test Acc: {test_acc:.4f} | Time: {time.time() - epoch_t0:.1f}s ---")

    stats["wall_clock_time"] = time.time() - t_start
    stats["internal_overhead_time"] = stats["wall_clock_time"] - stats["function_evaluation_time"]

    print("\n" + "="*40)
    print("MÉTRICAS FINALES (v10c)")
    print(f"Final Test Accuracy: {stats['final_objective']:.4f}")
    print(f"Total Evaluations: {stats['total_evaluations']}")
    print("="*40)

    res_path = Path("results/raw/v10c_dge_rank2_stochastic.json")
    res_path.parent.mkdir(parents=True, exist_ok=True)
    with open(res_path, "w") as f: json.dump(stats, f, indent=2)

if __name__ == "__main__": main()
