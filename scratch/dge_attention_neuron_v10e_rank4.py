import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import math
import time
import json
import sys
import os
from pathlib import Path

# Path al repo de dge-optimizer
DGE_REPO_PATH = r"C:\Users\mrcm_\Local\proj\algorithms\dge-optimizer"
sys.path.append(DGE_REPO_PATH)

try:
    from dge.torch_optimizer import TorchDGEOptimizer
except ImportError:
    print("ERROR: No se encontró dge.torch_optimizer.")
    sys.exit(1)

class FlatStochasticRankLayer(nn.Module):
    def __init__(self, arch, rank=4, mask_prob=0.8): # RANK 4
        super().__init__()
        self.arch = arch
        self.rank = rank
        self.mask_prob = mask_prob
        self.layer_info = []
        total_params = 0
        for i, (in_f, out_f) in enumerate(zip(arch[:-1], arch[1:])):
            std = math.sqrt(2.0 / in_f)
            self.register_buffer(f'w_init_{i}', torch.randn(out_f, in_f) * std)
            # Parámetros: delta_in_m, delta_out_m, delta_in_a, delta_out_a, theta
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
    print(f"Entrenando V10e (Rank 4 + Stagnation Patience) en: {device}")

    ARCH = (784, 512, 10)
    RANK = 4
    INITIAL_BATCH_SIZE = 4
    MAX_BATCH_SIZE = 8192
    TEST_EVERY_STEPS = 20
    PATIENCE_LIMIT = 5 # Duplicar batch si no hay mejora en 5 mediciones
    LR = 0.005        
    MASK_PROB = 0.8
    MAX_STEPS = 2000

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    fast_test_loader = DataLoader(Subset(test_dataset, range(1000)), batch_size=512, shuffle=False)
    full_test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    model = FlatStochasticRankLayer(ARCH, rank=RANK, mask_prob=MASK_PROB).to(device)
    optimizer = TorchDGEOptimizer(
        dim=model.dim, layer_sizes=[sum(i['counts']) for i in model.layer_info], 
        k_blocks=[512, 128], lr=LR, delta=1e-3, total_steps=MAX_STEPS, device=device, chunk_size=128
    )

    current_batch_size = INITIAL_BATCH_SIZE
    train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True)
    train_iter = iter(train_loader)

    best_acc = 0.0
    patience_counter = 0
    params_vec = model.params.data.clone()
    criterion = nn.CrossEntropyLoss(reduction='none')
    t_start = time.time()
    
    for step in range(1, MAX_STEPS + 1):
        try:
            data, target = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data, target = next(train_iter)
        
        data, target = data.to(device), target.to(device)
        step_masks = [torch.bernoulli(torch.full((info['out'], info['in']), MASK_PROB, device=device)) for info in model.layer_info]

        def f_batched(p_batch):
            with torch.no_grad():
                logits = model.forward_with_params(data, p_batch, training=True, masks=step_masks)
                P, B, C = logits.shape
                loss = criterion(logits.view(P*B, C), target.repeat(P)).view(P, B).mean(dim=1)
            return loss

        params_vec, num_evals = optimizer.step(f_batched, params_vec)

        if step % TEST_EVERY_STEPS == 0:
            correct = 0
            with torch.no_grad():
                for d, t in fast_test_loader:
                    d, t = d.to(device), t.to(device)
                    out = model.forward_with_params(d, params_vec.unsqueeze(0), training=False).squeeze(0)
                    correct += out.argmax(1).eq(t).sum().item()
            acc = correct / 1000
            
            print(f"Step {step} | Batch: {current_batch_size} | Acc: {acc:.4f} | Best: {best_acc:.4f} | P: {patience_counter}")
            
            trigger = False
            if acc > best_acc:
                best_acc = acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE_LIMIT:
                    trigger = True
            
            if acc < best_acc * 0.99:
                trigger = True

            if trigger and current_batch_size < MAX_BATCH_SIZE:
                current_batch_size *= 2
                print(f" >>> HEURÍSTICA ACTIVADA (Regresión o Estancamiento). Nuevo Batch: {current_batch_size}")
                train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True)
                train_iter = iter(train_loader)
                patience_counter = 0

        if step % 200 == 0:
            correct = 0
            with torch.no_grad():
                for d, t in full_test_loader:
                    d, t = d.to(device), t.to(device)
                    out = model.forward_with_params(d, params_vec.unsqueeze(0), training=False).squeeze(0)
                    correct += out.argmax(1).eq(t).sum().item()
            print(f" --- STEP {step} FULL TEST ACC: {correct/10000:.4f} ---")

    print(f"\nFinalizado en {time.time() - t_start:.2f}s")

if __name__ == "__main__": main()
