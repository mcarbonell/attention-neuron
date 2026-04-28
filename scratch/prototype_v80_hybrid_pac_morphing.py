import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import time

class VectorCanvas(nn.Module):
    """
    Parametric Canvas that draws 15 Bezier curves.
    """
    def __init__(self, num_strokes=15, device='cpu'):
        super().__init__()
        self.num_strokes = num_strokes
        
        # 3 points per stroke (x, y)
        self.points = nn.Parameter(torch.rand(num_strokes, 3, 2) * 14.0 + 7.0)
        self.weights = nn.Parameter(torch.ones(num_strokes, 1))
        self.sigma = 1.2
        
        y, x = torch.meshgrid(torch.linspace(0, 27, 28), torch.linspace(0, 27, 28), indexing='ij')
        self.register_buffer("grid", torch.stack([x, y], dim=-1).view(1, 784, 2).to(device))

    def forward(self):
        t = torch.linspace(0, 1, 15, device=self.points.device).view(1, 15, 1)
        p0 = self.points[:, 0:1, :]
        p1 = self.points[:, 1:2, :]
        p2 = self.points[:, 2:3, :]
        bezier_points = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2 # (N, 15, 2)
        
        diff = self.grid.unsqueeze(0) - bezier_points.unsqueeze(2) # (N, 15, 784, 2)
        dist_sq = torch.sum(diff**2, dim=-1) # (N, 15, 784)
        min_dist_sq, _ = torch.min(dist_sq, dim=1) # (N, 784)
        
        strokes = torch.exp(-min_dist_sq / (2 * self.sigma**2)) # (N, 784)
        canvas = torch.sum(strokes * torch.relu(self.weights), dim=0) # (784,)
        
        return torch.clamp(canvas, 0, 1).view(28, 28)

def run_pac_clustering(dataset, device, max_iter=15):
    print("--- Phase 1A: PAC Clustering (Fast Dictionary Generation) ---")
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=60000, shuffle=False)
    data, targets = next(iter(loader))
    data, targets = data.to(device), targets.to(device)
    flat_data = data.view(60000, 784)
    
    image_cluster_assignment = targets.clone()
    next_cluster_id = 10
    cluster_to_label = {d: d for d in range(10)}
    
    for iteration in range(max_iter):
        active_clusters = torch.unique(image_cluster_assignment)
        arch_tensors = []
        arch_labels = []
        arch_cluster_ids = []
        
        for cid in active_clusters:
            cid = cid.item()
            mask = (image_cluster_assignment == cid)
            if mask.sum() > 0:
                arch_tensors.append(flat_data[mask].mean(dim=0))
                arch_labels.append(cluster_to_label[cid])
                arch_cluster_ids.append(cid)
                
        arch_tensors = torch.stack(arch_tensors)
        arch_labels = torch.tensor(arch_labels, device=device)
        arch_cluster_ids = torch.tensor(arch_cluster_ids, device=device)
        
        dist = torch.cdist(flat_data, arch_tensors, p=2)
        best_arch_idx = torch.argmin(dist, dim=1)
        pred_labels = arch_labels[best_arch_idx]
        
        correct = (pred_labels == targets)
        acc = correct.float().mean().item()
        print(f"Gen {iteration} | Active Archetypes: {len(active_clusters):3d} | Train Acc: {acc*100:.2f}%")
        
        if iteration == max_iter - 1:
            break
            
        image_cluster_assignment[correct] = arch_cluster_ids[best_arch_idx[correct]]
        
        for digit in range(10):
            mask_err = (~correct) & (targets == digit)
            if mask_err.sum() > 0:
                image_cluster_assignment[mask_err] = next_cluster_id
                cluster_to_label[next_cluster_id] = digit
                next_cluster_id += 1
                
    return arch_tensors, arch_labels

def vectorize_archetypes(arch_tensors, arch_labels, device, epochs=150):
    print(f"\n--- Phase 1B: Vectorizing {len(arch_tensors)} Archetypes ---")
    vector_archetypes = []
    
    for i, (tensor, label) in enumerate(zip(arch_tensors, arch_labels)):
        target_img = tensor.view(28, 28)
        model = VectorCanvas(num_strokes=15, device=device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.5)
        criterion = nn.MSELoss()
        
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            generated = model()
            loss = criterion(generated, target_img)
            loss.backward()
            optimizer.step()
            
        vector_archetypes.append({
            'label': label.item(),
            'pixel_tensor': tensor.clone(),
            'points': model.points.detach().clone(),
            'weights': model.weights.detach().clone()
        })
        if (i+1) % 10 == 0 or (i+1) == len(arch_tensors):
            print(f"Vectorized {i+1}/{len(arch_tensors)} archetypes...")
            
    return vector_archetypes

def hybrid_inference(test_loader, vector_archetypes, device, lambda_elastic=0.5, steps=30, top_k=5, num_test=50):
    print(f"\n--- Phase 2: Hybrid Active Inference (Testing on {num_test} images) ---")
    correct = 0
    total = 0
    
    # Pre-stack pixel tensors for fast L2 calculation
    arch_pixels = torch.stack([v['pixel_tensor'] for v in vector_archetypes]).to(device) # (K, 784)
    
    results_to_plot = []
    
    for img, label in test_loader:
        img = img.squeeze().to(device)
        actual_label = label.item()
        
        # --- FAST PASS (SYSTEM 1) ---
        flat_img = img.view(1, 784)
        dist = torch.cdist(flat_img, arch_pixels, p=2).squeeze(0) # (K,)
        # Get top-K closest archetypes
        # We need to make sure K doesn't exceed the total number of archetypes
        k_to_use = min(top_k, len(vector_archetypes))
        top_indices = torch.topk(dist, k_to_use, largest=False).indices
        
        # --- SLOW PASS (SYSTEM 2) ---
        best_loss = float('inf')
        predicted_class = -1
        morphed_canvases = []
        top_labels = []
        
        for idx in top_indices:
            arch = vector_archetypes[idx.item()]
            top_labels.append(arch['label'])
            
            model = VectorCanvas(num_strokes=15, device=device).to(device)
            model.points.data = arch['points'].clone()
            model.weights.data = arch['weights'].clone()
            
            base_points = arch['points'].clone()
            
            optimizer = optim.Adam(model.parameters(), lr=0.2)
            criterion_mse = nn.MSELoss()
            
            final_loss_val = 0
            for step in range(steps):
                optimizer.zero_grad()
                generated = model()
                mse_loss = criterion_mse(generated, img)
                elastic_loss = torch.mean((model.points - base_points)**2)
                
                # HIGH ELASTIC PENALTY because the archetype should already be very close
                loss = mse_loss + lambda_elastic * elastic_loss
                
                loss.backward()
                optimizer.step()
                final_loss_val = loss.item()
                
            morphed_canvases.append(model().detach().cpu().numpy())
            
            if final_loss_val < best_loss:
                best_loss = final_loss_val
                predicted_class = arch['label']
                
        if predicted_class == actual_label:
            correct += 1
        total += 1
        
        if total <= 5:
            results_to_plot.append((img.cpu().numpy(), actual_label, predicted_class, top_labels, morphed_canvases))
            
        print(f"Test {total}/{num_test} | True: {actual_label} | Pred: {predicted_class} | Correct: {predicted_class == actual_label}")
        if total >= num_test:
            break
            
    accuracy = 100 * correct / total
    print(f"\nFinal Accuracy on {num_test} tests: {accuracy:.2f}%")
    return results_to_plot

def plot_results(results_to_plot, top_k=5):
    num_examples = len(results_to_plot)
    fig, axes = plt.subplots(num_examples, top_k + 2, figsize=(2 * (top_k + 2), 2.5 * num_examples))
    fig.suptitle("V80 Hybrid Classifier: Target vs Top-5 Morphed Sub-Archetypes", fontsize=16)
    
    for i, (target_img, actual_label, pred_label, top_labels, morphed_canvases) in enumerate(results_to_plot):
        axes[i, 0].imshow(target_img, cmap='gray')
        axes[i, 0].set_title(f"Target: {actual_label}")
        axes[i, 0].axis('off')
        
        # Blank separator
        axes[i, 1].axis('off')
        
        for k in range(min(top_k, len(top_labels))):
            ax = axes[i, k+2]
            ax.imshow(morphed_canvases[k], cmap='magma')
            
            l = top_labels[k]
            color = 'black'
            weight = 'normal'
            if l == pred_label and l == actual_label:
                color = 'green'
                weight = 'bold'
            elif l == pred_label and l != actual_label:
                color = 'red'
                weight = 'bold'
                
            ax.set_title(f"Morph '{l}'", color=color, fontweight=weight)
            ax.axis('off')
            
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    save_path = "results/figures/v80_hybrid_inference.png"
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    start_time = time.time()
    # Phase 1A: Get ~50 pure archetypes using 15 iterations of PAC
    arch_tensors, arch_labels = run_pac_clustering(train_dataset, device, max_iter=200)
    print(f"Phase 1A Time: {time.time() - start_time:.2f}s")
    
    start_time = time.time()
    # Phase 1B: Vectorize them (150 epochs each)
    vector_archetypes = vectorize_archetypes(arch_tensors, arch_labels, device, epochs=150)
    print(f"Phase 1B Time: {time.time() - start_time:.2f}s")
    
    start_time = time.time()
    # Phase 2: Hybrid inference with strong elastic penalty (0.5)
    results_to_plot = hybrid_inference(test_loader, vector_archetypes, device, lambda_elastic=0.5, steps=30, top_k=5, num_test=50)
    print(f"Phase 2 Time: {time.time() - start_time:.2f}s")
    
    plot_results(results_to_plot, top_k=5)

if __name__ == "__main__":
    main()
