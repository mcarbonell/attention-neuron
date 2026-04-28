import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

def train_refined_archetypes_cosine():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- V77: Purifying Archetypes with Cosine Similarity ---")
    print(f"Device: {device}")
    
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=60000, shuffle=False)
    train_data, train_targets = next(iter(train_loader))
    train_data, train_targets = train_data.to(device), train_targets.to(device)
    flat_train = train_data.view(60000, 784)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=False)
    test_data, test_targets = next(iter(test_loader))
    test_data, test_targets = test_data.to(device), test_targets.to(device)
    flat_test = test_data.view(10000, 784)

    image_cluster_assignment = train_targets.clone()
    next_cluster_id = 10
    cluster_to_label = {d: d for d in range(10)}
    
    MAX_ITER = 200
    
    print("\nStarting purification process using COSINE SIMILARITY...")
    for iteration in range(MAX_ITER):
        active_clusters = torch.unique(image_cluster_assignment)
        arch_tensors = []
        arch_labels = []
        arch_cluster_ids = []
        
        for cid in active_clusters:
            cid = cid.item()
            mask = (image_cluster_assignment == cid)
            if mask.sum() > 0:
                arch_tensors.append(flat_train[mask].mean(dim=0))
                arch_labels.append(cluster_to_label[cid])
                arch_cluster_ids.append(cid)
                
        arch_tensors = torch.stack(arch_tensors) # (K, 784)
        arch_labels = torch.tensor(arch_labels, device=device) # (K,)
        arch_cluster_ids = torch.tensor(arch_cluster_ids, device=device)
        
        # --- COSINE SIMILARITY EVALUATION ---
        # 1. Normalize both vectors to unit length
        norm_train = F.normalize(flat_train, p=2, dim=1)
        norm_arch = F.normalize(arch_tensors, p=2, dim=1)
        
        # 2. Dot product of normalized vectors = Cosine Similarity
        # Shape: (60000, 784) x (784, K) -> (60000, K)
        cos_sim = torch.mm(norm_train, norm_arch.t())
        
        # 3. We want to MAXIMIZE similarity
        best_arch_idx = torch.argmax(cos_sim, dim=1)
        pred_labels = arch_labels[best_arch_idx]
        
        correct = (pred_labels == train_targets)
        acc = correct.float().mean().item()
        print(f"Gen {iteration:2d} | Active Archetypes: {len(active_clusters):3d} | Train Acc: {acc*100:.2f}%")
        
        if acc >= 0.99 or iteration == MAX_ITER - 1: 
            break
            
        # Purify
        image_cluster_assignment[correct] = arch_cluster_ids[best_arch_idx[correct]]
        
        # Isolate Errors
        for digit in range(10):
            mask_err = (~correct) & (train_targets == digit)
            if mask_err.sum() > 0:
                image_cluster_assignment[mask_err] = next_cluster_id
                cluster_to_label[next_cluster_id] = digit
                next_cluster_id += 1

    # --- EVALUATE ON TEST SET ---
    norm_test = F.normalize(flat_test, p=2, dim=1)
    norm_arch = F.normalize(arch_tensors, p=2, dim=1)
    cos_sim_test = torch.mm(norm_test, norm_arch.t())
    pred_test_indices = torch.argmax(cos_sim_test, dim=1)
    preds_test = arch_labels[pred_test_indices]
    
    test_acc = (preds_test == test_targets).float().mean().item()
    print(f"\n=======================================")
    print(f"FINAL TEST ACCURACY (COSINE): {test_acc*100:.2f}%")
    print(f"=======================================")


    # --- VISUALIZATION ---
    print("\nGenerating purified taxonomy grid...")
    archs_by_digit = {i: [] for i in range(10)}
    for tensor, label in zip(arch_tensors, arch_labels):
        archs_by_digit[label.item()].append(tensor)
        
    # Sort them by "density" or just keep the original first
    max_archs = max(len(lst) for lst in archs_by_digit.values())
    # Cap max_archs for plotting if it gets too large
    plot_cols = min(max_archs, 15) 
    
    fig, axes = plt.subplots(10, plot_cols, figsize=(plot_cols*1.5, 15))
    fig.suptitle(f"V77 Purified Archetypes COSINE SIMILARITY (Test Acc: {test_acc*100:.2f}%)", fontsize=18)
    
    for digit in range(10):
        lst = archs_by_digit[digit]
        for col in range(plot_cols):
            ax = axes[digit, col]
            if col < len(lst):
                tensor = lst[col]
                ax.imshow(tensor.cpu().view(28, 28).numpy(), cmap='magma')
                if col == 0:
                    ax.set_title(f"Base '{digit}'", color='red')
            ax.axis('off')
            
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    save_path = "results/figures/v77_refined_archetypes_cosine.png"
    plt.savefig(save_path)
    print(f"Purified Taxonomy saved to: {save_path}")    

if __name__ == "__main__":
    train_refined_archetypes_cosine()
