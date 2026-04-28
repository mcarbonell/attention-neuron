import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

def train_pac_bifurcation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- V78: PAC with Error Bifurcation (Overfitting Quest) ---")
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
    
    MAX_ITER = 200 # Reduced from 200 since it should spawn clusters twice as fast
    
    print("\nStarting purification process with ERROR BIFURCATION...")
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
                
        arch_tensors = torch.stack(arch_tensors)
        arch_labels = torch.tensor(arch_labels, device=device)
        arch_cluster_ids = torch.tensor(arch_cluster_ids, device=device)
        
        # --- COSINE SIMILARITY ---
        norm_train = F.normalize(flat_train, p=2, dim=1)
        norm_arch = F.normalize(arch_tensors, p=2, dim=1)
        
        cos_sim = torch.mm(norm_train, norm_arch.t())
        max_sim, best_arch_idx = torch.max(cos_sim, dim=1)
        pred_labels = arch_labels[best_arch_idx]
        
        correct = (pred_labels == train_targets)
        acc = correct.float().mean().item()
        print(f"Gen {iteration:3d} | Active Archetypes: {len(active_clusters):4d} | Train Acc: {acc*100:.2f}%")
        
        if acc >= 0.999 or iteration == MAX_ITER - 1: 
            break
            
        # Purify
        image_cluster_assignment[correct] = arch_cluster_ids[best_arch_idx[correct]]
        
        # BIFURCATE ERRORS
        for digit in range(10):
            mask_err = (~correct) & (train_targets == digit)
            misclassified_indices = torch.nonzero(mask_err).squeeze(1)
            
            num_errors = len(misclassified_indices)
            if num_errors > 0:
                if num_errors == 1:
                    # Only one error, assign to a new cluster
                    image_cluster_assignment[misclassified_indices] = next_cluster_id
                    cluster_to_label[next_cluster_id] = digit
                    next_cluster_id += 1
                else:
                    # Sort errors by how "confidently wrong" they were (max_sim)
                    sims_for_errors = max_sim[misclassified_indices]
                    sorted_args = torch.argsort(sims_for_errors)
                    
                    half = len(sorted_args) // 2
                    
                    # Split 1: The "less confident" errors
                    half1_indices = misclassified_indices[sorted_args[:half]]
                    image_cluster_assignment[half1_indices] = next_cluster_id
                    cluster_to_label[next_cluster_id] = digit
                    next_cluster_id += 1
                    
                    # Split 2: The "more confident" errors
                    half2_indices = misclassified_indices[sorted_args[half:]]
                    image_cluster_assignment[half2_indices] = next_cluster_id
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
    print(f"FINAL TEST ACCURACY (BIFURCATION): {test_acc*100:.2f}%")
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
    fig.suptitle(f"V78 Purified Archetypes BIFURCATION (Test Acc: {test_acc*100:.2f}%)", fontsize=18)
    
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
    save_path = "results/figures/v78_refined_archetypes_bifurcation.png"
    plt.savefig(save_path)
    print(f"Purified Taxonomy saved to: {save_path}")    

if __name__ == "__main__":
    train_pac_bifurcation()
