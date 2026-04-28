import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

def train_iterative_archetypes():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- V75: Iterative Sub-Archetype Discovery ---")
    print(f"Device: {device}")
    
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Load ALL data into GPU memory (MNIST is small enough)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=60000, shuffle=False)
    train_data, train_targets = next(iter(train_loader))
    train_data, train_targets = train_data.to(device), train_targets.to(device)
    flat_train = train_data.view(60000, 784)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=False)
    test_data, test_targets = next(iter(test_loader))
    test_data, test_targets = test_data.to(device), test_targets.to(device)
    flat_test = test_data.view(10000, 784)

    # Store archetypes as a list of (tensor, label, generation)
    archetypes = []
    
    # Generation 0: Global Means
    for digit in range(10):
        mask = (train_targets == digit)
        archetypes.append((flat_train[mask].mean(dim=0), digit, 0))
        
    MAX_ITER = 6
    
    print("\nStarting iterative discovery...")
    for iteration in range(MAX_ITER):
        arch_tensors = torch.stack([a[0] for a in archetypes]) # (K, 784)
        arch_labels = torch.tensor([a[1] for a in archetypes], device=device) # (K,)
        
        # Calculate distance from all train images to all current archetypes
        # torch.cdist calculates Euclidean distance, which is equivalent to MSE for argmin
        dist = torch.cdist(flat_train, arch_tensors, p=2) # (60000, K)
        pred_indices = torch.argmin(dist, dim=1)
        preds = arch_labels[pred_indices]
        
        correct = (preds == train_targets)
        acc = correct.float().mean().item()
        print(f"Gen {iteration} | Total Archetypes: {len(archetypes)} | Train Acc: {acc*100:.2f}%")
        
        if acc >= 0.99: 
            print("Reached >99% accuracy. Stopping.")
            break
            
        # Generate new archetypes from the failures
        new_archetypes = []
        for digit in range(10):
            # Find images of 'digit' that were misclassified
            mask_digit = (train_targets == digit)
            mask_failed = (~correct) & mask_digit
            
            num_failed = mask_failed.sum().item()
            if num_failed > 0:
                # The new archetype is the mean of the failed cases!
                new_arch = flat_train[mask_failed].mean(dim=0)
                new_archetypes.append((new_arch, digit, iteration + 1))
                
        archetypes.extend(new_archetypes)
        
    # --- EVALUATE ON TEST SET ---
    arch_tensors = torch.stack([a[0] for a in archetypes])
    arch_labels = torch.tensor([a[1] for a in archetypes], device=device)
    
    dist_test = torch.cdist(flat_test, arch_tensors, p=2)
    pred_test_indices = torch.argmin(dist_test, dim=1)
    preds_test = arch_labels[pred_test_indices]
    
    test_acc = (preds_test == test_targets).float().mean().item()
    print(f"\n=======================================")
    print(f"FINAL TEST ACCURACY: {test_acc*100:.2f}%")
    print(f"=======================================")

    # --- VISUALIZATION: THE ARCHEOLOGY OF DIGITS ---
    print("\nGenerating taxonomy grid...")
    # Group archetypes by digit for plotting
    archs_by_digit = {i: [] for i in range(10)}
    for tensor, label, gen in archetypes:
        archs_by_digit[label].append((tensor, gen))
        
    max_archs = max(len(lst) for lst in archs_by_digit.values())
    
    fig, axes = plt.subplots(10, max_archs, figsize=(max_archs*2, 15))
    fig.suptitle(f"The Evolutionary Taxonomy of MNIST Digits (Test Acc: {test_acc*100:.2f}%)", fontsize=20)
    
    for digit in range(10):
        lst = archs_by_digit[digit]
        for col in range(max_archs):
            ax = axes[digit, col]
            if col < len(lst):
                tensor, gen = lst[col]
                ax.imshow(tensor.cpu().view(28, 28).numpy(), cmap='magma')
                ax.set_title(f"Gen {gen}")
            ax.axis('off')
            
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    save_path = "results/figures/v75_iterative_archetypes.png"
    plt.savefig(save_path)
    print(f"Taxonomy saved to: {save_path}")

if __name__ == "__main__":
    train_iterative_archetypes()
