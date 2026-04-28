import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import time

def run_pac_clustering(train_data, train_targets, device, max_iter=200):
    print("--- Phase 1: PAC Clustering (Fast Dictionary Generation) ---")
    flat_data = train_data.view(-1, 784)
    
    image_cluster_assignment = train_targets.clone()
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
        
        correct = (pred_labels == train_targets)
        acc = correct.float().mean().item()
        
        if (iteration+1) % 10 == 0 or iteration == max_iter - 1:
            print(f"Gen {iteration+1} | Active Archetypes: {len(active_clusters):3d} | Train Acc: {acc*100:.2f}%")
        
        if iteration == max_iter - 1:
            break
            
        image_cluster_assignment[correct] = arch_cluster_ids[best_arch_idx[correct]]
        
        for digit in range(10):
            mask_err = (~correct) & (train_targets == digit)
            if mask_err.sum() > 0:
                image_cluster_assignment[mask_err] = next_cluster_id
                cluster_to_label[next_cluster_id] = digit
                next_cluster_id += 1
                
    return arch_tensors, arch_labels

def pac_knn_inference(test_data, test_targets, arch_tensors, arch_labels, k=10):
    print(f"\n--- Phase 2: PAC K-NN Inference (Voting with Top-{k}) ---")
    flat_test = test_data.view(-1, 784)
    
    # 1. Distancias desde cada imagen de test a todos los arquetipos PAC
    start_time = time.time()
    dist = torch.cdist(flat_test, arch_tensors, p=2)
    
    # 2. Encontrar los K más cercanos para cada imagen
    topk_indices = torch.topk(dist, k, dim=1, largest=False).indices
    
    # 3. Mapear esos índices a sus etiquetas reales
    topk_labels = arch_labels[topk_indices] # (num_test, k)
    
    # 4. Votación (Majority Vote)
    # torch.mode devuelve el valor más frecuente en la dimensión especificada
    predicted_labels = torch.mode(topk_labels, dim=1).values
    
    inference_time = time.time() - start_time
    
    # 5. Calcular Accuracy
    correct = (predicted_labels == test_targets).sum().item()
    total = test_targets.size(0)
    accuracy = 100 * correct / total
    
    print(f"Inference Time for {total} images: {inference_time:.4f}s")
    print(f"Final Test Accuracy (PAC + {k}-NN): {accuracy:.2f}%")
    return accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=60000, shuffle=False)
    train_data, train_targets = next(iter(train_loader))
    train_data, train_targets = train_data.to(device), train_targets.to(device)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=False)
    test_data, test_targets = next(iter(test_loader))
    test_data, test_targets = test_data.to(device), test_targets.to(device)
    
    start_time = time.time()
    # Fase 1: Entrenar el Diccionario PAC
    arch_tensors, arch_labels = run_pac_clustering(train_data, train_targets, device, max_iter=200)
    print(f"PAC Extraction Time: {time.time() - start_time:.2f}s")
    
    # Fase 2: Probar el sistema de votación para diferentes valores de K
    print("\nEvaluando con K variable:")
    for k in [1, 3, 5, 10, 15]:
        pac_knn_inference(test_data, test_targets, arch_tensors, arch_labels, k=k)

if __name__ == "__main__":
    main()
