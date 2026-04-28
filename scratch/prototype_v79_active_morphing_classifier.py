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

def extract_archetypes(dataset, device, num_strokes=15, epochs=150):
    print("--- Phase 1: Extracting Archetypes (Dictionary Generation) ---")
    archetype_params = {}
    
    for digit in range(10):
        # Compute the mean image for the current digit
        images_of_digit = [img.squeeze() for img, label in dataset if label == digit]
        mean_img = torch.stack(images_of_digit).mean(dim=0).to(device)
        
        print(f"Vectorizing Archetype '{digit}'...", end='', flush=True)
        model = VectorCanvas(num_strokes=num_strokes, device=device).to(device)
        
        # Use a high learning rate to quickly fit the mean image
        optimizer = optim.Adam(model.parameters(), lr=0.5)
        criterion = nn.MSELoss()
        
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            generated_img = model()
            loss = criterion(generated_img, mean_img)
            loss.backward()
            optimizer.step()
            
        # Store the learned parameters
        archetype_params[digit] = {
            'points': model.points.detach().clone(),
            'weights': model.weights.detach().clone()
        }
        print(f" Final Loss: {loss.item():.4f}")
        
    return archetype_params

def active_inference(test_loader, archetype_params, device, lambda_elastic=0.05, steps=30, num_test=50):
    print(f"\n--- Phase 2: Active Inference (Testing on {num_test} images) ---")
    correct = 0
    total = 0
    
    results_to_plot = [] # For visualization
    
    for img, label in test_loader:
        img = img.squeeze().to(device)
        actual_label = label.item()
        
        best_loss = float('inf')
        predicted_class = -1
        best_canvas = None
        all_class_canvases = [] # To store final morphed shapes for visualization
        
        # Run the optimization loop for all 10 archetypes
        for digit in range(10):
            model = VectorCanvas(num_strokes=15, device=device).to(device)
            
            # Seed the model with the archetype's base parameters
            model.points.data = archetype_params[digit]['points'].clone()
            model.weights.data = archetype_params[digit]['weights'].clone()
            
            # Keep a reference to the base points to calculate the elastic penalty
            base_points = archetype_params[digit]['points'].clone()
            
            optimizer = optim.Adam(model.parameters(), lr=0.2)
            criterion_mse = nn.MSELoss()
            
            final_loss_val = 0
            for step in range(steps):
                optimizer.zero_grad()
                generated = model()
                
                # Visual discrepancy
                mse_loss = criterion_mse(generated, img)
                
                # Elastic deformation penalty
                elastic_loss = torch.mean((model.points - base_points)**2)
                
                # Total loss
                loss = mse_loss + lambda_elastic * elastic_loss
                
                loss.backward()
                optimizer.step()
                final_loss_val = loss.item()
                
            all_class_canvases.append(model().detach().cpu().numpy())
            
            # Select the class that achieved the lowest total loss
            if final_loss_val < best_loss:
                best_loss = final_loss_val
                predicted_class = digit
                best_canvas = all_class_canvases[-1]
                
        if predicted_class == actual_label:
            correct += 1
        total += 1
        
        if total <= 5: # Save the first 5 examples for the gallery plot
            results_to_plot.append((img.cpu().numpy(), actual_label, predicted_class, all_class_canvases))
            
        print(f"Test {total}/{num_test} | True: {actual_label} | Pred: {predicted_class} | Correct: {predicted_class == actual_label}")
        
        if total >= num_test:
            break
            
    accuracy = 100 * correct / total
    print(f"\nFinal Accuracy on {num_test} tests: {accuracy:.2f}%")
    return results_to_plot

def plot_results(results_to_plot):
    num_examples = len(results_to_plot)
    fig, axes = plt.subplots(num_examples, 12, figsize=(20, 2.5 * num_examples))
    fig.suptitle("Active Morphing Classifier: Target vs 10 Morphed Archetypes", fontsize=16)
    
    for i, (target_img, actual_label, pred_label, morphed_canvases) in enumerate(results_to_plot):
        axes[i, 0].imshow(target_img, cmap='gray')
        axes[i, 0].set_title(f"Target: {actual_label}")
        axes[i, 0].axis('off')
        
        # Blank separator
        axes[i, 1].axis('off')
        
        for d in range(10):
            ax = axes[i, d+2]
            ax.imshow(morphed_canvases[d], cmap='magma')
            
            # Highlight the prediction in green, others in red/black
            color = 'green' if d == pred_label else 'black'
            weight = 'bold' if d == pred_label else 'normal'
            if d != pred_label and actual_label != pred_label:
                # If predicted wrong, highlight the wrong guess in red
                if d == pred_label:
                    color = 'red'
            elif d == pred_label and actual_label != pred_label:
                color = 'red'
                
            ax.set_title(f"Morph '{d}'", color=color, fontweight=weight)
            ax.axis('off')
            
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    save_path = "results/figures/v79_active_inference_examples.png"
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    transform = transforms.ToTensor()
    # For Phase 1, we use the training set to extract pure archetypes
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # For Phase 2, we test on unseen data
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    start_time = time.time()
    archetypes = extract_archetypes(train_dataset, device, num_strokes=15, epochs=150)
    print(f"Phase 1 Time: {time.time() - start_time:.2f}s")
    
    start_time = time.time()
    # Test on 50 images, with lambda=0.05 and 30 optimization steps per morph
    results_to_plot = active_inference(test_loader, archetypes, device, lambda_elastic=0.05, steps=30, num_test=50)
    print(f"Phase 2 Time: {time.time() - start_time:.2f}s")
    
    plot_results(results_to_plot)

if __name__ == "__main__":
    main()
