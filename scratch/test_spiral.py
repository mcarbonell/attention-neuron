import numpy as np
import matplotlib.pyplot as plt

def get_spiral_indices(size=32):
    """
    Generates indices for a center-out spiral scan on a square grid.
    Returns a 1D array of indices that can be used to reorder a flattened image.
    """
    indices = []
    x, y = size // 2, size // 2
    dx, dy = 0, -1
    
    # We need size*size indices
    for _ in range(size * size):
        if 0 <= x < size and 0 <= y < size:
            indices.append(y * size + x)
        
        # Change direction if needed
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        
        x, y = x + dx, y + dy
    
    # If the spiral goes out of bounds before reaching N*N, we need a more robust version
    return indices

def get_spiral_indices_v2(size=32):
    """
    Correct center-out spiral for any square grid.
    """
    indices = []
    r, c = size // 2, size // 2
    indices.append(r * size + c)
    
    step = 1
    while len(indices) < size * size:
        # Move Right
        for _ in range(step):
            c += 1
            if 0 <= r < size and 0 <= c < size:
                indices.append(r * size + c)
        # Move Down
        for _ in range(step):
            r += 1
            if 0 <= r < size and 0 <= c < size:
                indices.append(r * size + c)
        step += 1
        # Move Left
        for _ in range(step):
            c -= 1
            if 0 <= r < size and 0 <= c < size:
                indices.append(r * size + c)
        # Move Up
        for _ in range(step):
            r -= 1
            if 0 <= r < size and 0 <= c < size:
                indices.append(r * size + c)
        step += 1
        
    return indices[:size*size]

# Test and visualize
size = 8
idx = get_spiral_indices_v2(size)
grid = np.zeros((size, size))
for i, pos_idx in enumerate(idx):
    r, c = divmod(pos_idx, size)
    grid[r, c] = i

plt.imshow(grid, cmap='viridis')
plt.colorbar()
plt.title("Spiral Order (0=Center)")
plt.savefig("spiral_test.png")
print("Spiral indices generated and saved to spiral_test.png")
print(f"Indices: {idx[:10]}...")
