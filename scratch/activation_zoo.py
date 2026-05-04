import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x): return 1 / (1 + np.exp(-x))

x = np.linspace(-2, 2, 2000)
mask = np.abs(np.cos(np.pi * x)) < 0.01

# 1. Sigmoid-Tan (Rampa Sigmoidal)
y1 = sigmoid(np.tan(np.pi * x))
y1[mask] = np.nan

# 2. Arctan-Tan (Modulo / Sawtooth)
y2 = (1/np.pi) * np.arctan(np.tan(np.pi * x))
y2[mask] = np.nan

# 3. Differentiable Staircase (El Redondeo!)
# f(x) = x - Modulo(x)
y3 = x - y2

# 4. Tanh-Tan (Simétrica)
y4 = np.tanh(np.tan(np.pi * x))
y4[mask] = np.nan

fig, axs = plt.subplots(2, 2, figsize=(15, 10), facecolor='#0f172a')
fig.suptitle('Zoológico de Activaciones v2: De lo Analógico a lo Digital', color='white', fontsize=20)

styles = [
    (y1, r'$\sigma(\tan(\pi x))$', '#38bdf8', 'Rampa Sigmoidal'),
    (y2, r'$\frac{1}{\pi}\arctan(\tan(\pi x))$', '#10b981', 'Módulo (Sawtooth)'),
    (y3, r'$x - \text{Mod}(x)$', '#fbbf24', 'Escalera (Differentiable Round)'),
    (y4, r'$\tanh(\tan(\pi x))$', '#a855f7', 'Rampa Simétrica')
]

for i, (data, title, color, desc) in enumerate(styles):
    ax = axs[i//2, i%2]
    ax.set_facecolor('#0f172a')
    ax.plot(x, data, color=color, linewidth=3)
    ax.set_title(f"{title} - {desc}", color='white', fontsize=14)
    ax.grid(color='#1e293b', linestyle=':')
    ax.tick_params(colors='#94a3b8')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('results/figures/activation_zoo_v2.png', dpi=300)
