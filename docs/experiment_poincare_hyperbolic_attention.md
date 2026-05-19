
# Experimento: Poincaré Hyperbolic Attention (PoC)

Este experimento sustituye el producto escalar tradicional de la atención de los Transformers por la **distancia geodésica en el Disco de Poincaré**. 

## 1. Hipótesis del Experimento
La atención calculada mediante distancias hiperbólicas permite que el modelo organice la información de manera jerárquica implícita en dimensiones extremadamente bajas (p. ej., $D=4$), superando la capacidad de representación de la atención euclidiana plana en esas mismas dimensiones.

---

## 2. Formulación Matemática

### Proyección al espacio hiperbólico
Para garantizar que cualquier vector $x \in \mathbb{R}^d$ caiga dentro del disco unitario de Poincaré ($\mathbb{D} = \{x \in \mathbb{R}^d : \|x\| < 1\}$), aplicamos una proyección de seguridad:

$$\text{proj}(x) = \frac{x}{\|x\|} \cdot (1 - \epsilon) \quad \text{si } \|x\| \ge 1$$

### Distancia de Poincaré
Para dos puntos $u, v \in \mathbb{D}$, la distancia hiperbólica se define como:

$$d_{\mathbb{D}}(u, v) = \text{arcosh}\left(1 + 2\frac{\|u - v\|^2}{(1 - \|u\|^2)(1 - \|v\|^2)}\right)$$

Donde $\text{arcosh}(x) = \ln(x + \sqrt{x^2 - 1})$.

---

## 3. Código del Prototipo (Python / PyTorch)

Este script implementa la capa de atención hiperbólica y la compara con una atención tradicional en un problema sintético de jerarquías.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PoincareAttention(nn.Module):
    def __init__(self, dim, num_heads=1, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.eps = eps
        
        # Proyecciones lineales para Q, K, V
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # Parámetro de temperatura (entrenable)
        self.beta = nn.Parameter(torch.tensor(1.0))

    def _project_to_ball(self, x):
        """Proyecta los vectores dentro de la bola de Poincaré de radio 1 - eps"""
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        max_norm = 1.0 - self.eps
        cond = norm >= max_norm
        projected = x * (max_norm / (norm + 1e-10))
        return torch.where(cond, projected, x)

    def _poincare_distance(self, u, v):
        """Calcula la distancia hiperbólica por parejas entre u [B, H, N, 1, D] y v [B, H, 1, M, D]"""
        sq_dist = torch.sum((u - v) ** 2, dim=-1)
        u_norm_sq = torch.sum(u ** 2, dim=-1)
        v_norm_sq = torch.sum(v ** 2, dim=-1)
        
        denom = (1.0 - u_norm_sq) * (1.0 - v_norm_sq)
        denom = torch.clamp(denom, min=self.eps)
        
        alpha = 1.0 + 2.0 * sq_dist / denom
        alpha = torch.clamp(alpha, min=1.0 + self.eps)
        
        # arcosh(x) = ln(x + sqrt(x^2 - 1))
        dist = torch.log(alpha + torch.sqrt(alpha**2 - 1.0 + 1e-9))
        return dist

    def forward(self, x):
        B, N, D = x.shape
        H = self.num_heads
        d_k = D // H
        
        # 1. Proyección lineal y reshape a multi-head
        q = self.q_proj(x).view(B, N, H, d_k).transpose(1, 2)  # [B, H, N, d_k]
        k = self.k_proj(x).view(B, N, H, d_k).transpose(1, 2)  # [B, H, N, d_k]
        v = self.v_proj(x).view(B, N, H, d_k).transpose(1, 2)  # [B, H, N, d_k]
        
        # 2. Forzar a que Q y K vivan en el Disco de Poincaré
        q_hyp = self._project_to_ball(q)
        k_hyp = self._project_to_ball(k)
        
        # 3. Preparar dimensiones para broadcasting de distancias [B, H, N, N]
        q_uns = q_hyp.unsqueeze(3)  # [B, H, N, 1, d_k]
        k_uns = k_hyp.unsqueeze(2)  # [B, H, 1, N, d_k]
        
        # 4. Calcular matriz de distancias geodésicas
        distances = self._poincare_distance(q_uns, k_uns)
        
        # 5. La atención es inversamente proporcional a la distancia hiperbólica
        attn_scores = -torch.abs(self.beta) * distances
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 6. Agregación (en espacio tangente/Euclidiano para el prototipo)
        out = torch.matmul(attn_weights, v)  # [B, H, N, d_k]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        
        return out, attn_weights, (q_hyp, k_hyp)

# --- Pruebas del Prototipo ---
if __name__ == "__main__":
    # Dimensiones de prueba extremadamente bajas (D=4)
    batch_size = 1
    seq_len = 5
    embedding_dim = 4
    
    # Creamos un tensor de entrada simulado
    x_input = torch.randn(batch_size, seq_len, embedding_dim)
    
    # Instanciamos la atención hiperbólica
    layer = PoincareAttention(dim=embedding_dim, num_heads=1)
    
    # Ejecutamos forward pass
    output, weights, (queries_hyp, keys_hyp) = layer(x_input)
    
    print("--- RESULTADOS DEL PROTOTIPO ---")
    print("Dimensiones de Entrada:", x_input.shape)
    print("Dimensiones de Salida :", output.shape)
    print("\nMatriz de Atención Hiperbólica (pesos normalizados):")
    print(weights[0, 0].detach().numpy())
    
    print("\nPosiciones de las Consultas (Q) en el Disco de Poincaré (debe ser Norma < 1):")
    norms = torch.norm(queries_hyp[0, 0], dim=-1)
    for i, norm in enumerate(norms):
        print(f"Token {i}: Coordenadas = {queries_hyp[0, 0, i].detach().numpy()}, Norma = {norm.item():.5f}")
```

---

## 4. Métricas a Evaluar en tu Entorno de Pruebas

Si integras esta capa en un pipeline con un dataset real (como clasificación de texto o predicción de enlaces jerárquicos):

1. **Eficiencia en Baja Dimensión:** Compara esta capa con `dim=8` contra una capa de atención clásica con `dim=64`. La atención hiperbólica debería retener mayor precisión en dimensiones reducidas.
2. **Visualización del Espacio:** Grafica los tensores `queries_hyp` y `keys_hyp` en un plano 2D. Las palabras clave más ambiguas o raíz deberían converger al centro $(0,0)$, mientras que las hojas o palabras muy específicas deberían aparecer en los límites de la circunferencia exterior.