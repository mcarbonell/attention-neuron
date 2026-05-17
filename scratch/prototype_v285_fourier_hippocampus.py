"""
prototype_v285_fourier_hippocampus.py
=====================================
V285: Spectral KV Cache (The Fourier Hippocampus)

Demuestra la capacidad de la arquitectura Matrix-Free para mantener
un contexto infinito con memoria O(1) cruzando la frontera de los chunks
mediante el almacenamiento de bajas frecuencias.
"""

import torch, torch.nn as nn, torch.nn.functional as F
import math, time

CFG = dict(
    vocab_size=32,
    d_model=128, n_layers=3, k_walsh=32,
    k_mem=16, # El hipocampo solo guarda las 16 frecuencias más bajas!
    chunk_size=32,
    num_chunks=5, # Secuencia total = 32 * 5 = 160 tokens
    epochs=150, lr=0.01,
    seed=42, batch_size=64, steps_per_epoch=100,
    gamma=0.9, # Factor de decaimiento temporal del Hipocampo
)

# ── Data: Tarea Sintética de Memoria Oculta ───────────────────────────
# La red debe aprender que en el Chunk 1 se le da una clave-valor (ej. 5=9;)
# Y en el Chunk N (lejos de su ventana de atención de chunk) se le pregunta (?5)
# y debe predecir 9.
def generate_synthetic_batch(bs, chunk_size, num_chunks, vocab_size):
    total_len = chunk_size * num_chunks
    x = torch.randint(15, vocab_size, (bs, total_len)) # tokens de ruido (15 a V-1)
    y = torch.full((bs, total_len), -100) # solo nos importa la predicción final
    
    for i in range(bs):
        # Chunk 1: inyectamos la clave-valor [K, =, V, ;]
        key = torch.randint(0, 10, (1,)).item()
        val = torch.randint(0, 10, (1,)).item()
        
        pos_write = torch.randint(0, chunk_size - 4, (1,)).item()
        x[i, pos_write] = key
        x[i, pos_write+1] = 10 # '='
        x[i, pos_write+2] = val
        x[i, pos_write+3] = 11 # ';'
        
        # Chunk N: preguntamos por la clave [?, K] y el target es V
        pos_read = total_len - chunk_size + torch.randint(0, chunk_size - 3, (1,)).item()
        x[i, pos_read] = 12 # '?'
        x[i, pos_read+1] = key
        
        # El target a predecir es el valor, desplazado en 1 para el modelo autorregresivo
        # Es decir, cuando la red ve 'key' (pos_read+1), debe predecir 'val'.
        y[i, pos_read+1] = val
        
    return x, y

# ══════════════════════════════════════════════════════════════════════
# UTILS (nGPT & Walsh)
# ══════════════════════════════════════════════════════════════════════
def norm_sphere(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

class SphericalHead(nn.Module):
    def __init__(self, in_features, out_features, init_tau=10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        self.tau = nn.Parameter(torch.tensor(init_tau))

    def forward(self, x):
        return F.linear(F.normalize(x, dim=-1), F.normalize(self.weight, dim=-1)) * self.tau

def get_walsh_matrix_1d(dim):
    if dim == 1: return torch.tensor([[1.]])
    H = get_walsh_matrix_1d(dim // 2)
    return torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0) / math.sqrt(2)

class WalshLinear(nn.Module):
    def __init__(self, in_features, out_features, k):
        super().__init__()
        self.k = k
        self.core = nn.Parameter(torch.randn(k, k) / math.sqrt(k))
        self.scale = nn.Parameter(torch.ones(1))
        self.register_buffer('H_in', get_walsh_matrix_1d(in_features))
        self.register_buffer('H_out', get_walsh_matrix_1d(out_features))

    def forward(self, x):
        W = self.H_out[:, :self.k] @ self.core @ self.H_in[:self.k, :] 
        return F.linear(x, F.normalize(W, dim=-1)) * self.scale

# ══════════════════════════════════════════════════════════════════════
# FOURIER HIPPOCAMPUS (Stateful Mixer)
# ══════════════════════════════════════════════════════════════════════
class StatefulComplexFFTMixer(nn.Module):
    def __init__(self, T, D, k_walsh, k_mem, gamma):
        super().__init__()
        self.T = T
        self.pad_T = 1
        while self.pad_T < 2*T: self.pad_T *= 2
        self.n_freq = self.pad_T // 2 + 1
        
        self.log_amp = nn.Parameter(torch.zeros(self.n_freq))
        self.phase   = nn.Parameter(torch.zeros(self.n_freq))
        
        mask = torch.zeros(self.pad_T)
        mask[:T] = 1.0
        self.register_buffer('causal_mask', mask)
        self.out_proj = WalshLinear(D, D, k_walsh)
        
        # Hippocampus Projections
        self.k_mem = min(k_mem, self.n_freq)
        self.gamma = gamma
        self.read_gate  = nn.Parameter(torch.ones(self.k_mem, 1, dtype=torch.complex64))
        self.write_gate = nn.Parameter(torch.ones(self.k_mem, 1, dtype=torch.complex64))

    def forward(self, x, memory_state=None):
        B, T, D = x.shape
        xt = x.permute(0, 2, 1) # B, D, T
        pad = torch.zeros(B, D, self.pad_T-T, device=x.device)
        xt_pad = torch.cat([xt, pad], dim=-1)
        
        X = torch.fft.rfft(xt_pad, dim=-1) # B, D, freq

        # Causal Gate (Local Context)
        gate_raw  = torch.exp(self.log_amp) * torch.exp(1j * self.phase)
        h_raw     = torch.fft.irfft(gate_raw, n=self.pad_T)
        h_causal  = h_raw * self.causal_mask
        gate_causal = torch.fft.rfft(h_causal, n=self.pad_T)
        
        X_gated = X * gate_causal
        X_gated_perm = X_gated.permute(0, 2, 1) # B, freq, D
        
        # ========================================================
        # HIPOCAMPO: LECTURA
        # Si tenemos un estado previo, lo sumamos a las frecuencias bajas
        # ========================================================
        if memory_state is not None:
            # memory_state es (B, k_mem, D)
            X_gated_perm[:, :self.k_mem, :] += memory_state * self.read_gate
            
        X_gated = X_gated_perm.permute(0, 2, 1) # Back to B, D, freq
        
        # IFFT al tiempo
        out = torch.fft.irfft(X_gated, n=self.pad_T, dim=-1)[..., :T]
        out = out.permute(0, 2, 1)
        
        # ========================================================
        # HIPOCAMPO: ESCRITURA
        # Extraemos las frecuencias procesadas y actualizamos el estado
        # ========================================================
        current_mem = X_gated_perm[:, :self.k_mem, :].clone()
        if memory_state is None:
            new_memory_state = current_mem * self.write_gate
        else:
            # Mezcla exponencial
            new_memory_state = self.gamma * memory_state + (1 - self.gamma) * (current_mem * self.write_gate)
            
        return self.out_proj(out), new_memory_state

class NarrowFFN(nn.Module):
    def __init__(self, D, k_walsh):
        super().__init__()
        self.proj = WalshLinear(D, D, k_walsh)
    def forward(self, x):
        return F.gelu(self.proj(x))

class nGPTBlockStateful(nn.Module):
    def __init__(self, D, T, k_walsh, k_mem, gamma):
        super().__init__()
        self.mixer = StatefulComplexFFTMixer(T, D, k_walsh, k_mem, gamma)
        self.ffn = NarrowFFN(D, k_walsh)
        self.alpha_m = nn.Parameter(torch.full((D,), 0.05))
        self.alpha_f = nn.Parameter(torch.full((D,), 0.05))

    def forward(self, x, state=None):
        m, next_state = self.mixer(x, state)
        m = norm_sphere(m)
        x = norm_sphere(x + self.alpha_m.abs() * m)
        
        f = norm_sphere(self.ffn(x))
        x = norm_sphere(x + self.alpha_f.abs() * f)
        
        return x, next_state

# ══════════════════════════════════════════════════════════════════════
# MODELO GLOBAL CON PROCESAMIENTO POR CHUNKS
# ══════════════════════════════════════════════════════════════════════
class HippocampusLM(nn.Module):
    def __init__(self, vocab_size, T_chunk, D, L, k_walsh, k_mem, gamma):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, D)
        self.blocks = nn.ModuleList([nGPTBlockStateful(D, T_chunk, k_walsh, k_mem, gamma) for _ in range(L)])
        self.head = SphericalHead(D, vocab_size, init_tau=15.0)

    def forward(self, x_full, chunk_size):
        """
        x_full: (B, total_len)
        Troceamos internamente en chunks, mantenemos el estado vivo, y devolvemos 
        los logits concatenados. Así entrenamos con BPTT completo a través del estado.
        """
        B, total_len = x_full.shape
        num_chunks = total_len // chunk_size
        
        h_full = norm_sphere(self.embed(x_full))
        
        out_chunks = []
        states = [None] * len(self.blocks)
        
        for c in range(num_chunks):
            h_chunk = h_full[:, c*chunk_size : (c+1)*chunk_size, :]
            
            for i, block in enumerate(self.blocks):
                h_chunk, states[i] = block(h_chunk, states[i])
                
            out_chunks.append(h_chunk)
            
        h_final = torch.cat(out_chunks, dim=1)
        return self.head(h_final)

# ══════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════
def run_experiment():
    print(f"\nV285: Spectral KV Cache (The Fourier Hippocampus)")
    print(f"Tarea: Memoria Oculta a Largo Plazo (Contexto > Ventana)")
    
    torch.manual_seed(CFG['seed'])
    
    model = HippocampusLM(CFG['vocab_size'], CFG['chunk_size'], CFG['d_model'], 
                          CFG['n_layers'], CFG['k_walsh'], CFG['k_mem'], CFG['gamma'])
                          
    print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,} | Chunks: {CFG['num_chunks']} de {CFG['chunk_size']} tokens.")
    print(f"Tamaño del Hipocampo (RAM persistente): {CFG['k_mem']} frecuencias por capa.")
    
    opt = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
    
    for ep in range(1, CFG['epochs'] + 1):
        model.train()
        ep_loss = 0.0
        corrects = 0
        total_eval = 0
        
        for b in range(CFG['steps_per_epoch']):
            x, y = generate_synthetic_batch(CFG['batch_size'], CFG['chunk_size'], CFG['num_chunks'], CFG['vocab_size'])
            
            # Forward pasando toda la secuencia troceada y con memoria persistente
            logits = model(x, CFG['chunk_size'])
            
            # Solo penalizamos en el momento de la predicción final (-100 se ignora)
            loss = F.cross_entropy(logits.reshape(-1, CFG['vocab_size']), y.reshape(-1))
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            ep_loss += loss.item()
            
            # Calcular Accuracy sobre los tokens que no son -100
            mask = y != -100
            preds = logits.argmax(dim=-1)[mask]
            targets = y[mask]
            corrects += (preds == targets).sum().item()
            total_eval += targets.numel()

        acc = (corrects / total_eval) * 100
        print(f"Ep {ep:03d} | Loss: {ep_loss/CFG['steps_per_epoch']:.4f} | Recuperación Hipocampo: {acc:.1f}%")
        
        if acc >= 99.0:
            print("\n[ÉXITO MASIVO] ¡El Hipocampo ha resuelto la paradoja de memoria infinita O(1)!")
            break

if __name__ == '__main__':
    run_experiment()
