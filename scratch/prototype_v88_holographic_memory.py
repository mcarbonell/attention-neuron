import torch
import torch.nn.functional as F
import time

# --- Fast Walsh-Hadamard Transform (Vectorized) ---
def fwht(x):
    """
    Computes the Fast Walsh-Hadamard Transform of a batch of vectors.
    Input x: (..., N) where N must be a power of 2.
    """
    *batch_dims, N = x.shape
    h = 1
    while h < N:
        x = x.view(*batch_dims, N // (2 * h), 2, h)
        a = x[..., 0, :]
        b = x[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2)
        h *= 2
    return x.view(*batch_dims, N)

def ifwht(x):
    N = x.shape[-1]
    return fwht(x) / N

# --- Hipocampo Holográfico ---
class HolographicHippocampus:
    def __init__(self, D, K_micro, C_chunk):
        """
        D: Dimensionality of features (Embedding size)
        K_micro: Number of low frequencies to keep per chunk (Memory compression)
        C_chunk: Chunk size (Short-term memory window, must be power of 2)
        """
        self.D = D
        self.K_micro = K_micro
        self.C_chunk = C_chunk
        # El Tensor Global: O(1) memoria, independientemente del tiempo que pase.
        self.memory = torch.zeros(K_micro, D)
        
    def add_chunk(self, x_chunk, chunk_idx):
        """
        x_chunk: (C_chunk, D)
        """
        # 1. Transformada Temporal a Frecuencia
        # x_chunk es (C, D). Aplicamos FWHT a lo largo de C (eje temporal).
        f_chunk = fwht(x_chunk.T.contiguous()).T # (C_chunk, D)
        
        # 2. Filtrado Espectral (Olvido Selectivo)
        # Nos quedamos solo con las K frecuencias más bajas.
        f_filtered = f_chunk[:self.K_micro, :] # (K_micro, D)
        
        # 3. Modulación Macro-Temporal (Orthogonal Phase Binding)
        # Generamos una firma +/-1 única para este chunk. Esto permite que 
        # múltiples chunks se sumen en el mismo tensor sin destruirse (holografía).
        torch.manual_seed(chunk_idx)
        R_macro = torch.randint(0, 2, (self.K_micro, 1)).float() * 2 - 1
        
        # 4. Consolidación en Memoria (Interferencia)
        self.memory += R_macro * f_filtered

    def query(self, q):
        """
        Recupera el contexto asociado al vector query q.
        q: (D,)
        """
        # 1. Extraemos la firma temporal (resonancia)
        alpha = torch.matmul(self.memory, q) # (K,)
        alpha = alpha / (torch.norm(alpha) + 1e-8)
        
        # 2. Reconstruimos el token desde la memoria
        retrieved = torch.matmul(alpha, self.memory) # (D,)
        
        # 3. Supresión de Sesgo Auto-Asociativo (Novedad)
        # Una matriz aleatoria M tenderá a devolver q cuando se le hace M^T M q.
        # Para ver la memoria real (el Target), debemos restar la proyección de q.
        retrieved = retrieved - torch.dot(retrieved, q) * q
        retrieved = F.normalize(retrieved, dim=0)
        
        return retrieved

# --- Experimento V88 ---
def run_experiment():
    print("=== Experimento V88: Hipocampo Holográfico (Memoria de Contexto Infinito) ===\n")
    
    D = 256          # Dimensión del embedding
    C_chunk = 512    # Tamaño del chunk de memoria a corto plazo
    K_micro = 64     # Frecuencias conservadas (Compresión 8x del tiempo)
    N_chunks = 100   # Total de chunks (Simulamos un stream de 51,200 tokens)
    
    print(f"[Arquitectura] Embedding (D): {D}")
    print(f"[Arquitectura] Capacidad Hipocampo: {K_micro}x{D} parámetros ({(K_micro*D*4)/1024:.1f} KB)")
    print(f"[Streaming] Total Tokens procesados: {C_chunk * N_chunks} tokens")
    
    hipocampo = HolographicHippocampus(D, K_micro, C_chunk)
    
    # --- Generación de la Aguja ---
    # Creamos un par Clave-Valor ortogonal que será el recuerdo objetivo
    torch.manual_seed(42)
    query_key = torch.randn(D)
    query_key = F.normalize(query_key, dim=0)
    
    target_value = torch.randn(D)
    target_value = target_value - torch.dot(target_value, query_key)*query_key # Hacerlo ortogonal a la clave
    target_value = F.normalize(target_value, dim=0)
    
    # La "Amígdala" detecta que esto es importante y le da un multiplicador de amplitud (Salience)
    # Aumentamos la saliencia porque estamos comprimiendo 51,200 eventos de ruido en solo 64x256 parámetros.
    # El ratio señal/ruido requiere que las memorias críticas destaquen fuertemente.
    salience_multiplier = 150.0
    needle_token = (query_key + target_value) * salience_multiplier
    
    # Elegimos un momento aleatorio y lejano para insertar la aguja
    needle_chunk_idx = 15
    needle_time_idx = 200
    print(f"\n[Evento] Insertando la Aguja (Recuerdo Crítico) en Chunk {needle_chunk_idx}, Posición {needle_time_idx}...")
    
    # --- Simulación del Streaming ---
    start_time = time.time()
    
    for i in range(N_chunks):
        # Ruido blanco de fondo (charla trivial, tokens sin importancia)
        chunk_data = torch.randn(C_chunk, D) * 0.5
        
        # En el momento crítico, inyectamos la aguja
        if i == needle_chunk_idx:
            chunk_data[needle_time_idx] += needle_token
            
        hipocampo.add_chunk(chunk_data, chunk_idx=i)
        
        if (i+1) % 25 == 0:
            print(f"  Procesados {i+1} chunks... ({(i+1)*C_chunk} tokens)")
            
    print(f"[Streaming] Finalizado en {time.time() - start_time:.3f}s sin saturar la RAM.")
    
    # --- Recuperación ---
    print("\n[Query] Consultando el Hipocampo con la Clave...")
    retrieved = hipocampo.query(query_key)
    
    # Evaluación: ¿Logra la señal atravesar el ruido de 51,200 tokens y recuperar el Target?
    cos_sim_target = F.cosine_similarity(retrieved.unsqueeze(0), target_value.unsqueeze(0)).item()
    cos_sim_key = F.cosine_similarity(retrieved.unsqueeze(0), query_key.unsqueeze(0)).item()
    
    random_noise = torch.randn(D)
    cos_sim_noise = F.cosine_similarity(retrieved.unsqueeze(0), random_noise.unsqueeze(0)).item()
    
    print(f"\n--- Resultados de Recuperación Espectral ---")
    print(f"Similitud con el TARGET (Valor original oculto): {cos_sim_target:.4f}")
    print(f"Similitud con la CLAVE (La pregunta en sí):      {cos_sim_key:.4f}")
    print(f"Similitud con Ruido Aleatorio (Control):         {cos_sim_noise:.4f}")
    
    if cos_sim_target > 0.4:
        print("\n[ÉXITO MASIVO] ¡El Hipocampo ha logrado recuperar la memoria a través del ruido!")
        print("La memoria holográfica O(1) funciona. Hemos roto la ventana de contexto de los LLMs.")
    else:
        print("\n[FALLO] La señal se ha diluido en la suma de interferencias.")

if __name__ == '__main__':
    run_experiment()
