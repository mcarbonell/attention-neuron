import random
import torch
from torch.utils.data import Dataset, DataLoader

class MQARDataset(Dataset):
    """
    Dataset sintético Multi-Query Associative Recall (MQAR) según el estándar de la literatura 
    (Zoology - Arora et al., 2023 / H3 - Dao et al., 2022).
    
    Estructura de la secuencia MQAR:
    - Formato de pares continuos: [K_1, V_1, K_2, V_2, ..., K_m, V_m]
    - Relleno con tokens pasivos / espacios.
    - Marcador explícito de consulta al final: [QUERY_MARKER, K_target] -> Objetivo: V_target.
    """
    def __init__(self, num_samples: int = 2000, seq_len: int = 128, num_pairs: int = 8, vocab_size: int = 64):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_pairs = num_pairs
        self.vocab_size = vocab_size
        
        self.PAD = 0
        self.QUERY_MARKER = 1
        self.VOCAB_START = 2

    def __len__(self):
        return self.num_samples

    def _generate_single_sample(self):
        available_keys = list(range(self.VOCAB_START, self.VOCAB_START + (self.vocab_size - 2) // 2))
        available_vals = list(range(self.VOCAB_START + (self.vocab_size - 2) // 2, self.vocab_size))
        
        # 1. Generar pares K-V únicos
        keys = random.sample(available_keys, self.num_pairs)
        vals = [random.choice(available_vals) for _ in range(self.num_pairs)]
        
        # 2. Construir secuencia con espacio entre pares
        seq = [self.PAD] * self.seq_len
        
        # Insertar los pares en la primera mitad de la secuencia de forma ordenada/espaciada
        spacing = (self.seq_len - 10) // (self.num_pairs * 2)
        for i in range(self.num_pairs):
            pos_k = i * 2 * spacing
            pos_v = pos_k + 1
            seq[pos_k] = keys[i]
            seq[pos_v] = vals[i]
            
        # 3. Elegir una clave como consulta al final
        target_idx = random.randint(0, self.num_pairs - 1)
        query_key = keys[target_idx]
        target_val = vals[target_idx]
        
        seq[-2] = self.QUERY_MARKER
        seq[-1] = query_key
        
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target_val, dtype=torch.long)

    def __getitem__(self, idx):
        return self._generate_single_sample()

def get_mqar_dataloader(num_samples: int = 2000, seq_len: int = 128, batch_size: int = 32, vocab_size: int = 64, num_pairs: int = 8):
    dataset = MQARDataset(num_samples=num_samples, seq_len=seq_len, num_pairs=num_pairs, vocab_size=vocab_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
