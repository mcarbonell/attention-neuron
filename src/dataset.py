import random
import torch
from torch.utils.data import Dataset, DataLoader

class DynamicAssociativeRecallDataset(Dataset):
    """
    Dataset sintético con generación DINÁMICA sobre la marcha.
    Evita que los modelos memoricen muestras estáticas (Overfitting de memorización)
    y los fuerza a aprender el algoritmo real de búsqueda Clave-Valor.
    """
    def __init__(self, num_samples: int = 2000, seq_len: int = 256, num_pairs: int = 4, vocab_size: int = 64):
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
        available_vocab = list(range(self.VOCAB_START, self.vocab_size))
        
        # 1. Ruido aleatorio en toda la secuencia
        seq = [random.choice(available_vocab) for _ in range(self.seq_len)]
        
        # 2. Claves y Valores únicos
        keys = random.sample(available_vocab, self.num_pairs)
        vals = [random.choice(available_vocab) for _ in range(self.num_pairs)]
        
        # 3. Posiciones sin solapamiento
        positions = []
        curr_pos = 0
        for i in range(self.num_pairs):
            remaining = self.num_pairs - i - 1
            max_pos = self.seq_len - 4 - (remaining * 3) - 2
            pos = random.randint(curr_pos, max(curr_pos, max_pos))
            positions.append(pos)
            curr_pos = pos + 3
            
        for i, pos in enumerate(positions):
            seq[pos] = keys[i]
            seq[pos + 1] = vals[i]
            
        # 4. Consulta al final
        target_idx = random.randint(0, self.num_pairs - 1)
        query_key = keys[target_idx]
        target_val = vals[target_idx]
        
        seq[-2] = self.QUERY_MARKER
        seq[-1] = query_key
        
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target_val, dtype=torch.long)

    def __getitem__(self, idx):
        # Generación dinámica fresca por cada item solicitado
        return self._generate_single_sample()

def get_dataloader(num_samples: int = 2000, seq_len: int = 256, batch_size: int = 32, vocab_size: int = 64, num_pairs: int = 4):
    dataset = DynamicAssociativeRecallDataset(num_samples=num_samples, seq_len=seq_len, num_pairs=num_pairs, vocab_size=vocab_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
