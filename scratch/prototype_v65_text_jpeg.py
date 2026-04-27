import torch
import torch.nn.functional as F
import math
import sys
import os

# Add tiny-thinker to path to allow unpickling the model structure
tiny_thinker_path = r"C:\Users\mrcm_\Local\proj\tiny-thinker"
sys.path.append(tiny_thinker_path)

from tokenizers import Tokenizer

def get_dct_matrix_1d(N, device='cpu'):
    """Generates a 1D DCT-II basis matrix of size (N, N)."""
    dct_mat = torch.zeros(N, N, device=device)
    for i in range(N):
        for j in range(N):
            if i == 0:
                dct_mat[i, j] = math.sqrt(1/N)
            else:
                dct_mat[i, j] = math.sqrt(2/N) * math.cos((math.pi * i * (2*j + 1)) / (2*N))
    return dct_mat

def get_nearest_tokens(reconstructed_embs, vocab_embs):
    """Finds the closest token ID for each reconstructed embedding vector using Cosine Similarity."""
    # reconstructed_embs: (seq_len, dim)
    # vocab_embs: (vocab_size, dim)
    
    # Normalize both to use fast cosine similarity (dot product of normalized vectors)
    rec_norm = F.normalize(reconstructed_embs, p=2, dim=1)
    vocab_norm = F.normalize(vocab_embs, p=2, dim=1)
    
    # Cosine similarities: (seq_len, vocab_size)
    similarities = torch.matmul(rec_norm, vocab_norm.t())
    
    # Get max similarity index
    best_ids = torch.argmax(similarities, dim=1)
    return best_ids.tolist()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V65: TEXT JPEG (THE SOUL OF A SENTENCE) ---")
    
    # 1. Load Tokenizer
    tokenizer_path = os.path.join(tiny_thinker_path, "model", "tokenizer.json")
    print("Loading Tokenizer...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # 2. Load Model Embeddings
    ckpt_path = os.path.join(tiny_thinker_path, "checkpoints", "ckpt_pretrain_best.pt")
    print(f"Loading Model Checkpoint from {ckpt_path}...")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        # Depending on how the model was saved, the dict might be inside 'model'
        if 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
            
        # Extract embeddings
        emb_weights = state_dict['tok_embeddings.weight']
        print(f"Successfully loaded embeddings! Shape: {emb_weights.shape}")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # 3. Target Sentence
    # We want a logical/rich sentence from tiny-thinker domain, or just a complex thought
    sentence = "The brave little dog decided to explore the dark forest, even though he was scared of the strange noises."
    print(f"\nOriginal Sentence: \n> \"{sentence}\"")
    
    # Tokenize
    encoded = tokenizer.encode(sentence)
    token_ids = encoded.ids
    tokens_str = [tokenizer.decode([t]) for t in token_ids]
    seq_len = len(token_ids)
    print(f"Tokenized ({seq_len} tokens): {tokens_str}")
    
    # 4. Get Sequence Embeddings
    # X shape: (seq_len, dim)
    X = emb_weights[token_ids].clone()
    dim = X.shape[1]
    
    # 5. Apply DCT along the sequence dimension!
    D = get_dct_matrix_1d(seq_len, device=device)
    
    # Transform: X_dct = D @ X
    X_dct = torch.matmul(D, X)
    
    print("\nApplying DCT Compression (JPEG for text)...")
    
    # Let's test different compression levels: keeping 100%, 50%, 25%, 15%
    keep_percentages = [1.0, 0.5, 0.25, 0.15]
    
    for pct in keep_percentages:
        k = max(1, int(seq_len * pct)) # Number of low frequencies to keep
        
        # Truncate high frequencies
        X_dct_truncated = X_dct.clone()
        X_dct_truncated[k:] = 0.0  # Zero out anything beyond frequency 'k'
        
        # Inverse DCT: X_hat = D^T @ X_dct_truncated
        X_hat = torch.matmul(D.t(), X_dct_truncated)
        
        # Reconstruct sentence by finding nearest tokens
        reconstructed_ids = get_nearest_tokens(X_hat, emb_weights)
        reconstructed_sentence = tokenizer.decode(reconstructed_ids)
        
        print(f"\n[Keep {pct*100:.0f}% of frequencies ({k}/{seq_len} components)]")
        # print(f"Tokens: {[tokenizer.decode([t]) for t in reconstructed_ids]}")
        print(f"Result: > \"{reconstructed_sentence}\"")

if __name__ == "__main__":
    main()
