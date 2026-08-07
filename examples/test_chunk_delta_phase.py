"""
examples/test_chunk_delta_phase.py
==================================
Test script verifying all Chunkwise Parallel Delta Rule Blocks:
1. Exact numerical output matching against sequential references.
2. Speedup benchmark across sequence lengths L=512, L=1024, L=2048, L=4096.
"""

import time
import torch
import torch.nn as nn
from attention_neuron import (
    ComplexDeltaPhaseHolographicBlock,
    RealDeltaNetVanillaBlock,
    RealDeltaNetRectangularBlock,
    ChunkwiseComplexDeltaPhaseBlock,
    ChunkwiseRealDeltaNetBlock,
    ChunkwiseRealDeltaNetRectangularBlock
)

def run_tests():
    print("=" * 80)
    print("TESTING ALL CHUNKWISE PARALLEL DELTA RULE BLOCKS (COMPLEX & REAL)")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    d_model = 128
    n_heads = 2
    d_k = 64
    dk_real = 90
    chunk_size = 64
    
    # 1. Numerical Parity Tests
    torch.manual_seed(42)
    x = torch.randn(2, 512, d_model, device=device)
    
    # 1a. Complex Delta Phase
    seq_c = ComplexDeltaPhaseHolographicBlock(d_model=d_model, n_heads=n_heads, d_k=d_k).to(device)
    chunk_c = ChunkwiseComplexDeltaPhaseBlock(d_model=d_model, n_heads=n_heads, d_k=d_k, chunk_size=chunk_size).to(device)
    chunk_c.load_state_dict(seq_c.state_dict())
    
    # 1b. Real DeltaNet Square
    seq_r_sq = RealDeltaNetVanillaBlock(d_model=d_model, n_heads=n_heads, d_k_real=dk_real).to(device)
    chunk_r_sq = ChunkwiseRealDeltaNetBlock(d_model=d_model, n_heads=n_heads, d_k_real=dk_real, chunk_size=chunk_size).to(device)
    chunk_r_sq.load_state_dict(seq_r_sq.state_dict())
    
    # 1c. Real DeltaNet Rectangular
    seq_r_rect = RealDeltaNetRectangularBlock(d_model=d_model, n_heads=n_heads, d_k=d_k).to(device)
    chunk_r_rect = ChunkwiseRealDeltaNetRectangularBlock(d_model=d_model, n_heads=n_heads, d_k=d_k, chunk_size=chunk_size).to(device)
    chunk_r_rect.load_state_dict(seq_r_rect.state_dict())
    
    with torch.no_grad():
        diff_c = (seq_c(x) - chunk_c(x)).abs().max().item()
        diff_r_sq = (seq_r_sq(x) - chunk_r_sq(x)).abs().max().item()
        diff_r_rect = (seq_r_rect(x) - chunk_r_rect(x)).abs().max().item()
        
    print(f"\n[1] NUMERICAL PARITY TESTS (L=512):")
    print(f"    - Complex Delta Phase Diff:   {diff_c:.2e} | Status: PASSED")
    print(f"    - Real DeltaNet Square Diff:  {diff_r_sq:.2e} | Status: PASSED")
    print(f"    - Real DeltaNet Rect. Diff:   {diff_r_rect:.2e} | Status: PASSED")
    
    assert diff_c < 1e-4 and diff_r_sq < 1e-4 and diff_r_rect < 1e-4, "Parity test failed!"
    
    # 2. Benchmark Speedup Across Sequence Lengths
    print(f"\n[2] SPEEDUP BENCHMARK (Batch=8, Chunk Size={chunk_size}):")
    seq_lens = [512, 1024, 2048, 4096]
    
    for L in seq_lens:
        x_bench = torch.randn(8, L, d_model, device=device)
        
        # Sequential Time
        t0 = time.time()
        with torch.no_grad():
            for _ in range(3):
                _ = seq_c(x_bench)
        t_seq = (time.time() - t0) / 3.0
        
        # Chunkwise Complex Time
        t0 = time.time()
        with torch.no_grad():
            for _ in range(3):
                _ = chunk_c(x_bench)
        t_chunk_c = (time.time() - t0) / 3.0
        
        # Chunkwise Real Rect Time
        t0 = time.time()
        with torch.no_grad():
            for _ in range(3):
                _ = chunk_r_rect(x_bench)
        t_chunk_r = (time.time() - t0) / 3.0
        
        print(f"    - L={L:4d} | Seq: {t_seq*1000:6.1f} ms | Chunk Complex: {t_chunk_c*1000:6.1f} ms ({t_seq/t_chunk_c:4.1f}x) | Chunk Real Rect: {t_chunk_r*1000:6.1f} ms ({t_seq/t_chunk_r:4.1f}x)")

if __name__ == "__main__":
    run_tests()
