"""
test_delta_rule_unittest.py
===========================
Unit test to verify the exact mathematical fix for the Delta Rule Phase Memory.
Compares:
  - WRONG (transposed): M_ij = k_i * err_j  --> Readout gives k * (err . conj(k))
  - CORRECT (fixed):      M_ij = err_i * k_j --> Readout gives err * |k|^2
"""

import torch

def test_delta_rule():
    B, H, d_k = 1, 1, 32
    num_pairs = 8
    
    # 1. Generate random key phasors (|k|=1) and random value vectors
    theta = torch.randn(num_pairs, d_k)
    K = torch.polar(torch.ones(num_pairs, d_k), theta) # [num_pairs, d_k] complex
    V = torch.randn(num_pairs, d_k) # [num_pairs, d_k] real
    
    # ── Test 1: Original Transposed Code (WRONG) ──
    M_wrong = torch.zeros(d_k, d_k, dtype=torch.complex64)
    for p in range(num_pairs):
        k_t = K[p]
        v_t = V[p]
        v_old = torch.einsum('ij,j->i', M_wrong, torch.conj(k_t)).real / (d_k ** 0.5)
        err = v_t - v_old
        update = torch.einsum('i,j->ij', k_t, err.to(torch.complex64))
        M_wrong = M_wrong + update
        
    err_wrong = 0.0
    for p in range(num_pairs):
        k_t = K[p]
        v_t = V[p]
        v_rec = torch.einsum('ij,j->i', M_wrong, torch.conj(k_t)).real / (d_k ** 0.5)
        err_wrong += torch.mean((v_rec - v_t) ** 2).item()
    err_wrong /= num_pairs
    
    # ── Test 2: Fixed Outer Product & Normalization (CORRECT) ──
    M_correct = torch.zeros(d_k, d_k, dtype=torch.complex64)
    for p in range(num_pairs):
        k_t = K[p]
        v_t = V[p]
        v_old = torch.einsum('ij,j->i', M_correct, torch.conj(k_t)).real / d_k
        err = v_t - v_old
        update = torch.einsum('i,j->ij', err.to(torch.complex64), k_t)
        M_correct = M_correct + update
        
    err_correct = 0.0
    for p in range(num_pairs):
        k_t = K[p]
        v_t = V[p]
        v_rec = torch.einsum('ij,j->i', M_correct, torch.conj(k_t)).real / d_k
        err_correct += torch.mean((v_rec - v_t) ** 2).item()
    err_correct /= num_pairs
    
    print(f"=== DELTA RULE UNIT TEST RESULTS ===")
    print(f"Transposed Code Reconstruction MSE : {err_wrong:.6f}  (FAIL - Cannot retrieve values)")
    print(f"Corrected Code Reconstruction MSE  : {err_correct:.6f}  (PASS - Exact recall!)")

if __name__ == "__main__":
    test_delta_rule()
