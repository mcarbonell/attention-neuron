"""
test_strict_logical_and_v2.py
=============================
Strict Logical AND (Intersection) via Cleanup Threshold / Min Operator (Fuzzy Logic Intersection).
"""

import torch

def fuzzy_min_logical_and(M, Q1, Q2, inv_dk, threshold=0.5):
    """
    Fuzzy Min Logical AND:
    r1 = Readout(M, Q1)
    r2 = Readout(M, Q2)
    Mask = (r1 > threshold) & (r2 > threshold)
    return min(r1, r2) * Mask
    """
    r1 = torch.matmul(M, torch.conj(Q1).unsqueeze(-1)).squeeze(-1).real * inv_dk
    r2 = torch.matmul(M, torch.conj(Q2).unsqueeze(-1)).squeeze(-1).real * inv_dk
    
    mask = (r1 > threshold) & (r2 > threshold)
    return torch.minimum(r1, r2) * mask.float()

def run_test():
    torch.manual_seed(42)
    d_k = 32
    inv_dk = 1.0 / float(d_k)
    
    # Claves A, B
    theta_A = torch.randn(1, d_k)
    theta_B = torch.randn(1, d_k)
    
    A = torch.complex(torch.cos(theta_A), torch.sin(theta_A))
    B = torch.complex(torch.cos(theta_B), torch.sin(theta_B))
    
    V_target = torch.ones(1, d_k)
    
    # Memoria M1 con solo A
    M_A = torch.matmul(V_target.to(torch.complex64).unsqueeze(-1), A.unsqueeze(-2))
    # Memoria M2 con A y B (Ambos presentes)
    M_AB = torch.matmul(V_target.to(torch.complex64).unsqueeze(-1), A.unsqueeze(-2)) + \
           torch.matmul(V_target.to(torch.complex64).unsqueeze(-1), B.unsqueeze(-2))
           
    print("=" * 80)
    print("FUZZY MIN LOGICAL AND (INTERSECTION) WITH CLEANUP THRESHOLD")
    print("=" * 80)
    
    and_on_A_only = fuzzy_min_logical_and(M_A, A, B, inv_dk, threshold=0.5).sum().item()
    and_on_AB = fuzzy_min_logical_and(M_AB, A, B, inv_dk, threshold=0.5).sum().item()
    
    print(f"Strict AND (A & B) en Memoria con solo A: {and_on_A_only:.6f} (CERO ABSOLUTO)")
    print(f"Strict AND (A & B) en Memoria con A y B: {and_on_AB:.6f} (ACTIVACIÓN LIMPIA)")
    
    print("=" * 80)
    print(f"RESULTADO: {'PASADO CON ÉXITO [ANCLA]' if and_on_A_only == 0.0 and and_on_AB > 0.0 else 'FALLIDO'}")
    print("=" * 80)

if __name__ == "__main__":
    run_test()
