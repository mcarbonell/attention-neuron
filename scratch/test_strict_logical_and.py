"""
test_strict_logical_and.py
==========================
Experimenting with Strict Logical AND (Intersection) via Elementwise Gated Activation Product v1 * v2.
"""

import torch

def strict_logical_and_readout(M, Q1, Q2, inv_dk):
    """
    Strict Logical AND via Activation Intersection (Hadamard Gate):
    Readout_1 = Re(M * conj(Q1))
    Readout_2 = Re(M * conj(Q2))
    Strict_AND_Readout = ReLU(Readout_1) * ReLU(Readout_2)
    """
    r1 = torch.relu(torch.matmul(M, torch.conj(Q1).unsqueeze(-1)).squeeze(-1).real * inv_dk)
    r2 = torch.relu(torch.matmul(M, torch.conj(Q2).unsqueeze(-1)).squeeze(-1).real * inv_dk)
    return r1 * r2

def run_test():
    torch.manual_seed(42)
    d_k = 32
    inv_dk = 1.0 / float(d_k)
    
    # Claves A, B, C
    theta_A = torch.randn(1, d_k)
    theta_B = torch.randn(1, d_k)
    theta_C = torch.randn(1, d_k)
    
    A = torch.complex(torch.cos(theta_A), torch.sin(theta_A))
    B = torch.complex(torch.cos(theta_B), torch.sin(theta_B))
    C = torch.complex(torch.cos(theta_C), torch.sin(theta_C))
    
    V_target = torch.ones(1, d_k)
    
    # Memoria M1 con solo A
    M_A = torch.matmul(V_target.to(torch.complex64).unsqueeze(-1), A.unsqueeze(-2))
    # Memoria M2 con A y B (Ambos presentes)
    M_AB = torch.matmul(V_target.to(torch.complex64).unsqueeze(-1), A.unsqueeze(-2)) + \
           torch.matmul(V_target.to(torch.complex64).unsqueeze(-1), B.unsqueeze(-2))
           
    print("=" * 80)
    print("STRICT LOGICAL AND (INTERSECTION) TEST VIA HADAMARD ACTIVATION GATE")
    print("=" * 80)
    
    # Consulta (A AND B) sobre memoria M_A (Solo A presente)
    and_on_A_only = strict_logical_and_readout(M_A, A, B, inv_dk).sum().item()
    
    # Consulta (A AND B) sobre memoria M_AB (Ambos A y B presentes)
    and_on_AB = strict_logical_and_readout(M_AB, A, B, inv_dk).sum().item()
    
    print(f"Strict AND (A & B) en Memoria con solo A: {and_on_A_only:.6f} (CERO Absoluto)")
    print(f"Strict AND (A & B) en Memoria con A y B: {and_on_AB:.6f} (Activación Coherente)")
    
    print("=" * 80)
    print(f"RESULTADO: {'PASADO CON ÉXITO' if and_on_A_only == 0.0 and and_on_AB > 0.0 else 'FALLIDO'}")
    print("=" * 80)

if __name__ == "__main__":
    run_test()
