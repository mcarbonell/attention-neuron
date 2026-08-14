"""
prototype_v334_logic_phase_ops.py
==================================
Experiment v334: LogicPhase Memory - Differentiable Phasor Symbolic Operators (BIND, UNBIND, NOT, AND).
Evaluates exact phase-space logic operations in S^1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LogicPhaseCore(nn.Module):
    def __init__(self, d_k=32):
        super().__init__()
        self.d_k = d_k
        self.inv_dk = 1.0 / float(d_k)

    def encode_phasor(self, theta):
        """Maps angle tensor theta to unit circle S^1 in complex space"""
        return torch.complex(torch.cos(theta), torch.sin(theta))

    def bind(self, K, V):
        """BIND(K, V) -> Phasor Hadamard product K * V (Angle Addition)"""
        V_complex = V.to(torch.complex64) if not V.is_complex() else V
        return K * V_complex

    def unbind(self, K, M_bind):
        """UNBIND(K, M) -> Conjugate multiplication conj(K) * M (Angle Subtraction)"""
        return (torch.conj(K) * M_bind).real

    def not_op(self, Q):
        """NOT(Q) -> Phase Shift by pi radians (e^{i pi} = -1)"""
        return Q * torch.complex(torch.tensor(-1.0), torch.tensor(0.0))

    def and_op(self, Q1, Q2):
        """AND(Q1, Q2) -> Superposition and Phase Normalization"""
        superpos = Q1 + Q2
        mags = torch.abs(superpos) + 1e-8
        return superpos / mags

def run_v334_experiment():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    d_k = 32
    core = LogicPhaseCore(d_k=d_k)
    
    print("=" * 85)
    print("EXPERIMENT v334: LOGICPHASE SYMBOLIC OPERATORS IN S^1 PHASE SPACE")
    print("=" * 85)
    
    # 1. Test BIND & UNBIND (Exact Retrieval)
    theta_K = torch.randn(10, d_k, device=device)
    theta_V = torch.randn(10, d_k, device=device)
    K = core.encode_phasor(theta_K)
    V = core.encode_phasor(theta_V)
    
    M_bind = core.bind(K, V)
    V_recovered = core.unbind(K, M_bind)
    target_V_real = V.real
    
    unbind_error = (V_recovered - target_V_real).abs().max().item()
    print(f"[v334 - UNBIND] Exact Phase Recovery Max Absolute Error: {unbind_error:.6e}")
    
    # 2. Test NOT Operator (Destructive Interference)
    Q_A = K[0:1]
    Q_NOT_A = core.not_op(Q_A)
    
    # Readout of Q_A vs NOT(Q_A) against memory containing A
    readout_A = (torch.conj(Q_A) * M_bind[0:1]).real.sum()
    readout_NOT_A = (torch.conj(Q_NOT_A) * M_bind[0:1]).real.sum()
    
    print(f"[v334 - NOT]   Readout(A): {readout_A.item():.4f} | Readout(NOT A): {readout_NOT_A.item():.4f}")
    print(f"[v334 - NOT]   Phase Inversion Cancellation Ratio: {(readout_NOT_A / readout_A).item():.4f} (Exact -1.0000)")
    
    # 3. Test AND Operator (Constructive Interference)
    Q_1 = K[0:1]
    Q_2 = K[1:2]
    Q_AND = core.and_op(Q_1, Q_2)
    
    readout_AND_1 = (torch.conj(Q_AND) * M_bind[0:1]).real.sum().item()
    readout_AND_2 = (torch.conj(Q_AND) * M_bind[1:2]).real.sum().item()
    readout_AND_unrelated = (torch.conj(Q_AND) * M_bind[5:6]).real.sum().item()
    
    print(f"[v334 - AND]   Readout(A AND B on A): {readout_AND_1:.4f} | Readout(A AND B on Unrelated): {readout_AND_unrelated:.4f}")
    
    acc_success = unbind_error < 1e-5 and (readout_NOT_A / readout_A).item() == -1.0
    print("=" * 85)
    print(f"EXPERIMENT v334 RESULT: {'PASSED [ANCLA]' if acc_success else 'FAILED'}")
    print("=" * 85)

if __name__ == "__main__":
    run_v334_experiment()
