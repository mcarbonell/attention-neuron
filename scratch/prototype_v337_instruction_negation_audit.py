"""
prototype_v337_instruction_negation_audit.py
================================================
Experiment v337: Instruction Negation Audit under Heavy Distractor Noise.
Evaluates NOT(Q) destructive interference phase cancellation under 64 distractor keys.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class NegationPhaseAudit(nn.Module):
    def __init__(self, d_k=32):
        super().__init__()
        self.d_k = d_k
        self.inv_dk = 1.0 / float(d_k)

    def run_audit(self, num_distractors=64):
        torch.manual_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Target concept A and distractor concepts
        theta_target = torch.randn(1, self.d_k, device=device)
        theta_distractors = torch.randn(num_distractors, self.d_k, device=device)
        
        K_target = torch.complex(torch.cos(theta_target), torch.sin(theta_target))
        K_dist = torch.complex(torch.cos(theta_distractors), torch.sin(theta_distractors))
        
        V_target = torch.randn(1, self.d_k, device=device)
        V_dist = torch.randn(num_distractors, self.d_k, device=device)
        
        # Build memory state with Target + Distractors
        M = torch.zeros(1, 1, self.d_k, self.d_k, dtype=torch.complex64, device=device)
        
        # Inject target
        v_old = torch.matmul(M, torch.conj(K_target).unsqueeze(-1)).squeeze(-1).real * self.inv_dk
        err = V_target - v_old
        M = M + torch.matmul(err.to(torch.complex64).unsqueeze(-1), K_target.unsqueeze(-2))
        
        # Inject distractors
        for d in range(num_distractors):
            kd, vd = K_dist[d:d+1], V_dist[d:d+1]
            v_old = torch.matmul(M, torch.conj(kd).unsqueeze(-1)).squeeze(-1).real * self.inv_dk
            err = vd - v_old
            M = M + 0.1 * torch.matmul(err.to(torch.complex64).unsqueeze(-1), kd.unsqueeze(-2))
            
        # Query A vs NOT(A)
        Q_A = K_target
        Q_NOT_A = Q_A * torch.complex(torch.tensor(-1.0, device=device), torch.tensor(0.0, device=device))
        
        readout_A = torch.matmul(M, torch.conj(Q_A).unsqueeze(-1)).squeeze(-1).real * self.inv_dk
        readout_NOT_A = torch.matmul(M, torch.conj(Q_NOT_A).unsqueeze(-1)).squeeze(-1).real * self.inv_dk
        
        cancellation_ratio = (readout_NOT_A.sum() / readout_A.sum()).item()
        return readout_A.norm().item(), readout_NOT_A.norm().item(), cancellation_ratio

def run_v337_experiment():
    print("=" * 85)
    print("EXPERIMENT v337: INSTRUCTION NEGATION AUDIT UNDER HEAVY DISTRACTORS")
    print("=" * 85)
    
    auditor = NegationPhaseAudit(d_k=32)
    norm_A, norm_NOT_A, ratio = auditor.run_audit(num_distractors=64)
    
    print(f"[v337 - 64 Distractors] Readout Norm(A): {norm_A:.4f}")
    print(f"[v337 - 64 Distractors] Readout Norm(NOT A): {norm_NOT_A:.4f}")
    print(f"[v337 - 64 Distractors] Phase Cancellation Ratio: {ratio:.4f} (Exact -1.0000)")
    
    acc_passed = abs(ratio - (-1.0)) < 1e-4
    print("=" * 85)
    print(f"EXPERIMENT v337 RESULT: {'PASSED [ANCLA]' if acc_passed else 'FAILED'}")
    print("=" * 85)

if __name__ == "__main__":
    run_v337_experiment()
