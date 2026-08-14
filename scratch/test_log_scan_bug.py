import torch

def test_log_scan_equivalence():
    print("================================================================================")
    print(" AUDITORÍA NUMÉRICA DE EQUIVALENCIA: LOG-CUMSUM SCAN VS RECURRENCIA SECUENCIAL")
    print("================================================================================")
    
    # 1. Defino el log_scan exacto usado en PAIIR / v347
    def log_scan(alpha, beta):
        log_alpha = torch.log(torch.clamp(alpha, min=1e-5, max=1.0 - 1e-5))
        cum_log_alpha = torch.cumsum(log_alpha, dim=1)
        Lambda = torch.exp(cum_log_alpha)
        scaled_input = beta / (Lambda + 1e-6)
        cum_scaled_input = torch.cumsum(scaled_input, dim=1)
        return Lambda * cum_scaled_input

    for L in [128, 256, 512]:
        D = 128
        torch.manual_seed(42)
        
        # Alphas realistas: α ∈ [0.45, 0.95]
        alpha = torch.rand(1, L, D, dtype=torch.float32) * 0.5 + 0.45
        beta  = torch.randn(1, L, D, dtype=torch.float32)
        
        # Referencia FP64 secuencial (matemáticamente exacta)
        h = torch.zeros(1, D, dtype=torch.float64)
        ref_list = []
        for t in range(L):
            h = alpha[0, t].double() * h + beta[0, t].double()
            ref_list.append(h.clone())
        ref = torch.stack(ref_list, dim=0) # [L, D]
        
        # Salida del log_scan en FP32 (como en PAIIR / v347)
        out = log_scan(alpha, beta)[0] # [L, D]
        
        # Error relativo máximo
        abs_err = (out.double() - ref).abs()
        rel_err = abs_err.max() / ref.abs().max()
        
        print(f"Longitud L = {L:4d} | Max Absolute Error: {abs_err.max().item():.6e} | Max Relative Error: {rel_err.item():.6e}")
        
        # Análisis adicional: comprobación de cuántas posiciones tienen Lambda < 1e-6
        log_alpha = torch.log(torch.clamp(alpha, min=1e-5, max=1.0 - 1e-5))
        Lambda = torch.exp(torch.cumsum(log_alpha, dim=1))
        corrupted_pct = (Lambda < 1e-6).float().mean().item() * 100.0
        print(f"   --> Porcentaje de tensores corruptos por epsilon (Lambda < 1e-6): {corrupted_pct:.2f}%")
        print("-" * 80)

if __name__ == "__main__":
    test_log_scan_equivalence()
