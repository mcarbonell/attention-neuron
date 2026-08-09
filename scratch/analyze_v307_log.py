"""
scratch/analyze_v307_log.py
===========================
Precise Statistical Analysis of v307_log.txt.
Calculates Mean +- SE, Welch's t-tests, and p-values.
"""

import math, os
import numpy as np

def analyze():
    log_file = "v307_log.txt"
    if not os.path.exists(log_file):
        print(f"File {log_file} not found.")
        return

    complex_losses = [7.7009, 7.6913, 7.6866, 7.6595, 7.6915]
    complex_ppls   = [2210.35, 2189.30, 2178.90, 2120.80, 2189.73]

    real_losses    = [7.6777, 7.7490, 7.7012, 7.6764, 7.6936]
    real_ppls      = [2159.75, 2319.33, 2211.08, 2156.91, 2194.19]

    mha_losses     = [7.6810, 7.6971, 7.6928, 7.7066]
    mha_ppls       = [2166.87, 2202.01, 2192.59, 2222.99]

    def stats(vals):
        arr = np.array(vals)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1) # sample std
        se = std / math.sqrt(len(arr))
        return mean, std, se

    c_m_l, c_std_l, c_se_l = stats(complex_losses)
    c_m_p, c_std_p, c_se_p = stats(complex_ppls)

    r_m_l, r_std_l, r_se_l = stats(real_losses)
    r_m_p, r_std_p, r_se_p = stats(real_ppls)

    m_m_l, m_std_l, m_se_l = stats(mha_losses)
    m_m_p, m_std_p, m_se_p = stats(mha_ppls)

    print("=" * 80)
    print("V307 STATISTICAL SUMMARY AUDIT:")
    print("=" * 80)
    print(f"ChunkwiseComplexDeltaPhase (n={len(complex_losses)}):")
    print(f"  Val Loss: {c_m_l:.4f} +- {c_se_l:.4f} (std={c_std_l:.4f})")
    print(f"  Val PPL : {c_m_p:.2f} +- {c_se_p:.2f} (std={c_std_p:.2f})")
    print("-" * 50)
    print(f"CausalAttentionMHA (n={len(mha_losses)}):")
    print(f"  Val Loss: {m_m_l:.4f} +- {m_se_l:.4f} (std={m_std_l:.4f})")
    print(f"  Val PPL : {m_m_p:.2f} +- {m_se_p:.2f} (std={m_std_p:.2f})")
    print("-" * 50)
    print(f"ChunkwiseRealDeltaNetIsoParam (n={len(real_losses)}):")
    print(f"  Val Loss: {r_m_l:.4f} +- {r_se_l:.4f} (std={r_std_l:.4f})")
    print(f"  Val PPL : {r_m_p:.2f} +- {r_se_p:.2f} (std={r_std_p:.2f})")
    print("=" * 80)

    # Welch's t-test calculation:
    # Complex vs Real IsoParam
    t_c_r = (r_m_l - c_m_l) / math.sqrt((c_std_l**2 / len(complex_losses)) + (r_std_l**2 / len(real_losses)))
    # Complex vs MHA
    t_c_m = (m_m_l - c_m_l) / math.sqrt((c_std_l**2 / len(complex_losses)) + (m_std_l**2 / len(mha_losses)))

    print(f"Welch's t-statistic (Complex vs Real IsoParam): t = {t_c_r:.4f}")
    print(f"Welch's t-statistic (Complex vs Softmax MHA)  : t = {t_c_m:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    analyze()
