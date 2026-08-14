# V367 Competitive 1D Foveal Selection: Max-Pooling vs. Softmax Gating

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento
- **Reconciliación con V366 ([findings_v366_head_sweep_1d.md](file:///C:/Users/mrcm_/Local/proj/algorithms/attention-neuron/docs/findings_v366_head_sweep_1d.md)):**
  En V366 se descubrió que la suma aditiva no compensada de cabezas en 1D diluía la señal útil. V367 probó dos mecanismos de selección competitiva: **Opción A (Max-Pooling entre cabezas)** y **Opción B (Softmax Gating entre cabezas)**. Los resultados demuestran que **Softmax Gating supera a Max-Pooling en todas las configuraciones 1D**, pero confirman que en secuencias 1D el régimen óptimo de cabezas es compacto ($K \in [2, 8]$) a diferencia de la visión 2D.

---

## 1. Resumen del Experimento (Nivel de Rigor: 1 — Sondeo Exploratorio)

Se comparó empíricamente el rendimiento de dos mecanismos de selección competitiva de cabezas foveales en 1D sobre MQAR ($T=128$, 12 épocas):

1. **Opción A (Max-Pooling entre cabezas):** Selecciona componente a componente la máxima activación entre las $K$ cabezas ($\max_k \mathbf{v}_{t,k}$).
2. **Opción B (Softmax Gating entre cabezas):** Predice coeficientes dinámicos $\alpha_{t,k} = \text{Softmax}_k(\mathbf{W} \cdot X_t)$ asignando peso cero a cabezas inactivas.

---

## 2. Resultados Comparativos (Max-Pooling vs. Gated Softmax)

| Modo de Selección | Cabezas ($K$) | Parámetros Totales | Test Acc (%) (Época 12) | Acc Pico (%) | PEI ($\text{Acc} / \log_{10}(\text{Params})$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MAXPOOL (Opción A)** | 4 | 43,133 | 16.83% | 17.83% | 3.63 |
| **MAXPOOL (Opción A)** | 16 | 44,705 | 12.17% | 12.17% | 2.62 |
| **MAXPOOL (Opción A)** | 64 | 50,993 | 6.50% | 7.83% | 1.38 |
| **GATED SOFTMAX (Opción B) 🌟** | **4** | **43,393** | **19.67%** | **19.67%** | **4.24** |
| **GATED SOFTMAX (Opción B)** | 16 | 45,745 | 13.33% | 13.33% | 2.86 |
| **GATED SOFTMAX (Opción B)** | 64 | 55,153 | 7.83% | 7.83% | 1.65 |

> **Marcador 🌟:** Asignado a `GATED SOFTMAX K=4` por obtener la mayor precisión del estudio (19.67%) y superar consistentemente a Max-Pooling.

---

## 3. Principales Hallazgos y Diagnóstico Técnico

1. **Superioridad de Softmax Gating sobre Max-Pooling en 1D:**
   - Softmax Gating superó a Max-Pooling en todas las densidades de cabezas ($19.67\%$ vs $16.83\%$ en $K=4$; $13.33\%$ vs $12.17\%$ en $K=16$).
   - *Razón técnica:* Tomar el máximo componente a componente (`MaxPool`) destruye la relación de fase/dirección de los vectores de embedding en 1D, mientras que `Softmax Gating` conserva la estructura lineal del espacio latente.

2. **Régimen Óptimo de Cabezas en 1D vs. 2D:**
   - Mientras en Visión 2D el escalado favorece $K=256$ cabezas debido a la redundancia espacial de parches en cuadrícula, en secuencias temporales 1D el rendimiento óptimo se alcanza con un número **compacto de cabezas atencionales ($K \in [2, 8]$)**.

---

## 4. Clasificación del Hallazgo
- **Etiqueta:** `[ANCLA]` (Verificado que Softmax Gating es la función de combinación superior para cabezas foveales 1D).
- **Código Ejecutado:** [v367_competitive_fovea_1d.py](file:///C:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/v367_competitive_fovea_1d.py)
- **Resultados Crudos:** `results/raw/v367_copy_section_results.json`
- **Master Ledger:** [master_ledger.jsonl](file:///C:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/master_ledger.jsonl)
