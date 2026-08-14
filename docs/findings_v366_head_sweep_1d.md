# V366 1D Foveal Head Sweep: Additive Summation vs. Competitive Head Gating

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento
- **Reconciliación con V363 (Visión 2D - [findings_v363_head_sweep.md](file:///C:/Users/mrcm_/Local/proj/algorithms/attention-neuron/docs/findings_v363_head_sweep.md)):**
  En visión 2D (V363), aumentar las cabezas $K=1 \to 256$ incrementó la precisión de $25.35\%$ a $84.90\%$ debido al uso de **Max-Pooling competitivo entre cabezas**. En V366 (1D MQAR), el escalado con **Suma Aditiva Directa de Cabezas** mostró una tendencia inversa ($K=2 \to 23.17\%$, $K=64 \to 16.17\%$), revelando que en secuencias 1D, la acumulación aditiva no compensada de múltiples cabezas diluye la señal del objetivo.

---

## 1. Resumen del Experimento (Nivel de Rigor: 1 — Sondeo Exploratorio)

Se evaluó el barrido de cabezas $K \in \{2, 4, 8, 16, 32, 64\}$ para la arquitectura `ContentBasedFovea1DNet` sobre el benchmark MQAR ($T=128$, 12 épocas):

- **Mecanismo de Mezcla de Cabezas:** Suma aditiva y proyección lineal conjunta ($D / K \to D$).
- **Muestreo:** Vectorizado en PyTorch mediante `torch.matmul` sobre matrices de pesos foveales $T \times T$.

---

## 2. Resultados Comparativos del Barrido 1D (K=2 a K=64)

| Cabezas ($K$) | Parámetros Totales | Test Acc (%) (Época 12) | Acc Pico (%) | PEI ($\text{Acc} / \log_{10}(\text{Params})$) | Tiempo / Época (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$K = 2$ 🌟** | **42,871** | **23.17%** | **23.17%** | **5.00** | 4.4s |
| **$K = 4$** | 43,133 | 20.17% | 20.83% | 4.35 | 4.8s |
| **$K = 8$** | 43,657 | 19.83% | 19.83% | 4.27 | 6.6s |
| **$K = 16$** | 44,705 | 17.17% | 17.17% | 3.69 | 9.7s |
| **$K = 32$** | 46,801 | 16.83% | 16.83% | 3.60 | 16.2s |
| **$K = 64$** | 50,993 | 16.17% | 16.17% | 3.43 | 34.5s |

> **Marcador 🌟:** Asignado a $K=2$ por obtener la máxima precisión final (23.17%) y el mejor PEI (5.00) bajo el esquema de suma aditiva.

---

## 3. Diagnóstico Técnico y Descubrimiento Arquitectónico

### El Problema del Ruido Aditivo en 1D
En secuencias de texto/MQAR, el valor correcto `Value` se encuentra en una única clave `Key` específica del pasado.
- Cuando $K=2$, las cabezas activas concentran la señal.
- Cuando $K=64$, las cabezas adicionales que no están enfocadas en la clave emiten ruido residual de fondo. Al **sumar linealmente las 64 cabezas**, el ruido sumado de 62 cabezas cancela la señal útil de las 2 cabezas correctas.

### Solución Arquitectónica para V367 (Gated / Max-Pooled 1D Fovea):
Para lograr en 1D el mismo escalado masivo que en 2D (V363):
1. **Max-Pooling sobre Cabezas ($\max_k \mathbf{v}_k$):** Selecciona el vector de características de la cabeza con mayor activación.
2. **Gating Softmax entre Cabezas ($\sum_k \alpha_k \mathbf{v}_k$):** Ponderar las $K$ cabezas mediante una distribución Softmax competitiva para filtrar el ruido de cabezas inactivas.

---

## 4. Clasificación del Hallazgo
- **Etiqueta:** `[ANCLA]` (Descubierto el requisito de selección competitiva/gating entre cabezas en 1D para evitar la dilución por suma aditiva).
- **Código Ejecutado:** [v366_content_fovea_1d_head_sweep.py](file:///C:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/v366_content_fovea_1d_head_sweep.py)
- **Resultados Crudos:** `results/raw/v366_copy_section_results.json`
- **Master Ledger:** [master_ledger.jsonl](file:///C:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/master_ledger.jsonl)
