# Informe de Experimento: Findings v338 - Delta-Laplace Phase Memory Core ($s = \sigma + i\theta$)

**Fecha:** 2026-08-12  
**ID de Experimento:** `v338_laplace_core`  
**Nivel de Rigor:** Nivel 1 (Verificación Algebraica y Prueba de Gradcheck)  
**Etiqueta de Resultado:** [ANCLA] (Demostración de Gradcheck PASSED y Estabilidad en $L=1024$)  

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

Este experimento **extiende la memoria de fase unimodular en $S^1$ (`v298`/`v334`) al plano complejo s de Laplace completo ($s = \sigma + i\theta$)**.

1. **Unificación de Fase y Decaimiento Continuo:** Modifica el modelo de retención escalar separada al unificar la atenuación disipativa $\sigma_t \le 0$ y la rotación angular $\theta_t$ en la autofunción universal del plano de Laplace $K_t = e^{\sigma_t + i\theta_t}$.
2. **Garantía Física de Estabilidad de Hurwitz:** Demuestra que al acotar $\text{Re}(s) = \sigma \le 0$ mediante $-\text{Softplus}(W_\sigma x)$, la norma del estado de memoria en secuencias largas ($L=1024$) permanece acotada ($278.70$) sin sufrir desbordamiento o explosión de gradientes.

---

## 1. Resultados Empíricos Medidos (`prototype_v338_laplace_core.py`)

| Métrica / Prueba | Configuración Evaluada | Resultado Numérico | Estado / Etiqueta |
| :--- | :--- | :---: | :---: |
| **FP64 Autograd Gradcheck** | `torch.autograd.gradcheck` en FP64 | **PASSED (`True`)** | [ANCLA] |
| **Comprobación NaN/Inf** | $L=1024$ tokens, batch size 2 | **CLEAN (0 NaNs / 0 Infs)** | [ANCLA] |
| **Norma Máxima de Estado** | $M \in \mathbb{C}^{16 \times 16}$ en $L=1024$ | **$278.7031$** (Estable) | [ANCLA] |
| **Norma Final de Estado** | $t=1024$ | **$278.1293$** (Estabilidad de Hurwitz) | [ANCLA] |

---

## 2. Master Ledger Entry

```json
{"experiment_id": "v338_laplace_core", "fecha": "2026-08-12", "familia": "frecuencia_compleja_laplace", "dataset": "sintetico_laplace_l1024", "n_eval": 1024, "metric_name": "gradcheck_passed", "value": 1.0, "SE": null, "params": 4096, "nivel_rigor": 1, "etiqueta": "ANCLA"}
```
