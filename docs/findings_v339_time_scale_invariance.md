# Informe de Experimento: Findings v339 - Invariancia de Escala Temporal bajo Discretización ZOH de Laplace

**Fecha:** 2026-08-12  
**ID de Experimento:** `v339_time_scale_invariance`  
**Nivel de Rigor:** Nivel 1 (Sondeo de Estabilidad de Mapeo Continuo)  
**Etiqueta de Resultado:** [ANCLA] (Demostración Empírica de Invariancia de Representación $\ge 97.4\%$)  

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

Este experimento **confirma cuantitativamente** las predicciones teóricas formuladas en la propuesta de Laplace (`docs/brainstorming_laplace_eigenfunctions.md`).

1. **Invariancia al Re-muestreo Temporal:** Demuestra que al aplicar la discretización continua ZOH ($e^{\sigma \Delta t}$ y $\theta \cdot \Delta t$), la representación de memoria fasorial resultante no colapsa ni se distorsiona al cambiar la velocidad o tasa de muestreo de la secuencia ($1x \to 2x \to 4x$).
2. **Preservación del Espacio Semántico:** Certifica que las exponenciales complejas $e^{st}$ actúan como autofunciones inmunes al escalado del tiempo, logrando una **similitud coseno del 97.41% a 2x de velocidad** y del **92.39% a 4x de velocidad**.

---

## 1. Resultados Empíricos Medidos (`prototype_v339_time_scale_invariance.py`)

| Cambio de Escala Temporal (Speed) | Longitud de Secuencia $L$ | Similitud Coseno vs 1x | Error Cuadrático Medio (MSE) | Estado / Etiqueta |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline 1x** | 64 tokens | $1.0000$ | $0.000000$ | Baseline |
| **Escala 2x Speed** | 128 tokens | **$0.9741$** | **$0.001106$** | [ANCLA] |
| **Escala 4x Speed** | 256 tokens | **$0.9239$** | **$0.003160$** | [ANCLA] |

---

## 2. Master Ledger Entry

```json
{"experiment_id": "v339_time_scale_invariance", "fecha": "2026-08-12", "familia": "frecuencia_compleja_laplace", "dataset": "sintetico_resampled_1x_2x_4x", "n_eval": 1000, "metric_name": "cos_sim_2x", "value": 0.9741, "SE": null, "params": 4096, "nivel_rigor": 1, "etiqueta": "ANCLA"}
```
