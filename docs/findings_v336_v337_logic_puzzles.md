# Informe de Experimento: Findings v336 & v337 - Puzles de Deducción Transitiva y Auditoría de Negación bajo Ruido

**Fecha:** 2026-08-12  
**IDs de Experimento:** `v336_logical_puzzle_benchmark` & `v337_instruction_negation_audit`  
**Nivel de Rigor:** Nivel 1 (Sondeo Exploratorio y Evaluación de Coherencia)  
**Etiqueta de Resultado:** [ANCLA] (Demostración Empírica de Coherencia de Coherencia de Fase $\ge 97.7\%$)  

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

Este experimento **amplía la serie `v334`/`v335`** hacia entornos de alta densidad de ruido y deducciones transitivas compuestas.

1. **Persistencia de Coherencia en Cadenas Profundas:** Muestra que la señal de fase no colapsa exponencialmente en saltos múltiples; retiene un **$97.76\%$ de coherencia tras 2 saltos** ($A \to B \to C$) y sostiene una norma de respuesta limpia incluso a 4 saltos ($A \to B \to C \to D \to E$).
2. **Invariancia del Operador NOT frente a 64 Distractores:** Demuestra que la cancelación por fase invertida ($\pi$) mantiene un ratio exacto de **$-1.0000$** incluso en presencia de 64 claves distractoras inyectadas en la misma matriz de memoria.

---

## 1. Experimento v336: Puzles de Deducción Transitiva (`prototype_v336_logical_puzzle_benchmark.py`)

### Resultados de Coherencia de Señal:

| Salto Lógico (Hop) | Cadena de Deducción | Norma de Salida | Coherencia Relativa (%) | Estado / Etiqueta |
| :--- | :--- | :---: | :---: | :---: |
| **Hop 1** | $A \to B$ | $5.4963$ | $100.0\%$ (Baseline) | [ANCLA] |
| **Hop 2** | $A \to B \to C$ | $5.3731$ | **$97.76\%$** | [ANCLA] |
| **Hop 4** | $A \to B \to C \to D \to E$ | $5.2606$ | **$95.71\%$** | [ANCLA] |

---

## 2. Experimento v337: Auditoría de Negación con 64 Distractores (`prototype_v337_instruction_negation_audit.py`)

### Resultados de Cancelación de Fase:

| Condición de Consulta | Norma de Respuesta | Ratio de Cancelación de Fase | Estado / Etiqueta |
| :--- | :---: | :---: | :---: |
| **Consulta Positiva $Q(A)$** | $3.6456$ | Baseline $+1.0000$ | [ANCLA] |
| **Consulta Negativa $\text{NOT}(Q(A))$** | $3.6456$ | **$-1.0000$** (Cancelación Exacta) | [ANCLA] |

---

## 3. Master Ledger Entries

```json
{"experiment_id": "v336_logical_puzzle_benchmark", "fecha": "2026-08-12", "familia": "logica_fase_transitiva", "dataset": "sintetico_puzle_deductivo", "n_eval": 1000, "metric_name": "coherence_ratio", "value": 0.9776, "SE": null, "params": 4096, "nivel_rigor": 1, "etiqueta": "ANCLA"}
{"experiment_id": "v337_instruction_negation_audit", "fecha": "2026-08-12", "familia": "logica_fase_negacion", "dataset": "sintetico_64_distractors", "n_eval": 1000, "metric_name": "cancellation_ratio", "value": -1.0, "SE": null, "params": 1024, "nivel_rigor": 1, "etiqueta": "ANCLA"}
```
