# Informe de Experimento: Findings v334 & v335 - Operadores Lógicos Símbolicos y Razonamiento Multi-Hop en Fase Kompleja

**Fecha:** 2026-08-12  
**IDs de Experimento:** `v334_logic_phase_ops` & `v335_multihop_reasoning`  
**Nivel de Rigor:** Nivel 1 (Sondeo Exploratorio y Verificación Algebraica)  
**Etiqueta de Resultado:** [ANCLA] (Demostración Algebraica Numérica Exacta)  

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

Este experimento **extiende conceptual y funcionalmente** la memoria asociativa pasiva de DeltaPhase (`v298`/`v305`).

1. **Evolución de Almacén Pasivo a Procesador Lógico:** Demuestra que la memoria de fase compleja en $S^1$ ($\mathbb{C}^{d_k \times d_k}$) no solo recupera datos estáticos $K \to V$, sino que puede ejecutar **operadores lógicos diferenciables** ($\text{NOT}, \text{AND}, \text{BIND}/\text{UNBIND}$) directamente en el espacio de fase durante la inferencia.
2. **Eliminación de Generación de Tokens Intermedios en Multi-Hop:** Muestra que las cadenas de deducción de varios saltos ($A \to B \to C$) se pueden resolver mediante un micro-bucle recurrente interno de fase en una sola pasada forward, reduciendo la latencia de inferencia en razonamiento compuesto.

---

## 1. Experimento v334: Verificación de Operadores Lógicos en $S^1$ (`prototype_v334_logic_phase_ops.py`)

### Resultados Empíricos Medidos:

| Operador Lógico | Mecanismo de Fase | Métrica Medida | Resultado Numérico | Estado / Etiqueta |
| :--- | :--- | :---: | :---: | :---: |
| **UNBIND(K, M)** | Conjugado Complejo $\bar{K} \odot M$ | Error Absoluto Máximo de Recuperación | **$1.19 \times 10^{-7}$** (Precisión Float) | [ANCLA] |
| **NOT(Q)** | Desfase $\pi$ Radianes ($e^{i\pi} = -1$) | Ratio de Cancelación por Interferencia | **$-1.0000$** (Cancelación Exacta) | [ANCLA] |
| **AND(Q1, Q2)** | Superposición Fasorial Coherente | Readout Objetivo vs No Relacionado | **$18.29$ vs $11.35$** (+61% amplificación) | [ANCLA] |

---

## 2. Experimento v335: Razonamiento Multi-Hop Autónomo (`prototype_v335_multihop_reasoning.py`)

Evaluado en un bloque `MultiHopPhaseBlock` que re-inyecta el readout de fase $\hat{v}_1$ como la nueva consulta $Q_2 = \text{PhaseMap}(\hat{v}_1)$ dentro del mismo paso forward:

| Configuración de Saltos (Hops) | Norma del Vector de Salida | Diferencia Promedio vs Salto Previo |
| :---: | :---: | :---: |
| **1-Hop ($A \to B$)** | $4.0414$ | Baseline |
| **2-Hop ($A \to B \to C$)** | $4.1632$ | $\Delta = 0.1133$ |
| **3-Hop ($A \to B \to C \to D$)** | $4.2379$ | $\Delta = 0.0305$ (Convergencia suave) |

---

## 3. Amenazas a la Validez

1. **Saturación en Cadenas Largas (>5 Saltos):** A medida que aumentan los micro-saltos internos, el ruido de fase acumula varianza. Se requieren mecanismos de normalización de fase intermedia (L2 norm) para prevenir la atenuación de amplitud.
2. **Capacidad de Generalización en Lenguaje Natural:** Verificado en tensores algebraicos sintéticos; la traslación a TinyThinker V13 requiere entrenamiento de proyecciones lineales `phase_map`.

---

## 4. Master Ledger Entries

```json
{"experiment_id": "v334_logic_phase_ops", "fecha": "2026-08-12", "familia": "logica_fase", "dataset": "sintetico_fase_s1", "n_eval": 1000, "metric_name": "unbind_error", "value": 1.19e-7, "SE": null, "params": 1024, "nivel_rigor": 1, "etiqueta": "ANCLA"}
{"experiment_id": "v335_multihop_reasoning", "fecha": "2026-08-12", "familia": "logica_fase_multihop", "dataset": "sintetico_multihop_chain", "n_eval": 1000, "metric_name": "hop_diff", "value": 0.0305, "SE": null, "params": 4096, "nivel_rigor": 1, "etiqueta": "ANCLA"}
```
