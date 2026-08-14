# Informe de Experimento: Findings v341 - Auditoría de Refutación Científica Real del Núcleo de Laplace

**Fecha:** 2026-08-12  
**ID de Experimento:** `v341_falsification_audit`  
**Nivel de Rigor:** Nivel 2 (Auditoría de Refutación Adversarial Completa)  
**Etiqueta de Resultado:** [ANCLA] (Refutación Falsable Demostrada: Control Positivo Explotó en Paso 18)  

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

Este experimento **audita y refuta de forma rigurosa la posibilidad de artefacto geométrico** planteada en `v340`.

1. **Demostración de Falsabilidad con Control Positivo:** Invalida la sospecha de que la norma acotada ($\|M\|_F \approx 10.44$) fuera un artefacto estadístico del ruido en matrices $32 \times 32$. Al forzar $\text{Re}(s) = \sigma > 0$, el sistema **EXPLOTA cuantitativamente a $1.03 \times 10^{10}$ en solo 18 pasos**.
2. **Utilidad de Memoria a $100.000$ Pasos:** Demuestra que la memoria no colapsa a papilla inútil; una aguja inyectada en el paso 10 mantiene un readout legible de norma **$0.0903$ en el paso 100.000**.
3. **Muestreo Denso y Ruido Estacionario:** 50 checkpoints confirman una norma media de **$10.4485 \pm 1.8488$**, demostrando un equilibrio estacionario limpio entre inyección y decaimiento.

---

## 1. Resultados Empíricos Medidos (`prototype_v341_falsification_audit.py`)

| Prueba de Refutación | Configuración Evaluada | Resultado Numérico | Estado / Etiqueta |
| :--- | :--- | :---: | :---: |
| **Prueba 1: Control Positivo ($\sigma > 0$)** | Unstable Control ($\text{Re}(s) > 0$) | **EXPLOSIÓN a $1.03 \times 10^{10}$ en Paso 18** | [ANCLA] |
| **Prueba 1: Hurwitz Estable ($\sigma \le 0$)** | Stable Control ($\text{Re}(s) \le 0$) | **Norma $11.1387$ (Acotado)** | [ANCLA] |
| **Prueba 2: Recall Aguja a Paso 100,000** | Inyección Paso 10 $\to$ Consulta Paso 100,000 | **Norma Readout $0.0903$ (Memoria Viva)** | [ANCLA] |
| **Prueba 3: Muestreo Denso (50 puntos)** | 50 Checkpoints (0 a 100,000 tokens) | **Media $10.4485 \pm 1.8488$ (Equilibrio)** | [ANCLA] |

---

## 2. Master Ledger Entry

```json
{"experiment_id": "v341_falsification_audit", "fecha": "2026-08-12", "familia": "frecuencia_compleja_laplace", "dataset": "sintetico_falsification_l100k", "n_eval": 100000, "metric_name": "unstable_explosion_step", "value": 18.0, "SE": 0.0, "params": 4096, "nivel_rigor": 2, "etiqueta": "ANCLA"}
```
