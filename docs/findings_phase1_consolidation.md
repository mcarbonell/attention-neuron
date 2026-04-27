# Findings: Fase 1 - Consolidación de Arquitectura

Este documento registra los resultados experimentales de la Fase 1 del plan de investigación, cuyo objetivo era aislar los componentes esenciales de la Attention Neuron y definir un baseline dorado.

## 1. Metodología
Se utilizó un script de experimentación parametrizado (`scratch/experiment_phase1.py`) sobre el dataset MNIST.
- **Arquitectura**: MLP (784 $\rightarrow$ 512 $\rightarrow$ 10).
- **Sustrato**: Pesos aleatorios congelados (Kaiming Normal).
- **Optimización**: Adam, LR=0.01, 10 Epochs.
- **Métrica**: Accuracy en Test Set.

---

## 2. Análisis de Ablación (Rango $k=2$)
Se evaluó la contribución de cada componente de modulación manteniendo el rango constante.

| ID | Configuración | Acc (Mean) | $\Delta$ vs Full | Conclusión |
| :--- | :--- | :--- | :--- | :--- |
| **A1** | **Full (Mult $\checkmark$, Add $\checkmark$, Phase $\checkmark$)** | **0.9570** | - | Baseline de referencia |
| **A2** | Mult $\checkmark$, Add $\checkmark$, Phase $\times$ | 0.9566 | -0.0004 | El Phase Bias es marginal en precisión bruta. |
| **A3** | Mult $\times$, Add $\checkmark$, Phase $\checkmark$ | 0.9175 | **-0.0395** | **Crítico**. El gating multiplicativo es el motor principal. |
| **A4** | Mult $\checkmark$, Add $\times$, Phase $\checkmark$ | 0.9478 | -0.0092 | El término aditivo es importante para el refinamiento. |

**Insight**: La arquitectura es altamente dependiente de la modulación multiplicativa. Sin ella, la capacidad de la red cae drásticamente ($\sim 4\%$), validando la tesis de que el "gating" de cables es la operación fundamental.

---

## 3. Análisis de Rango y Saturación
Se evaluó el impacto del rango $k$ en la capacidad expresiva.

| Rango ($k$) | Acc (Mean) | Ganancia Marginal | Observación |
| :--- | :--- | :--- | :--- |
| 1 | 0.9398 | - | Capacidad base |
| 2 | 0.9570 | +0.0172 | Salto significativo |
| 4 | **0.9670** | +0.0100 | Punto dulce de eficiencia |
| 8 | 0.9688 | +0.0018 | Saturación / Retornos decrecientes |

**Insight**: Existe un punto de saturación rápido. Pasar de $k=4$ a $k=8$ solo aporta un $0.18\%$ de mejora, mientras que el salto de $k=1$ a $k=4$ es masivo.

---

## 4. Validación Estadística (A6)
Para eliminar el ruido de la inicialización, se ejecutaron 5 semillas independientes para los rangos más prometedores.

- **Rank 2**: $0.9575 \pm 0.0010$
- **Rank 4**: $0.9655 \pm 0.0015$

La baja desviación estándar confirma que los resultados son robustos y reproducibles.

---

## 5. Conclusión: El Baseline Dorado

Se define la siguiente configuración como el **Baseline Dorado** para todas las comparaciones futuras:

- **Rango**: $k=4$
- **Componentes**: Multiplicativo $\checkmark$, Aditivo $\checkmark$, Phase Bias $\checkmark$
- **Rendimiento Esperado**: $\sim 96.5\%$ en MNIST.

Este baseline representa el estado del arte de la arquitectura consolidada antes de pasar a la Fase 2 (Comparación Justa).
