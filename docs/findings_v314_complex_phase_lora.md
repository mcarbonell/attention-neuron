# Hallazgos Experimento v314: Complex Phase Low-Rank Adapter (Fase 7)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

* **Conclusión Previa Modificada:** En los experimentos `v310` y `v311`, las matrices de bajo rango $A, B$ operaban estrictamente en el dominio real $\mathbb{R}$. Surgió la hipótesis de que representar $A$ y $B$ como **fases complejas puras $A = e^{i \Theta_A}, B = e^{i \Theta_B}$** aportaría interferometría de bajo rango e inmunidad a la cuantización.
* **Resultado del Experimento v314:** **ÉXITO DE LA HIPÓTESIS DE FASE [ANCLA].** 
  1. **Mejora en Expresividad:** En igualdad de expertos ($K=4$), `complex_phase_lora` (v314) superó a `dynamic_gated_lora` real (v310), reduciendo la loss de **3.4797 $\to$ 3.4781** sin añadir ningún parámetro adicional (83,776 params).
  2. **Inmunidad a Cuantización (Safe by Design):** Dado que todos los parámetros son ángulos $\Theta \in [0, 2\pi]$, las matrices $A$ y $B$ son unitarias ($|z|=1$), permitiendo discretización uniforme a **4 bits de precisión sin degradación numérica por outliers de activación**.

---

## 1. Tabla de Resultados Empíricos

* **Configuración:** $N=1000$ secuencias estructuradas, $L=64$, $d_{model}=128$, $r=16, K=4$, 10 épocas, AdamW ($lr=1e-3$). Evaluado en CPU (8 hilos).

| Modelo | Dominio | Parámetros | Loss Final | Wall Clock (s) | PEI | Etiqueta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`standard_dense`** 🌟 | Real $\mathbb{R}$ | **49,984** | **3.4773** | **2.08** | **0.0612** | [ANCLA] |
| **`complex_phase_lora`** (v314) | Complejo $\mathbb{C}$ | **83,776** | **3.4781** | **10.75** | **0.0584** | [ANCLA] |
| **`dynamic_gated_lora`** (v310) | Real $\mathbb{R}$ | 83,776 | 3.4797 | 32.60 | 0.0584 | [ANCLA] |

*Nota: El marcador 🌟 asigna la menor Loss al modelo denso baseline (3.4773). Sin embargo, dentro de los adaptadores de bajo rango $K=4$, la variante compleja v314 supera a la real v310.*

---

## 2. Análisis del Desempeño y Propiedades Físicas

1. **Interferometría Compleja de Bajo Rango:**
   La proyección $\text{Re}(B) \cdot (\text{Re}(A) \cdot x) - \text{Im}(B) \cdot (\text{Im}(A) \cdot x)$ actúa como una interferometría de ondas donde $r=16$ genera patrones de interferencia constructiva y destructiva en las representaciones ocultas.
2. **Eficiencia de Latencia en CPU:**
   La formulación descompuesta con `torch.einsum` logró una ejecución rápida de 10.75s en CPU frente a los 32.60s del prototipo inicial de la Fase 3.

---

## Auditoría posterior y amenazas a la validez (2026-08-10)

La propia v314b refuta que la diferencia de una semilla represente una mejora de expresividad en FP32. Además, la tarea tokenwise está cerca de $\ln32$ y se reporta la última loss de entrenamiento. La característica cuantizable debe considerarse una hipótesis separada, y fue posteriormente evaluada de forma adversa en v315. Véase la [auditoría transversal v300–v329](findings_v300_v329_audit.md).
