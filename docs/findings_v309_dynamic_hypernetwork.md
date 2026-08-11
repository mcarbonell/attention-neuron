# Hallazgos Experimento v309: Dynamic Low-Rank Hypernetwork (Fase 2)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

* **Conclusión Previa Modificada:** En el plan de investigación (`RESEARCH_LINE_DYNAMIC_LOW_RANK.md`), se hipotetizó que una Hypernetwork que proyecte $A(x) \in \mathbb{R}^{r \times d_{in}}$ y $B(x) \in \mathbb{R}^{d_{out} \times r}$ en tiempo real proporcionaría una rotación dinámica de subespacios superior a LoRA estático.
* **Resultado del Experimento v309:** La Hypernetwork directa sufrió una **explosión paramétrica masiva (1,098,560 parámetros vs 58,176 en LoRA estático)** y una ralentización computacional de casi 10 veces (32.80s vs 3.91s), logrando una loss final idéntica/marginalmente inferior (3.4864) a la de LoRA estático (3.4721) y Denso estándar (3.4773).
* **Demostración de Refutación (Hallazgo [ANCLA-NEGATIVO]):**
  * **Explosión Paramétrica:** Proyectar la matriz $A(x)$ con una capa lineal directa $W_{proj\_A} \in \mathbb{R}^{(r \cdot d_{in}) \times d_{in}}$ introduce $r \cdot d_{in}^2$ parámetros por matriz. Si $r=16, d_{in}=128$, la Hypernetwork añade $262,144$ parámetros por proyectores, lo que destruye el propósito del bajo rango (que era tener $O(r \cdot d)$ parámetros).
  * **Incongruencia Computacional:** La Hypernetwork simple requiere $2 \cdot r \cdot d_{in}^2$ parámetros extras, superando con creces el tamaño de la propia matriz densa $W_0$ ($d_{out} \cdot d_{in} = 16,384$).

---

## 1. Tabla de Resultados Empíricos

* **Configuración:** $N=1000$ secuencias estructuradas, $L=64$, $d_{model}=128$, $r=16$, 10 épocas, AdamW ($lr=1e-3$). Evaluado en CPU (8 hilos).

| Modelo | Parámetros | Loss Final | Wall Clock (s) | Internal Overhead (s) | PEI | Etiqueta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`static_lora`** 🌟 | **58,176** | **3.4721** | **3.91** | **0.16** | **0.0604** | [ANCLA] |
| **`standard_dense`** | 49,984 | 3.4773 | 2.82 | 0.15 | 0.0612 | [ANCLA] |
| **`dynamic_hypernetwork`** (v309) | 1,098,560 | 3.4864 | 32.80 | 2.21 | 0.0475 | [ANCLA-NEGATIVO] |

*Nota: El marcador 🌟 se asigna estrictamente al menor valor numérico real de Loss (3.4721).*

---

## 2. Análisis del Desempeño y Coste

1. **Rendimiento de Capacidad:** A pesar de contar con 1.09M parámetros (20 veces más que LoRA estático), la Hypernetwork directa logró una loss de 3.4864 frente a 3.4721 de LoRA. La sobreparametrización en los proyectores no se tradujo en una mejor extracción de características asociativas.
2. **Coste Computacional (Wall Clock Time):** La evaluación tardó 32.80s en `dynamic_hypernetwork` comparado con 3.91s en `static_lora`. El cálculo por token de los tensores de proyección $A(x)$ y $B(x)$ añade una penalización del 738% en tiempo de ejecución.

---

## 3. Amenazas a la Validez

1. **Ausencia de Bottleneck en la Hypernetwork:** Se proyectó de $d_{in} \to (r \cdot d_{in})$ directamente en un solo paso lineal. Sin un cuello de botella interno ($d_{in} \to r_{hyper} \to r \cdot d_{in}$), la Hypernetwork no está regularizada.
2. **Alternativa Recomendada (Fase 3 - MoLoRA):** Para obtener adaptación dinámica sin la sobreparametrización $r \cdot d_{in}^2$, se debe utilizar una Mezcla de Adaptadores de Bajo Rango con Ruteo Dinámico (**Dynamic Gated LoRA / MoLoRA**), donde los adaptadores son fijos y solo el vector de ruteo $g(x) \in \mathbb{R}^K$ es dinámico por token.

---

## Auditoría posterior y amenazas a la validez (2026-08-10)

La regla sintética depende de $x_{t-1}$, pero este modelo es tokenwise. Condicionado a $x_t$ quedan 32 respuestas equiprobables, con suelo $\ln32\approx3.4657$; las losses ~3.47 están cerca de ese límite. Sin validación retenida/multisemilla, las diferencias de milésimas no prueban superioridad de adaptadores. Véase la [auditoría transversal v300–v329](findings_v300_v329_audit.md).
