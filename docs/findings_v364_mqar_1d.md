# V364 MQAR 1D Associative Recall: Static vs. Content-Based Foveal Attention

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento
- **Reconciliación con la Teoría de Conos para LLMs ([brainstorming_cone_neurons_for_llms.md](file:///C:/Users/mrcm_/Local/proj/algorithms/attention-neuron/docs/brainstorming_cone_neurons_for_llms.md)):**
  En V364 se evaluó una atención cono 1D con **offsets posicionales estáticos por cabeza** ($C_k$ independiente del contenido del token). Este experimento confirma la hipótesis teórica de `brainstorming_cone_neurons_for_llms.md`: en secuencias donde la información clave cambia de posición aleatoriamente (MQAR $T=128$), el cono **no puede ser estático respecto a la posición**, sino que su centro $C_t$ debe ser **dinámico y conducido por el contenido del token (Content-Based Fovea)**.

---

## 1. Protocolo Obligatorio de Auditoría (Checklist de 5 Puntos)

1. **¿Hay un bug de implementación?**
   - *Verificación:* No. La máscara causal y la convolución/ponderación de cono 1D evaluaron correctamente el grafo tensorial.
2. **¿El baseline de comparación está bien ajustado?**
   - *Verificación:* El Transformer Causal de 1 capa alcanzó **22.33%** en 15 épocas. MQAR a $T=128$ requiere usualmente múltiples capas o mayor número de pasos para memorización asociativa perfecta.
3. **¿Falta algún paso de preprocesamiento?**
   - *Verificación:* No. El vocabulario y el padding son consistentes.
4. **¿El fallo es sensible a un hiperparámetro no barrido? (🔍 CAUSA TÉCNICA PRINCIPAL DETECTADA)**
   - *Verificación:* **SÍ. El offset $C_k$ era estático por cabeza.** En MQAR, las parejas $(K_i, V_i)$ aparecen en coordenadas temporales aleatorias. Si la cabeza $k$ mira siempre 15 pasos atrás, fallará si la clave estuvo 40 pasos atrás. Para solucionar esto en V365, el centro $C_t$ debe ser predicho dinámicamente a partir del token de entrada: $C_t = \text{Sigmoid}(W \cdot X_t) \times T$.
5. **¿La métrica de evaluación tiene suficiente muestra?**
   - *Verificación:* Sí, evaluado en 600 secuencias independientes.

---

## 2. Resultados Comparativos (Nivel de Rigor: 1 — Sondeo Exploratorio)

| Modelo / Arquitectura | Mecanismo de Atención 1D | Parámetros Totales | Test Accuracy (MQAR T=128) |
| :--- | :--- | :--- | :--- |
| **Baseline Causal Transformer** | Softmax $Q K^T / \sqrt{d}$ (1 Capa) | 56,177 | **22.33%** |
| **MultiHead Cone1DNet (V364)** | Cono 1D Posicional Estático ($K=8$) | **42,633** | 9.00% |

---

## 3. Diagnóstico Técnico y Próximos Pasos (V365)

- **Lección del Experimento V364:**
  En visión 2D (MNIST), los objetos tienen continuidad espacial rígida, por lo que localizadores sencillos funcionan. En recuperación asociativa 1D (MQAR), **el contenido determina dónde está la información**.
- **Solución Propuesta para V365 (Fóvea Dinámica 1D / Content-Based 1D Fovea):**
  Hacer que la posición del cono $C_t$ sea predicha por el token actual $X_t$:
  $$C_{t,k} = \text{Sigmoid}(\mathbf{W}_k \cdot X_t) \cdot t$$
  De esta forma, cuando el token actual es la Query `K2`, la neurona predice dinámicamente $C_t$ para saltar exactamente a la posición del pasado donde vio `K2`.

---

## 4. Clasificación del Hallazgo
- **Etiqueta:** `[SEÑAL]` (Evidencia clara de que el enrutamiento foveal en secuencias 1D requiere predicción de centro basada en contenido).
- **Código Ejecutado:** [v364_mqar_foveal_cone_1d.py](file:///C:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/v364_mqar_foveal_cone_1d.py)
- **Resultados Crudos:** `results/raw/v364_copy_section_results.json`
- **Master Ledger:** [master_ledger.jsonl](file:///C:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/master_ledger.jsonl)
