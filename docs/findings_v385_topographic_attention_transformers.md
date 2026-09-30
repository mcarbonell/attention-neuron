# Findings v385: Topographic Attention en Transformers Autorregresivos

**Fecha:** 2026-10-01  
**Lab:** `attention-neuron/`  
**Script:** [`scratch/prototype_v385_topographic_attention_transformers.py`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/prototype_v385_topographic_attention_transformers.py)  
**Registro Crudo:** [`results/raw/v385_topographic_attention.json`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/raw/v385_topographic_attention.json)  
**Nivel de Rigor:** Nivel 1 (Sondeo Exploratorio multi-semilla, 3 semillas por condición)  
**Etiqueta:** [SEÑAL] (Validación de atención topográfica y compresión 10x de cabezales en Transformers)  

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

En `v380` a `v384`, todos los experimentos de Regularización Celular Topográfica (TCR) se desarrollaron sobre perceptrones multicapa (MLPs) aplicados a tareas de visión sintética o estática (MNIST, Fashion-MNIST). Esto dejaba sin resolver la duda crítica de si el acoplamiento espacial continuo en los pesos era compatible con los mecanismos de enrutamiento dinámico de atención causal ($W_q, W_k, W_v, W_o$) en Transformers autorregresivos sobre lenguaje natural.

**Reconciliación y Nuevos Hallazgos de `v385`:**
1. **Compatibilidad Plena con la Atención Causal:** Aplicar tensión superficial Dirichlet 2D a las proyecciones de atención no desestabiliza el entrenamiento autorregresivo ni colapsa el gradiente. Con acoplamiento suave ($\epsilon = 5\times 10^{-4}$), la perplejidad de validación en Tiny Shakespeare se mantiene en $7.48 \pm 0.09$ frente a $6.91 \pm 0.12$ del baseline estándar (un coste contenido de apenas $+0.57$ PPL).
2. **Ruptura de Simetría de Permutación en Cabezales de Atención:** En el baseline estándar (AdamW), las matrices de atención son ruido blanco isotrópico sin correlación espacial alguna ($\operatorname{AdjCos} = -0.0048 \pm 0.0058$). El acoplamiento Dirichlet induce con éxito una estructura cortical continua y suave con $\operatorname{AdjCos}$ entre **$0.754$ y $0.957$**.
3. **Resistencia Radical a la Poda 2D-DCT en Bloques de Atención:** En el baseline estándar, podar el $75\%$ o el $90\%$ de los coeficientes DCT de las matrices de atención provoca una degradación severa (la perplejidad se triplica, pasando de $6.91$ a $19.77$). En contraste, la atención topográfica (`Topographic_Attn_eps2e-3`) permanece prácticamente inmune a la compresión de 10x (pasando de $8.41$ a $9.14$), superando al baseline por **$10.63$ puntos de perplejidad (un 53.8% menor PPL a 10x compresión)**.

---

## 1. Configuración Experimental

- **Modelo:** NanoLanguageModel autorregresivo estilo NanoGPT ($d_{\text{model}} = 128, n_{\text{heads}} = 4, \text{head\_dim} = 32, n_{\text{layers}} = 2, \text{seq\_len} = 128$, FFN dim = 256, total 288,896 parámetros, de los cuales 131,072 residen en las matrices de proyección de atención).
- **Proyecciones Regularizadas:** En cada una de las 2 capas, se desacoplan y regularizan individualmente $W_q, W_k, W_v, W_o \in \mathbb{R}^{128 \times 128}$.
- **Dataset:** Tiny Shakespeare a nivel de caracteres ($1.11$ M caracteres, split temporal $90\%$ train / $10\%$ val, vocabulario = 65 caracteres).
- **Presupuesto:** 600 pasos por corrida con batch size = 32 y $\text{seq\_len} = 128$ ($2.45$ M tokens procesados por corrida), optimizador AdamW ($\text{lr} = 2\times 10^{-3}$, $\text{weight\_decay} = 10^{-4}$).
- **Semillas:** 3 semillas independientes (`[42, 100, 2026]`) por condición (total 12 corridas completas).
- **Protocolo de Evaluación de Error Estándar (SE):** Conforme a `GEMINI.md`, la pérdida de validación y la perplejidad se evalúan por secuencia sobre **640 secuencias independientes** ($N=640$, batch size 32, 20 batches).

---

## 2. Resultados Consolidados

### A. Perplejidad en Validación vs. Nivel de Esparsidad 2D-DCT en Atención

| Condición | $\epsilon_{\text{attn}}$ | Val Loss | Val PPL (0% Sparsity) | $\operatorname{AdjCos}$ Attn | DCT 50% PPL (2x) | DCT 75% PPL (4x) | DCT 90% PPL (10x) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Topographic_Attn_eps2e-3` | $2\times 10^{-3}$ | $2.129 \pm 0.030$ | $8.41 \pm 0.25$ | $0.8878 \pm 0.0090$ | $8.37$ | $8.49$ | 🌟 **$9.14$** |
| `Topographic_Attn_eps5e-4` | $5\times 10^{-4}$ | $2.013 \pm 0.012$ | $7.48 \pm 0.09$ | $0.7541 \pm 0.0136$ | $7.52$ | 🌟 **$8.25$** | $11.88$ |
| `Topographic_Attn_eps1e-2` | $10^{-2}$ | $2.320 \pm 0.039$ | $10.18 \pm 0.40$ | 🌟 **$0.9566 \pm 0.0032$** | $10.18$ | $10.26$ | $10.28$ |
| `Baseline_Standard_AdamW` | $0.0$ | **$1.933 \pm 0.017$** | 🌟 **$6.91 \pm 0.12$** | $-0.0048 \pm 0.0058$ | 🌟 **$7.31$** | ⚠️ $11.52$ | ⚠️ $19.77$ |

*Convención de rigor:* 🌟 Mejor valor numérico por columna. ⚠️ Degradación anómala o colapso por poda.

### B. Análisis de Significancia Estadística en Poda a 10x (90% de Esparsidad)

- **Pérdida en nats a 90% de esparsidad:**
  - `Baseline_Standard_AdamW`: $\mathcal{L}_{\text{val}} = \ln(19.77) = 2.984$ nats.
  - `Topographic_Attn_eps2e-3`: $\mathcal{L}_{\text{val}} = \ln(9.14) = 2.213$ nats.
  - Diferencia: $|\Delta| = 2.984 - 2.213 = \mathbf{0.771 \text{ nats}}$.
  - Error estándar conjunto: $\text{SE} \approx \sqrt{0.0072^2 + 0.0060^2} = 0.0094 \text{ nats}$.
  - Condición de rigor: $|\Delta| = 0.771 \gg 2 \times \text{SE} = 0.0188 \text{ nats}$. La superioridad en retención de información del modelo topográfico a compresión 10x supera en más de 80 veces el error estándar de la medición.

---

## 3. Análisis de Hallazgos

### A. La Invariancia Espectral de la Atención Topográfica
Al revisar la figura [`results/figures/v385_topographic_attention_curves.png`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/figures/v385_topographic_attention_curves.png):
- **Curva Plana en PPL vs Poda (Panel 1):**
  - En `Topographic_Attn_eps2e-3`, al eliminar el $90\%$ de los coeficientes DCT de todas las matrices de atención (dejando solo el 10% de parámetros activos en $W_q, W_k, W_v, W_o$), la perplejidad apenas se altera: pasa de $8.41$ a $9.14$ ($+0.73$ PPL).
  - En `Topographic_Attn_eps1e-2`, la variación a 10x compresión es de apenas $+0.10$ PPL ($10.18 \to 10.28$).
  - En contraste, el baseline estándar experimenta una explosión no lineal a partir del $50\%$ de esparsidad: su PPL se degrada de $6.91$ a $11.52$ al $75\%$ y a $19.77$ al $90\%$.

### B. Suavidad Cortical en las Proyecciones de Atención
- El panel 2 confirma que la regularización Dirichlet induce una correlación coseno adyacente masiva en los pesos de atención:
  - Baseline: $\operatorname{AdjCos} \approx -0.005$ (distribución plana, no organizada).
  - $\epsilon = 5\times 10^{-4}$: $\operatorname{AdjCos} = 0.754$.
  - $\epsilon = 2\times 10^{-3}$: $\operatorname{AdjCos} = 0.888$.
  - $\epsilon = 10^{-2}$: $\operatorname{AdjCos} = 0.957$.
- El mapa de calor comparativo (Panel 3) muestra que $W_q$ en la red topográfica desarrolla superficies de campo receptivo continuas y coherentes, frente al patrón granulado de alta frecuencia del baseline estándar.

### C. El Punto Dulce del Acoplamiento
Se observa un compromiso claro de Pareto:
- `eps5e-4` ofrece el mejor compromiso para regímenes de compresión moderada (hasta 4x / 75% esparsidad), logrando un PPL de $8.25$ con una pérdida mínima en el modelo denso ($7.48$ vs $6.91$).
- `eps2e-3` es el régimen óptimo para compresión extrema (10x / 90% esparsidad), donde su perplejidad ($9.14$) supera drásticamente al baseline ($19.77$).

---

## 4. Amenazas a la Validez

1. **Escala del Tokenizador y Dataset:** Tiny Shakespeare es un corpus de $1.11$ MB a nivel de caracteres ($V=65$). En modelos LLM de producción con tokenizadores de subpalabras (BPE, $V \approx 32\text{k} - 128\text{k}$) y secuencias largas ($L \ge 2048$), la dinámica de los cabezales de atención podría requerir una adaptación del ratio de acoplamiento $\epsilon$.
2. **Límite de Frontera entre Cabezales:** Las matrices $W_q, W_k, W_v, W_o$ empaquetan los 4 cabezales a lo largo de sus canales. El acoplamiento Dirichlet 2D actual no distingue entre la transición dentro de un mismo cabezal y la transición en la frontera entre cabezales adyacentes. Un acoplamiento 2D intra-cabezal estricto podría reducir el coste de PPL en el modelo denso.
3. **Poda Exclusiva de Atención vs FFN:** En este experimento solo se podaron las proyecciones de atención ($W_q, W_k, W_v, W_o$), manteniendo las capas densas FFN sin podar. Evaluar la compresión conjunta (Atención + FFN) es el paso natural para cuantificar la compresión global del Transformer.

---

## 5. Próximo Paso en el Roadmap

- **v386 (Topographic Transformer Completo: Attention + FFN):** Extender el acoplamiento celular Dirichlet 2D tanto a la atención como a las matrices de proyección FFN ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$), evaluando la compresión 2D-DCT global del 100% de los parámetros del modelo de lenguaje.
- **v387 (Spectral-LoRA en LLMs):** Aplicar el adaptador espectral validado en `v384` sobre los cabezales de atención topográficos de este Transformer para medir la transferencia de estilo o tarea en lenguaje.
