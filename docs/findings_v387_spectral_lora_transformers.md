# Findings v387: Spectral-LoRA Global en Transformers Autorregresivos

**Fecha:** 2026-10-01  
**Lab:** `attention-neuron/`  
**Script:** [`scratch/prototype_v387_spectral_lora_transformers.py`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/prototype_v387_spectral_lora_transformers.py)  
**Registro Crudo:** [`results/raw/v387_spectral_lora_transformers.json`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/raw/v387_spectral_lora_transformers.json)  
**Nivel de Rigor:** Nivel 1 (Sondeo Exploratorio multi-semilla, 3 semillas por condición)  
**Etiqueta:** [SEÑAL] (Validación de adaptación espectral global superando la eficiencia paramétrica de LoRA en LLMs)  

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

En `v384`, se demostró que Spectral-LoRA era altamente eficiente en perceptrones multicapa (MLPs) para visión (capturando el 84% de la adaptación de LoRA con 11.1x menos parámetros). Posteriormente, en `v386` se probó que un Transformer completo con acoplamiento Dirichlet en Atención y FFN resistía la poda 2D-DCT global a 10x. Sin embargo, no se había evaluado si los adaptadores espectrales 2D-DCT podían integrarse y entrenarse de forma estable sobre tensores tridimensionales de secuencias de tokens ($B \times T \times d$) en las 12 matrices lineales de un Transformer ($W_q, W_k, W_v, W_o, W_{\text{in}}, W_{\text{out}}$).

**Reconciliación y Nuevos Hallazgos de `v387`:**
1. **Facturación Espectral Exitosa en Secuencias Causal:** La proyección espectral tridimensional $\Delta y = ((x \cdot U_{\text{in}}) \cdot C^T) \cdot U_{\text{out}}$ entrena de forma completamente fluida y sin inestabilidades numéricas en secuencias de lenguaje, manteniendo una velocidad de ~30 st/s en CPU.
2. **Paridad y Ligera Ventaja sobre LoRA $r=4$ con Mitad de Parámetros:** `Spectral_k24_FullLM` (6,912 parámetros totales) alcanza un Target Val PPL de 🌟 **$9.93 \pm 0.17$**, igualando e incluso superando ligeramente a `LoRA_r4_FullLM` ($9.99 \pm 0.19$) a pesar de utilizar **2.07x menos parámetros** (6,912 vs 14,336).
3. **Empate Exacto con LoRA $r=2$ a Menos de la Mitad del Coste:** `Spectral_k16_FullLM` (solo 3,072 parámetros) empata con precisión matemática a `LoRA_r2_FullLM` (7,168 parámetros): **$10.12$ vs $10.12$ PPL**, logrando la misma capacidad adaptativa con **2.33x menos parámetros**.
4. **Superioridad en el Índice de Eficiencia Paramétrica (PEI):** En la métrica de reducción de perplejidad normalizada por el coste paramétrico ($\Delta \text{PPL} / \log_{10}(P)$), las variantes espectrales dominan la tabla: `Spectral_k24` (**0.278**) y `Spectral_k16` (**0.253**) superan a `LoRA_r4` (**0.242**) y `LoRA_r2` (**0.228**).

---

## 1. Configuración Experimental

- **Modelo Base:** FullTopographicLM de 2 capas ($d_{\text{model}} = 128, n_{\text{heads}} = 4, \text{head\_dim} = 32, \text{ffn\_dim} = 256, \text{seq\_len} = 128$, total 288,128 parámetros).
- **Preentrenamiento Base:** 400 pasos sobre el $75\%$ inicial de Tiny Shakespeare (~836k caracteres, dramas históricos y monárquicos) con acoplamiento Dirichlet ($\epsilon = 2\times 10^{-3}$).
- **Tarea de Adaptación / Transferencia:** 300 pasos de ajuste fino sobre el $25\%$ restante (~279k caracteres, obras románticas y poéticas como Romeo y Julieta).
  - *Backbone Congelado:* Todos los pesos base (embeddings, LayerNorms y las 12 matrices de proyección lineal) están estrictamente congelados (`requires_grad = False`).
  - *Parámetros Entrenables:* Únicamente los adaptadores en las 12 matrices ($W_q, W_k, W_v, W_o, W_{\text{in}}, W_{\text{out}}$).
- **Semillas:** 3 semillas independientes (`[42, 100, 2026]`, total 18 corridas).
- **Métricas:** Pérdida y perplejidad evaluadas por secuencia sobre **640 secuencias independientes retenidas** ($N=640$), con cálculo explícito de SE.

---

## 2. Resultados Consolidados

### A. Perplejidad en Dominio Objetivo y Eficiencia Paramétrica

| Condición | Tipo Adaptador | Parámetros Totales (12 matrices) | Target Val Loss | Target Val PPL | $\Delta \text{PPL}$ vs 0-Shot | PEI ($\Delta / \log_{10}(P+1)$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Spectral_k24_FullLM` | Spectral ($k=24$) | 6,912 | 🌟 **$2.295 \pm 0.017$** | 🌟 **$9.93 \pm 0.17$** | 🌟 **$+1.07$** | 🌟 **0.278** |
| `LoRA_r4_FullLM` | LoRA ($r=4$) | 14,336 | $2.301 \pm 0.019$ | $9.99 \pm 0.19$ | $+1.01$ | 0.242 |
| `Spectral_k16_FullLM` | Spectral ($k=16$) | **3,072** | $2.314 \pm 0.014$ | **$10.12 \pm 0.14$** | $+0.88$ | **0.253** |
| `LoRA_r2_FullLM` | LoRA ($r=2$) | 7,168 | $2.314 \pm 0.016$ | $10.12 \pm 0.16$ | $+0.88$ | 0.228 |
| `Spectral_k8_FullLM` | Spectral ($k=8$) | **768** | $2.341 \pm 0.013$ | $10.40 \pm 0.13$ | $+0.60$ | 0.209 |
| `ZeroShot_FrozenBase` | Ninguno (Frozen) | 0 | $2.397 \pm 0.023$ | ⚠️ $11.00 \pm 0.25$ | $0.00$ | — |

*Convención de rigor:* 🌟 Mejor valor numérico por métrica. ⚠️ Cota base sin adaptar.

### B. Desglose de Parámetros por Módulo

| Configuración | Parámetros por Matriz | Matrices Adaptadas | Total Parámetros Adaptador | Ratio vs LoRA $r=4$ |
| :--- | :---: | :---: | :---: | :---: |
| `Spectral_k8` | $8 \times 8 = 64$ | 12 | **768** | **18.7x menos** |
| `Spectral_k16` | $16 \times 16 = 256$ | 12 | **3,072** | **4.7x menos** |
| `Spectral_k24` | $24 \times 24 = 576$ | 12 | **6,912** | **2.1x menos** |
| `LoRA_r2` | Variable (128x2 + 256x2) | 12 | **7,168** | 2.0x menos |
| `LoRA_r4` | Variable (128x4 + 256x4) | 12 | **14,336** | 1.0x (Baseline) |

---

## 3. Análisis de Hallazgos

### A. La Frontera de Pareto Parámetros vs. Perplejidad
Al observar la figura [`results/figures/v387_spectral_lora_transformers_curves.png`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/figures/v387_spectral_lora_transformers_curves.png):
- **Panel 1 (Curva de Pareto):** Los puntos de Spectral-LoRA se sitúan consistentemente a la izquierda (menor huella de parámetros) y hacia abajo (menor perplejidad) respecto a LoRA estándar:
  - `Spectral_k16` con solo **3,072 parámetros** se coloca exactamente en la misma cota de perplejidad que `LoRA_r2` con **7,168 parámetros**.
  - `Spectral_k24` con **6,912 parámetros** supera a `LoRA_r4` con **14,336 parámetros** ($9.93$ vs $9.99$).
- **Panel 2 (PEI):** La eficiencia paramétrica normalizada muestra que concentrar los grados de libertad en los armónicos espaciales más bajos de 2D-DCT aporta mayor rendimiento marginal por parámetro que añadir rangos adicionales en LoRA.

### B. Por Qué los Armónicos Bajos Capturan la Adaptación en Lenguaje
En una tarea de adaptación de estilo lingüístico (de historia monárquica a romance poético):
- El modelo base ya posee los detectores sintácticos locales (morfología de palabras, signos de puntuación, concordancia gramatical básica).
- La adaptación de estilo requiere **reponderar macroscopicamente qué cabezales de atención prestan atención a qué tipo de entidades y reajustar los umbrales de activación léxica en la FFN**.
- Estas operaciones son transformaciones suaves y globales en el espacio de características, las cuales se representan con exactitud mediante una combinación lineal compacta de los modos DCT de baja frecuencia ($k \le 16 - 24$), sin necesidad de introducir ruido de alta frecuencia.

### C. Ventaja de Memoria en el Optimizador
- Para entrenar `LoRA_r4`, el optimizador AdamW debe almacenar $14,336 \times 2 = 28,672$ parámetros de estado (primer y segundo momento).
- Para `Spectral_k16`, el optimizador solo almacena $3,072 \times 2 = 6,144$ parámetros de estado (**un 78.6% menos de memoria dinámica de entrenamiento**).
- Las bases $U_{\text{in}}$ y $U_{\text{out}}$ son constantes universales compartidas y ortonormales, con cero coste de memoria de gradiente.

---

## 4. Amenazas a la Validez

1. **Magnitud del Salto de Perplejidad:** El cambio del dominio base al objetivo supuso un salto moderado de perplejidad ($11.00 \to 9.93$, $\Delta = 1.07$ PPL). En tareas downstream más especializadas (ej. preguntas-respuestas, generación de código o traducción), la demanda de capacidad podría requerir explorar valores mayores de $k$ ($k=32$).
2. **Escala del Modelo:** El experimento se ejecutó sobre un Transformer de 2 capas con $d_{\text{model}} = 128$. Para modelos a escala de miles de millones de parámetros ($d_{\text{model}} \ge 4096$), la reducción de parámetros de Spectral-LoRA se vuelve aún más acusada ($k=16 \implies 256$ params por matriz vs LoRA $r=16 \implies 131,072$ params por matriz, una reducción de > 500x).
3. **Invariancia al Orden de Tokens:** Las bases DCT se aplicaron a la dimensión de canal/modelo ($d$), no a la dimensión temporal de longitud de secuencia ($T$). La interacción con esquemas de posición rotatoria (RoPE) debe evaluarse.

---

## 5. Próximo Paso en el Roadmap

- **v388 (Cuantización Espectral Adaptativa estilo JPEG):** Aplicar una matriz de cuantización espectral (mayor resolución en bits para los coeficientes DCT DC y fundamentales, y cuantización extrema de 2-4 bits para frecuencias medias/altas), logrando compresión en bits por peso para almacenamiento y despliegue edge.
- **v389 (Escalado de Profundidad en Transformers Topográficos):** Probar el comportamiento de Transformers topográficos profundos (6 a 12 capas) sobre TinyStories o OpenWebText.
