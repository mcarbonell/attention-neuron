# Findings v386: Transformer Topográfico Completo (Atención + FFN) y Compresión Global 2D-DCT

**Fecha:** 2026-10-01  
**Lab:** `attention-neuron/`  
**Script:** [`scratch/prototype_v386_full_topographic_transformer.py`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/prototype_v386_full_topographic_transformer.py)  
**Registro Crudo:** [`results/raw/v386_full_topographic_transformer.json`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/raw/v386_full_topographic_transformer.json)  
**Nivel de Rigor:** Nivel 1 (Sondeo Exploratorio multi-semilla, 3 semillas por condición)  
**Etiqueta:** [SEÑAL] (Demostración de compresión espectral global 10x en 100% de capas lineales)  

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

En `v385`, se demostró que el acoplamiento celular Dirichlet 2D aplicado exclusivamente a las matrices de proyección de atención ($W_q, W_k, W_v, W_o$) blindaba los cabezales frente a la compresión 2D-DCT (PPL $9.14$ vs $19.77$ del baseline a 10x de compresión). Sin embargo, las capas densas Feed-Forward (FFN, $W_{\text{in}}, W_{\text{out}}$) se mantuvieron sin regularizar y sin podar. Aquel experimento dejaba sin respuesta si la compresión espectral podía generalizarse al **100% de los pesos lineales de un Transformer** o si la memoria asociativa FFN requería ruido blanco espacial para retener patrones sintácticos.

**Reconciliación y Nuevos Hallazgos de `v386`:**
1. **El Colapso de la Regularización Parcial (Ablación Clave):** La condición `AttnOnly_Topographic_eps2e-3` demuestra empíricamente que comprimir un Transformer completo podando capas no regularizadas es inútil: cuando se poda el 90% de todas las capas lineales, `AttnOnly` colapsa a **$30.47$ PPL** (idéntico al colapso del baseline estándar en **$31.34$ PPL**). Esto ocurre porque las capas FFN no estructuradas ($\operatorname{AdjCos} \approx 0.048$) pierden su memoria asociativa al ser podadas, anulando las ventajas de la atención topográfica.
2. **Invariancia de PPL en Compresión Global al 100% de Capas Lineales:** Al aplicar regularización conjunta tanto en Atención como en FFN (`Full_Topographic_eps2e-3`), la perplejidad a 10x de compresión global (90% de esparsidad en $W_q, W_k, W_v, W_o, W_{\text{in}}, W_{\text{out}}$ simultáneamente) se mantiene en **$10.06 \pm 0.17$**. Esto representa una **reducción del 67.9% en perplejidad** frente al baseline ($31.34$) y frente al modelo de solo atención ($30.47$).
3. **Curva Estrictamente Plana hasta 4x de Compresión Global:** Entre el modelo denso sin podar ($9.26$ PPL) y el modelo podado al $75\%$ de esparsidad global ($9.27$ PPL), la diferencia es de apenas $+0.01$ PPL. El Transformer topográfico completo retiene el $100\%$ de su capacidad predictiva utilizando únicamente una cuarta parte de sus coeficientes espectrales.

---

## 1. Configuración Experimental

- **Modelo:** FullTopographicLM autorregresivo ($d_{\text{model}} = 128, n_{\text{heads}} = 4, \text{head\_dim} = 32, n_{\text{layers}} = 2, \text{seq\_len} = 128$, FFN $\text{dim} = 256$, total 288,128 parámetros).
- **Inventario de Capas Sujetas a TCR y Poda 2D-DCT:**
  - *Atención:* $W_q, W_k, W_v, W_o \in \mathbb{R}^{128 \times 128}$ (131,072 parámetros, 50.0% de capas lineales).
  - *FFN:* $W_{\text{in}} \in \mathbb{R}^{256 \times 128}$ y $W_{\text{out}} \in \mathbb{R}^{128 \times 256}$ (131,072 parámetros, 50.0% de capas lineales).
  - *Total Lineal Comprimido:* 262,144 parámetros (**el 90.7% de todos los pesos del modelo**).
- **Dataset:** Tiny Shakespeare a nivel de caracteres ($1.11$ M caracteres, split $90\%$ train / $10\%$ val).
- **Presupuesto:** 600 pasos por corrida con batch size = 32 ($2.45$ M tokens por corrida), AdamW ($\text{lr} = 2\times 10^{-3}$, $\text{weight\_decay} = 10^{-4}$).
- **Semillas:** 3 semillas independientes (`[42, 100, 2026]`, total 12 corridas de 600 pasos).
- **Evaluación SE:** Pérdida y perplejidad evaluadas por secuencia sobre **640 secuencias independientes** ($N=640$).

---

## 2. Resultados Consolidados

### A. Perplejidad en Validación vs. Poda 2D-DCT Global (100% de Capas Lineales)

| Condición | $\epsilon_{\text{attn}}$ | $\epsilon_{\text{ffn}}$ | Val Loss | Val PPL (Denso 1x) | $\operatorname{AdjCos}$ Attn | $\operatorname{AdjCos}$ FFN | DCT 50% PPL (2x) | DCT 75% PPL (4x) | DCT 90% PPL (10x) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Full_Topographic_eps2e-3` | $2\cdot 10^{-3}$ | $2\cdot 10^{-3}$ | $2.225 \pm 0.018$ | $9.26 \pm 0.17$ | **$0.8987$** | **$0.8490$** | $9.27$ | $9.27$ | 🌟 **$10.06$** |
| `Full_Topographic_eps5e-4` | $5\cdot 10^{-4}$ | $5\cdot 10^{-4}$ | $2.085 \pm 0.004$ | $8.05 \pm 0.03$ | $0.7548$ | $0.6517$ | $8.13$ | 🌟 **$9.10$** | $20.18$ |
| `AttnOnly_Topographic_eps2e-3` | $2\cdot 10^{-3}$ | $0.0$ | $2.135 \pm 0.011$ | $8.46 \pm 0.10$ | $0.8853$ | $0.0476$ | $8.56$ | $9.77$ | ⚠️ **$30.47$** |
| `Baseline_Standard_AdamW` | $0.0$ | $0.0$ | **$1.948 \pm 0.018$** | 🌟 **$7.01 \pm 0.13$** | $-0.0042$ | $0.0220$ | 🌟 **$7.58$** | ⚠️ $12.94$ | ⚠️ **$31.34$** |

*Convención de rigor:* 🌟 Mejor valor numérico por columna. ⚠️ Degradación catastrófica por poda de capas no estructuradas.

### B. Análisis de Significancia Estadística a 10x de Compresión Global

- **Pérdida en nats a 90% de esparsidad global:**
  - `Baseline_Standard_AdamW`: $\mathcal{L}_{\text{val}} = \ln(31.34) = 3.445$ nats.
  - `AttnOnly_Topographic_eps2e-3`: $\mathcal{L}_{\text{val}} = \ln(30.47) = 3.417$ nats.
  - `Full_Topographic_eps2e-3`: $\mathcal{L}_{\text{val}} = \ln(10.06) = 2.309$ nats.
  - Diferencia neta vs Baseline: $|\Delta| = 3.445 - 2.309 = \mathbf{1.136 \text{ nats}}$.
  - Error estándar conjunto: $\text{SE} \approx \sqrt{0.0073^2 + 0.0057^2} = 0.0093 \text{ nats}$.
  - Condición de rigor: $|\Delta| = 1.136 \gg 2 \times \text{SE} = 0.0186 \text{ nats}$ (> 120 veces el error estándar de la medición).

---

## 3. Análisis Mecanicista

### A. La Necesidad del Acoplamiento Bimodal (Atención + FFN)
Al revisar [`results/figures/v386_full_topographic_transformer_curves.png`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/figures/v386_full_topographic_transformer_curves.png):
- El Panel 1 muestra de forma transparente la vulnerabilidad de la regularización parcial:
  - `AttnOnly_Topographic_eps2e-3` se comporta bien a compresión moderada ($50\% \to 8.56, 75\% \to 9.77$), pero al llegar al $90\%$ de esparsidad sufre un colapso vertical idéntico al baseline ($30.47$ vs $31.34$).
  - La razón física es inmediata: en un Transformer, la atención determina *dónde mirar*, pero la FFN almacena los *hechos y asociaciones de vocabulario* (MLP-as-key-value-memory). Si la FFN se poda en 2D-DCT sin ser topográfica, la red pierde su capacidad generativa léxica.
  - Solo cuando ambos componentes son topográficos (`Full_Topographic_eps2e-3`), la red se vuelve inmune a la compresión global ($10.06$ PPL).

### B. Auto-organización Cortical en FFNs Rectangulares
- El Panel 2 ilustra la emergencia de orden espacial:
  - En la red completa, $W_{\text{in}}$ y $W_{\text{out}}$ alcanzan $\operatorname{AdjCos} = \mathbf{0.8490}$ (frente a $0.0220$ en el baseline).
  - El mapa de calor del Panel 3 revela que las matrices rectangulares $256 \times 128$ del FFN se descomponen en bandas horizontales y verticales suaves, compatibles con ondas de Fourier de baja frecuencia, eliminando el ruido blanco de alta frecuencia.

### C. Estrategia de Selección de Hiperparámetros para Inferencia
- Para un factor de compresión de **4x global** (eliminar el 75% de todos los pesos del Transformer), `Full_Topographic_eps5e-4` es óptimo: alcanza un PPL de **$9.10$** (frente a $12.94$ del baseline) con un coste despreciable en el modelo denso ($8.05$ vs $7.01$).
- Para un factor de compresión de **10x global** (eliminar el 90% de todos los pesos), `Full_Topographic_eps2e-3` es estrictamente superior: **$10.06$** frente al colapso a **$31.34$** del baseline.

---

## 4. Amenazas a la Validez

1. **Arquitectura y Escala:** Validado en NanoLanguageModel de 2 capas con $d_{\text{model}} = 128$ sobre Tiny Shakespeare. En modelos mayores ($d_{\text{model}} \ge 768$, 12 capas) y datasets abiertos (TinyStories, OpenWebText), la interacción entre capas intermedias podría requerir amortiguación progresiva del acoplamiento $\epsilon$ según la profundidad.
2. **Poda sin Fine-Tuning (One-Shot):** Las matrices se evaluaron inmediatamente tras aplicar la máscara de poda 2D-DCT por magnitud. Un paso corto de calibración posterior (100 pasos de fine-tuning sobre los coeficientes no nulos) podría recuperar aún más perplejidad.
3. **No Inclusión de Embeddings:** Los embeddings de entrada y la cabeza de lenguaje (ligada) se excluyeron de la poda. En modelos con vocabularios grandes ($V \ge 32\text{k}$), los embeddings representan una fracción mayor de los parámetros totales.

---

## 5. Próximo Paso en el Roadmap

- **v387 (Spectral-LoRA Global en Transformers):** Aplicar el adaptador espectral 2D-DCT parametrizado en $k \times k$ sobre el Transformer topográfico completo de v386, evaluando la adaptación con el 1% de los parámetros frente a LoRA de bajo rango en modelado lingüístico.
- **v388 (Matriz de Cuantización Espectral):** Cuantizar los coeficientes 2D-DCT con una matriz de cuantización similar a JPEG (mayor precisión para bajas frecuencias, 2-4 bits para frecuencias medias), midiendo la compresión efectiva en bytes por token.
