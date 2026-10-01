# Findings v388: Cuantización Espectral Adaptativa estilo JPEG en Transformers Autorregresivos

**Fecha:** 2026-10-01  
**Lab:** `attention-neuron/`  
**Script:** [`scratch/prototype_v388_jpeg_spectral_quantization.py`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/prototype_v388_jpeg_spectral_quantization.py)  
**Registro Crudo:** [`results/raw/v388_jpeg_spectral_quantization.json`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/raw/v388_jpeg_spectral_quantization.json)  
**Figura:** [`results/figures/v388_jpeg_spectral_quantization.png`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/figures/v388_jpeg_spectral_quantization.png)  
**Nivel de Rigor:** Nivel 1 (Sondeo Exploratorio multi-semilla, 3 semillas por condición)  
**Etiqueta:** [SEÑAL] (Validación de cuantización espectral sub-2-bit estilo JPEG con preservación de perplejidad en LLMs topográficos)  

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

En `v383`, `v385` y `v386`, la compresión espectral se evaluó exclusivamente mediante poda binaria de magnitud (truncamiento a cero de coeficientes 2D-DCT manteniendo los coeficientes retenidos en FP32). Dicho régimen demostró que las redes topográficas soportan una escasez del 90% a 10x de compresión paramétrica. Sin embargo, no se había investigado si era factible cuantizar los coeficientes espectrales no nulos a una representación discreta de multi-bit variable (estilo JPEG) para reducir el presupuesto de bits por parámetro por debajo de 2 bits sin colapso funcional.

**Reconciliación y Nuevos Hallazgos de `v388`:**
1. **Factibilidad de la Cuantización Espectral JPEG Sub-2-Bit:** La asignación adaptativa de resolución en bandas de frecuencia radiales 2D-DCT ($\rho \le 0.15 \to 8\text{ bits}$, $0.15 < \rho \le 0.35 \to 4\text{ bits}$, $0.35 < \rho \le 0.60 \to 2\text{ bits}$, $\rho > 0.60 \to 0\text{ bits}$) alcanza una tasa efectiva de **1.68 bits/parámetro (bpp)** (compresión de **$19.0\times$** frente a FP32) sobre el 100% de las 12 matrices lineales del Transformer (262,144 parámetros). En el Transformer Topográfico, la perplejidad se preserva con una variación mínima: de $9.29 \pm 0.08$ a **$10.24 \pm 0.28$** ($\Delta = +0.95$ PPL).
2. **Colapso Catastrófico del Baseline Estándar bajo Cuantización Espectral:** En el Transformer estándar AdamW sin regularización topográfica, la misma partición JPEG produce un colapso severo: la perplejidad se degrada de $7.00 \pm 0.05$ a **$58.29 \pm 10.19$** (un incremento de $+51.29$ PPL, degradación de **$8.3\times$**), con un error relativo de reconstrucción de pesos del **$87.7\%$** (frente al **$23.0\%$** del modelo topográfico).
3. **Fracaso de la Cuantización Espacial Uniforme INT2:** La cuantización simétrica uniforme a 2 bits aplicada directamente en el espacio de coordenadas neuronales (`Spatial Uniform INT2`, 2.00 bpp) colapsa catastróficamente en ambas arquitecturas: **$506.47 \pm 156.48$** en el modelo topográfico y **$121.29 \pm 23.14$** en el estándar. A pesar de que `Spatial INT2` utiliza *más* bits (2.00 bpp) que `JPEG Adaptive` (1.68 bpp), su distorsión es destructiva porque distribuye el error de forma homogénea en el espacio, destruyendo la coherencia continua. En contraste, JPEG concentra la precisión en los modos armónicos fundamentales donde reside la energía.
4. **Frontera Sub-1-Bit (`JPEG Aggressive` a 0.91 bpp):** Con una partición más agresiva ($35.0\times$ de compresión, sub-1-bit promedio), el modelo topográfico retiene coherencia lingüística con un Val PPL de **$12.72 \pm 0.51$**, reduciendo la huella de los 262,144 parámetros lineales de **1,024 KB a solo 29.2 KB**, mientras el baseline estándar explota a **$98.41 \pm 15.93$**.
5. **Cero Sobrecarga de Índices o Metadatos:** A diferencia de la poda dispersa que requiere almacenar índices o máscaras booleanas, la cuantización JPEG asigna las bandas de frecuencia de forma puramente determinista mediante la geometría $(M, N)$ de cada matriz, requiriendo únicamente almacenar 3 factores de escala de punto flotante por matriz (menos de 0.003 bpp de sobrecarga).

---

## 1. Configuración Experimental

- **Arquitectura:** FullTopographicLM de 2 capas ($d_{\text{model}} = 128, n_{\text{heads}} = 4, \text{head\_dim} = 32, \text{ffn\_dim} = 256, \text{seq\_len} = 128$, total 288,128 parámetros).
- **Parámetros Cuantizados:** El 100% de las 12 matrices de proyección lineal ($W_q, W_k, W_v, W_o, W_{\text{in}}, W_{\text{out}}$ en ambas capas, total **262,144 parámetros**, representando el $91.0\%$ de los parámetros de la red). Embeddings y LayerNorms se mantienen como buffers de precisión FP32.
- **Modelos Base Entrenados:**
  1. `Topographic_Full_eps2e-3`: Regularización Dirichlet macroscópica ($\epsilon_{\text{attn}} = 2\times 10^{-3}, \epsilon_{\text{ffn}} = 2\times 10^{-3}$).
  2. `Standard_AdamW_Baseline`: Transformer autorregresivo estándar ($\epsilon = 0.0$).
- **Presupuesto de Entrenamiento:** 600 pasos por semilla con AdamW ($\text{lr} = 2\times 10^{-3}$, batch size 32) sobre Tiny Shakespeare.
- **Esquemas de Cuantización Evaluados:**
  1. `FP32 (Unquantized)`: Referencia base a 32.00 bpp ($1.0\times$).
  2. `JPEG Adaptive (8b/4b/2b/0b)`: Bandas radiales $\rho \le 0.15 \to 8\text{b}$, $0.15 < \rho \le 0.35 \to 4\text{b}$, $0.35 < \rho \le 0.60 \to 2\text{b}$, $\rho > 0.60 \to 0\text{b}$. Promedio ponderado: **1.68 bpp** ($19.0\times$).
  3. `JPEG Aggressive (8b/4b/2b/0b)`: Bandas radiales $\rho \le 0.10 \to 8\text{b}$, $0.10 < \rho \le 0.25 \to 4\text{b}$, $0.25 < \rho \le 0.45 \to 2\text{b}$, $\rho > 0.45 \to 0\text{b}$. Promedio ponderado: **0.91 bpp** ($35.0\times$).
  4. `Spatial Uniform INT4`: Cuantización simétrica uniforme en espacio de pesos $W$ a 4 bits: **4.00 bpp** ($8.0\times$).
  5. `Spatial Uniform INT2`: Cuantización simétrica ternaria en espacio de pesos $W$ a 2 bits: **2.00 bpp** ($16.0\times$).
  6. `Spectral Uniform INT4`: Cuantización simétrica uniforme de todos los coeficientes 2D-DCT a 4 bits: **4.00 bpp** ($8.0\times$).
- **Evaluación y Rigor:** 3 semillas independientes (`[42, 100, 2026]`). Evaluación en cada condición sobre **640 secuencias independientes retenidas** ($N=640$, 81,920 tokens), con cálculo explícito de SE por secuencia según la norma de `GEMINI.md`.

---

## 2. Resultados Consolidados

### A. Comparativa de Tasa de Bits, Perplejidad y Error de Reconstrucción

| Modo de Cuantización | Bits/Param (bpp) | Ratio Compresión | Topographic Val Loss | Topographic Val PPL | Standard Val Loss | Standard Val PPL | Error Rel. Topo | Error Rel. Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `FP32 (Unquantized)` | 32.00 | 1.0x | $2.228 \pm 0.009$ | $9.29 \pm 0.08$ | 🌟 **$1.945 \pm 0.007$** | 🌟 **$7.00 \pm 0.05$** | 0.0% | 0.0% |
| `Spatial Uniform INT4` | 4.00 | 8.0x | $2.247 \pm 0.021$ | $9.46 \pm 0.19$ | $1.981 \pm 0.006$ | $7.25 \pm 0.05$ | 28.3% | 21.5% |
| `Spectral Uniform INT4` | 4.00 | 8.0x | $2.418 \pm 0.128$ | $11.32 \pm 1.50$ | $2.029 \pm 0.012$ | $7.61 \pm 0.09$ | 35.2% | 23.8% |
| `JPEG Adaptive` | **1.68** | **19.0x** | **$2.326 \pm 0.027$** | 🌟 **$10.24 \pm 0.28$** | $4.051 \pm 0.167$ | ⚠️ **$58.29 \pm 10.19$** | **23.0%** | **87.7%** |
| `JPEG Aggressive` | **0.91** | **35.0x** | **$2.542 \pm 0.041$** | **$12.72 \pm 0.51$** | $4.576 \pm 0.163$ | ⚠️ **$98.41 \pm 15.93$** | **32.2%** | **93.5%** |
| `Spatial Uniform INT2` | 2.00 | 16.0x | $6.169 \pm 0.357$ | ⚠️ **$506.47 \pm 156.48$** | $4.779 \pm 0.201$ | ⚠️ **$121.29 \pm 23.14$** | 93.5% | 96.0% |

*Convención de rigor:* 🌟 Mejor valor de perplejidad alcanzado por régimen de precisión. ⚠️ Colapso severo de perplejidad. Todos los valores reportan media $\pm$ desviación estándar entre semillas. El SE de evaluación por secuencia fue $\le 0.007$ nats en todas las corridas ($N=640$).

### B. Huella de Memoria Física de los Pesos Lineales (262,144 Parámetros)

| Formato / Modo | Bits por Peso | Huella en Memoria (KB) | Ahorro Relativo vs FP32 | Factor de Compresión |
| :--- | :---: | :---: | :---: | :---: |
| `FP32 (Unquantized)` | 32.00 | 1,024.0 KB (1.00 MB) | 0.0% | 1.0x |
| `Spatial INT4` | 4.00 | 128.0 KB | 87.5% | 8.0x |
| `Spatial INT2` | 2.00 | 64.0 KB | 93.8% | 16.0x |
| `JPEG Adaptive` | **1.68** | **53.8 KB** | **94.7%** | **19.0x** |
| `JPEG Aggressive` | **0.91** | **29.2 KB** | **97.1%** | **35.0x** |

---

## 3. Análisis de Hallazgos

### A. La Ruptura del Límite de 2 Bits Mediante Cuantización Espectral
En la literatura convencional de compresión de redes neuronales (PTQ), la cuantización uniforme a 2 bits (`Spatial Uniform INT2`) se considera típicamente inutilizable sin técnicas agresivas de fine-tuning consciente de cuantización (QAT) debido a que reduce el rango dinámico a 3 valores discretos $\{-s, 0, s\}$, destruyendo las direcciones sutiles de proyección.
Los datos de la Tabla 2A confirman este fenómeno: tanto en el Transformer topográfico como en el estándar, `Spatial Uniform INT2` colapsa con pérdidas superiores a 4.7 nats y perplejidades $>120 - 500$.

Sin embargo, al trasladar la cuantización al dominio 2D-DCT:
- **`JPEG Adaptive` utiliza solo 1.68 bpp** (un presupuesto total de bits inferior a 2 bits por peso).
- En lugar de degradar todos los pesos por igual, asigna **8 bits a la componente DC y a los primeros armónicos espaciales** ($\rho \le 0.15$), donde se concentra la información macroscópica de conectividad neuronal inducida por Dirichlet ($\operatorname{AdjCos} \approx 0.89$).
- Las frecuencias intermedias reciben 4 bits y 2 bits, y las frecuencias más altas ($\rho > 0.60$) se truncan a 0 bits.
- Como resultado, la reconstrucción de los pesos topográficos exhibe solo un **23.0% de error relativo**, manteniendo la perplejidad en **$10.24$**, a solo $+0.95$ PPL de su cota sin cuantizar.

### B. Por Qué el Transformer Estándar Colapsa a 58.29 PPL
En un Transformer entrenado sin acoplamiento Dirichlet (AdamW estándar), las matrices de pesos tienen $\operatorname{AdjCos} \approx 0.00$. Su espectro 2D-DCT es esencialmente ruido blanco bidimensional plano: la energía está uniformemente repartida entre los modos DC y los modos de alta frecuencia.
Al aplicar la matriz JPEG:
- Truncar $\rho > 0.60$ a cero y cuantizar a 2 bits la banda media descarta o distorsiona componentes de alta frecuencia que contienen magnitudes tan grandes como las bajas frecuencias.
- El error de reconstrucción relativo en el modelo estándar es del **87.7%**.
- Esto destruye completamente la afinidad entre cabezales de atención y la activación léxica de las FFN, provocando un salto catastrófico de perplejidad a **$58.29$**.

### C. La Frontera Sub-1-Bit (`JPEG Aggressive`)
En `JPEG Aggressive`, el 67.7% de los coeficientes de alta frecuencia se truncan a 0 bits, reduciendo el presupuesto promedio a **0.91 bits/parámetro**.
- Para el Transformer Topográfico, el Val PPL resultante es **$12.72 \pm 0.51$**. El modelo sigue generando texto sintácticamente coherente.
- El almacenamiento total de las 12 matrices de proyección pasa de **1 MB a únicamente 29.2 KB** (una reducción de **35x**).
- Esto sugiere que la topografía cortical de pesos proporciona un mecanismo de compresión análogo al de los códecs de medios analógicos, donde la información semántica reside en longitudes de onda largas y el detalle fino es desechable.

---

## 4. Amenazas a la Validez

1. **Evaluación de Inferencia Descomprimida vs Kernels Cuantizados Nativos:** En este benchmark, la cuantización se evaluó simulando la des-cuantización espectral ($W_{\text{recon}} = D_{\text{out}}^T \cdot \tilde{W}_{\text{dct}} \cdot D_{\text{in}}$) en FP32 antes del forward pass. La implementación en hardware requerirá kernels especializados (ej. empaquetamiento bit-plane y multiplicación con transformadas rápidas DCT) para traducir la reducción de memoria física en aceleración de inferencia directa.
2. **Brecha de Perplejidad en FP32 Base:** El Transformer topográfico parte de una perplejidad FP32 de $9.29$ frente a $7.00$ del estándar (una brecha de $+2.29$ PPL inducida por la regularización Dirichlet fuerte $\epsilon=2\times 10^{-3}$). No obstante, en el régimen comprimido a 1.68 bpp, la situación se invierte radicalmente: el modelo topográfico alcanza **$10.24$ PPL**, mientras el estándar queda totalmente destruido en **$58.29$ PPL**.
3. **Escala del Vocabulario y Longitud de Contexto:** El experimento se realizó con un vocabulario de caracteres de Tiny Shakespeare ($V=65, T=128$). La extensión a vocabularios de subpalabras tipo BPE ($V=32,000$ o $50,000$) y secuencias largas ($T \ge 2048$) debe validarse para confirmar si la conservación espectral a sub-2-bit se mantiene inalterada.

---

## 5. Próximo Paso en el Roadmap

- **v389 (Escalado de Profundidad en Transformers Topográficos):** Analizar si la adición de profundidad (de 2 capas a 6 y 12 capas) y el uso de datasets sintéticos estructurados (TinyStories o MQAR de múltiples saltos) permite estrechar la brecha FP32 preservando la compresibilidad espectral a 1.68 bpp.
- **v390 (Kernel Espectral Integrado para Inferencia Sub-2-Bit):** Prototipar el algoritmo de descompresión rápida en bloque para evaluar latencia real de decodificación frente a GGML/GGUF estándar.
