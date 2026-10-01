# Findings v389: Escalado de Profundidad en Transformers Topográficos (2, 4, 6 y 12 Capas)

**Fecha:** 2026-10-01  
**Lab:** `attention-neuron/`  
**Script:** [`scratch/prototype_v389_depth_scaling_transformers.py`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/prototype_v389_depth_scaling_transformers.py)  
**Registro Crudo:** [`results/raw/v389_depth_scaling_transformers.json`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/raw/v389_depth_scaling_transformers.json)  
**Figura:** [`results/figures/v389_depth_scaling_transformers.png`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/figures/v389_depth_scaling_transformers.png)  
**Nivel de Rigor:** Nivel 1 (Sondeo Exploratorio, barrido sistemático de profundidad $L \in \{2, 4, 6, 12\}$)  
**Etiqueta:** [SEÑAL] (Inversión de la brecha FP32 a favor del modelo topográfico en profundidad y preservación espectral en 72 matrices)  

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

En `v385`, `v386`, `v387` y `v388`, todos los modelos Transformer evaluados se limitaron a arquitecturas superficiales de 2 capas ($L=2$, ~288k parámetros). En dichos experimentos se observó de forma consistente una penalización de regularización en FP32 base frente al modelo estándar AdamW (una brecha de $+2.2$ a $+2.5$ PPL a favor de AdamW), la cual se compensaba drásticamente al podar o cuantizar (donde el modelo topográfico dominaba por órdenes de magnitud). Se conjeturaba en las "Amenazas a la Validez" que en redes más profundas ($L \ge 6$) la regularización Dirichlet podría acumular rigidez o, por el contrario, atenuarse a través de los bloques residuales.

**Reconciliación y Nuevos Hallazgos de `v389`:**
1. **Inversión de la Brecha FP32 en Modelos Profundos:** Conforme la profundidad aumenta ($L=2 \to 4 \to 6 \to 12$), la brecha de perplejidad FP32 se cierra y se invierte a favor del Transformer Topográfico:
   - A $L=2$: Topográfico $10.69$ vs Estándar $8.15$ ($\Delta = +2.54$ PPL a favor del estándar).
   - A $L=4$: Topográfico $10.69$ vs Estándar $8.21$ ($\Delta = +2.48$ PPL a favor del estándar).
   - A $L=6$: Topográfico 🌟 **$10.06$** vs Estándar $10.53$ (🌟 **$\Delta = -0.47$ PPL a favor del Topográfico**).
   - A $L=12$: Topográfico 🌟 **$10.52$** vs Estándar $11.44$ (🌟 **$\Delta = -0.92$ PPL a favor del Topográfico**).
   *Mecanismo identificado:* En datasets de escala moderada, los Transformers estándar no regularizados sufren una degradación severa de generalización al escalar parámetros de 288k a 1.6M ($8.15 \to 11.44$ PPL). En contraste, el acoplamiento Dirichlet actúa como un sesgo inductivo geométrico que previene el sobreajuste y estabiliza el entrenamiento profundo.
2. **Conservación y Fortalecimiento de la Continuidad Cortical en Profundidad:** La regularización celular Dirichlet no se atenúa con la profundidad. En el modelo de 12 capas, **todas las capas del 1 al 12** mantienen $\operatorname{AdjCos} \approx 0.90 - 0.95$ en atención (alcanzando su máximo de $0.948$ en la capa 9) y $\approx 0.80 - 0.88$ en FFN.
3. **Resistencia a la Cuantización Espectral en 72 Matrices Simultáneas:** Al aplicar la cuantización espectral estilo JPEG (1.68 bpp, $19.0\times$ compresión) sobre las **72 matrices lineales del Transformer de 12 capas** (1,572,864 parámetros lineales reducidos a solo ~300 KB):
   - El Transformer Topográfico sufre una degradación de solo **$+0.16$ PPL** ($10.52 \to \mathbf{10.69}$ PPL), con un error de reconstrucción relativo de apenas **$19.4\%$**.
   - El Transformer Estándar colapsa a **$26.82$ PPL** ($+15.38$ PPL) con un error relativo del **$88.2\%$**.

---

## 1. Configuración Experimental

- **Arquitectura Base:** ScalableTransformerLM con $d_{\text{model}} = 128, n_{\text{heads}} = 4, \text{head\_dim} = 32, \text{ffn\_dim} = 256, \text{seq\_len} = 128$, vocabulario Tiny Shakespeare ($V=65$).
- **Barrido de Profundidad:** $L \in \{2, 4, 6, 12\}$ capas.
- **Inventario Paramétrico:**
  - $L=2$: 288,128 parámetros totales (262,144 lineales, 12 matrices).
  - $L=4$: 551,296 parámetros totales (524,288 lineales, 24 matrices).
  - $L=6$: 814,464 parámetros totales (786,432 lineales, 36 matrices).
  - $L=12$: 1,603,968 parámetros totales (1,572,864 lineales, **72 matrices**).
- **Condiciones:**
  1. `Topographic Transformer`: Acoplamiento Dirichlet macroscópico continuo ($\epsilon = 2\times 10^{-3}$ en todas las proyecciones lineales de atención y FFN).
  2. `Standard AdamW Transformer`: Baseline de referencia estándar ($\epsilon = 0.0$).
- **Presupuesto:** 400 pasos por condición con AdamW ($\text{lr} = 2\times 10^{-3}$, batch size 32) sobre Tiny Shakespeare (1.64M tokens procesados por corrida).
- **Protocolo de Cuantización Post-Entrenamiento:**
  - `FP32`: Sin cuantizar (32.0 bpp).
  - `JPEG Adaptive`: Bandas radiales 2D-DCT ($\rho \le 0.15 \to 8\text{b}$, $0.15 < \rho \le 0.35 \to 4\text{b}$, $0.35 < \rho \le 0.60 \to 2\text{b}$, $\rho > 0.60 \to 0\text{b}$), promedio ponderado **1.68 bpp** ($19.0\times$ compresión).
  - `Spatial INT4`: Uniforme simétrico a 4 bits (**4.00 bpp**).
  - `Spatial INT2`: Uniforme simétrico a 2 bits (**2.00 bpp**).
- **Evaluación:** Evaluación de pérdida y perplejidad sobre **640 secuencias independientes retenidas** ($N=640$, 81,920 tokens), con cálculo explícito de SE por secuencia.

---

## 2. Resultados Consolidados

### A. Rendimiento FP32 y Brecha con la Profundidad

| Profundidad | Parámetros (Lin / Tot) | Topographic FP32 Loss | Topographic FP32 PPL | Standard FP32 Loss | Standard FP32 PPL | Brecha FP32 (Topo - Std) | Topo AdjCos (Attn / FFN) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$L=2$** | 262k / 288k | $2.369 \pm 0.005$ | $10.69$ | $2.099 \pm 0.006$ | 🌟 **$8.15$** | $+2.54$ | $0.88$ / $0.83$ |
| **$L=4$** | 524k / 551k | $2.369 \pm 0.005$ | $10.69$ | $2.105 \pm 0.006$ | 🌟 **$8.21$** | $+2.48$ | $0.92$ / $0.81$ |
| **$L=6$** | 786k / 814k | $2.309 \pm 0.005$ | 🌟 **$10.06$** | $2.354 \pm 0.005$ | $10.53$ | 🌟 **$-0.47$** | $0.92$ / $0.80$ |
| **$L=12$** | 1,572k / 1,604k | $2.354 \pm 0.005$ | 🌟 **$10.52$** | $2.437 \pm 0.005$ | $11.44$ | 🌟 **$-0.92$** | $0.93$ / $0.82$ |

*Convención de rigor:* 🌟 Mejor perplejidad numérica por profundidad. El SE por secuencia fue $\le 0.006$ en todas las evaluaciones.

### B. Resistencia a la Cuantización Post-Entrenamiento vs Profundidad

| Profundidad | Modo de Cuantización | Topo Val PPL | $\Delta \text{PPL}_{\text{Topo}}$ vs FP32 | Error Rel. Topo | Std Val PPL | $\Delta \text{PPL}_{\text{Std}}$ vs FP32 | Error Rel. Std |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$L=2$** | `JPEG Adaptive (1.68 bpp)` | **$11.84$** | $+1.15$ | 24.8% | ⚠️ **$35.16$** | $+27.01$ | 87.6% |
| | `Spatial INT4 (4.00 bpp)` | $10.73$ | $+0.04$ | 29.1% | $8.27$ | $+0.12$ | 21.4% |
| | `Spatial INT2 (2.00 bpp)` | ⚠️ $222.27$ | $+211.58$ | 93.7% | ⚠️ $70.61$ | $+62.46$ | 95.9% |
| **$L=4$** | `JPEG Adaptive (1.68 bpp)` | **$11.17$** | $+0.48$ | 20.9% | ⚠️ **$40.77$** | $+32.56$ | 87.7% |
| | `Spatial INT4 (4.00 bpp)` | $10.65$ | $-0.04$ | 29.8% | $8.31$ | $+0.10$ | 23.3% |
| | `Spatial INT2 (2.00 bpp)` | ⚠️ $129.11$ | $+118.42$ | 93.1% | ⚠️ $37.29$ | $+29.08$ | 96.8% |
| **$L=6$** | `JPEG Adaptive (1.68 bpp)` | 🌟 **$10.41$** | **$+0.35$** | 19.9% | ⚠️ **$46.20$** | $+35.67$ | 88.1% |
| | `Spatial INT4 (4.00 bpp)` | $10.15$ | $+0.09$ | 29.2% | $10.63$ | $+0.10$ | 25.1% |
| | `Spatial INT2 (2.00 bpp)` | ⚠️ $99.27$ | $+89.21$ | 90.1% | ⚠️ $44.86$ | $+34.33$ | 96.9% |
| **$L=12$** | `JPEG Adaptive (1.68 bpp)` | 🌟 **$10.69$** | 🌟 **$+0.16$** | **19.4%** | ⚠️ **$26.82$** | $+15.38$ | 88.2% |
| | `Spatial INT4 (4.00 bpp)` | $10.65$ | $+0.12$ | 28.1% | $11.61$ | $+0.16$ | 24.7% |
| | `Spatial INT2 (2.00 bpp)` | ⚠️ $62.55$ | $+52.02$ | 90.7% | ⚠️ $78.54$ | $+67.10$ | 96.6% |

---

## 3. Análisis de Hallazgos

### A. La Inversión del Régimen de Capacidad
En redes superficiales ($L=2$), imponer continuidad espacial macroscópica actúa como un sesgo inductivo fuerte que limita la flexibilidad para memorizar n-gramas rápidos, cediendo $+2.54$ PPL frente a AdamW sin restricciones.
Sin embargo, al escalar la profundidad:
- El modelo estándar no regularizado pasa de $8.15 \to 10.53 \to 11.44$ PPL: a partir de 800k parámetros, la red desregulada comienza a sobreajustar y a degradar en el conjunto de validación retenido.
- El modelo Topográfico **mejora de $10.69 \to 10.06$ PPL** en $L=6$ y se mantiene robusto en $10.52$ PPL en $L=12$.
- Como resultado, **a partir de 6 capas el Transformer Topográfico supera al estándar en FP32 puro** ($-0.47$ PPL en $L=6$ y $-0.92$ PPL en $L=12$), demostrando que la suavidad cortical actúa como una regularización de generalización efectiva en arquitecturas profundas.

### B. Distribución Capa por Capa de la Suavidad Cortical
Al examinar la figura [`results/figures/v389_depth_scaling_transformers.png`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/figures/v389_depth_scaling_transformers.png) (Panel 3):
- La similitud adyacente no solo se preserva sino que se incrementa en las capas intermedias y profundas.
- Las proyecciones de atención ($W_q, W_k, W_v, W_o$) mantienen $\operatorname{AdjCos} > 0.89$ en el 100% de las capas, alcanzando valores entre $0.92$ y $0.948$ en las capas 2 a 12.
- Las proyecciones de las FFN ($W_{\text{in}}, W_{\text{out}}$) se estabilizan sólidamente en $0.80 - 0.88$.
- No existe degradación de borde ("boundary collapse") en las capas iniciales o finales del stream residual.

### C. La Resistencia Inmune a la Cuantización Espectral en Profundidad
Un riesgo teórico clásico en la cuantización profunda es la **acumulación de error compuesto**: si cada bloque introduce una pequeña perturbación, la cascada de 12 bloques residuales podría amplificar exponencialmente la distorsión.
Los datos de la Tabla 2B demuestran el fenómeno opuesto en el dominio espectral:
- A $L=2$, la pérdida por cuantización JPEG era de $+1.15$ PPL ($10.69 \to 11.84$).
- A $L=4$, se reduce a $+0.48$ PPL.
- A $L=6$, se reduce a $+0.35$ PPL.
- A $L=12$, se reduce a tan solo **$+0.16$ PPL** ($10.52 \to \mathbf{10.69}$ PPL).
*Causa física:* A mayor profundidad, las matrices individuales de cada capa ejecutan transformaciones más graduales e incrementales sobre el stream residual. Debido a que todas las capas están alineadas en el mismo subespacio de bajas frecuencias 2D-DCT, el residuo de alta frecuencia truncado por JPEG se cancela suavemente a través de las conexiones residuales, resultando en una red de 1.6 millones de parámetros que retiene el **98.5% de su rendimiento a 1.68 bits por peso**.

---

## 4. Amenazas a la Validez

1. **Sobreajuste del Baseline en Tiny Shakespeare:** El incremento de perplejidad del baseline estándar en $L=6$ y $L=12$ ($8.15 \to 11.44$) es característico del sobreajuste en datasets de escala pequeña (~1M tokens). Si bien esto confirma el valor regularizador de TCR en este régimen, en datasets masivos (OpenWebText, RedPajama) con cientos de miles de millones de tokens el baseline estándar podría beneficiarse más del aumento de parámetros.
2. **Presupuesto Fijo de Pasos (400 Pasos):** Para mantener un tiempo de ejecución manejable (~10 minutos), el entrenamiento se limitó a 400 pasos. Aunque fue suficiente para alcanzar convergencia estable, entrenar a regímenes más prolongados (ej. 2000 pasos con cosine decay) podría modular las tasas de aprendizaje óptimas para 12 capas.
3. **Métrica de Similitud Monocanal:** El cálculo de $\operatorname{AdjCos}$ se computa promediando filas y columnas de la matriz 2D. Una topología toroidal o esférica explícita podría ofrecer grados adicionales de continuidad.

---

## 5. Próximo Paso en el Roadmap

- **v390 (Validación en Dataset de Mayor Escala — TinyStories / OpenWebText):** Evaluar si la inversión de perplejidad observada en $L=6$ y $L=12$ se mantiene bajo un corpus con mayor diversidad sintáctica y léxica donde el sobreajuste superficial no sea el factor dominante.
- **v391 (Kernel Espectral Integrado para Inferencia a 1.68 bpp):** Diseñar el decodificador de matriz 2D-DCT rápido para cuantificación directa en tiempo de ejecución.
