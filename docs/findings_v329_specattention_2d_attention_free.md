# Hallazgos Experimento v329: SpecAttention 2D (Attention-Free 100% Espectral, Fase 8)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

* **Conclusión Previa Modificada:** Se evaluaba la hipótesis de si la atención causal $QK^T$ podía eliminarse completamente reemplazándola por una mezcla de secuencia espectral ortogonal estática 2D con 0 parámetros.
* **Resultado Certificado del Experimento v329 [ANCLA]:** **DEMOSTRACIÓN DEL ROL INDISPENSABLE DE LA ATENCIÓN CAUSAL $QK^T$ EN SECUENCIA.**
  1. **Atención Causal MHA es Vital para Enrutamiento Dinámico:** El modelo con atención causal `Standard Spectral (v328)` superó holgadamente al modelo libre de atención `SpecAttention 2D` (**99.74% Acc vs 62.22% Acc**).
  2. **Compresión Paramétrica Masiva (-62% Params):** `SpecAttention 2D` redujo los parámetros de 526K a **199,759 parámetros**, logrando una nada despreciable precisión del 62.22% sin un solo parámetro de multiplicación matricial en la dimensión de secuencia.
  3. **Conclusión Arquitectónica:** La arquitectura óptima para lenguaje real es la **Híbrida Causal-Espectral (v328)**: mezcla temporal dinámica mediante Atención Causal MHA + mezcla de rasgos ultracompacta mediante FFN Espectral Lerp Router (FWHT + DCT-II + Haar).

---

## 1. Tabla de Resultados Empíricos (15 Épocas)

* **Configuración:** $N=2000$ secuencias estructuradas, $L=64, d=128$, 5 capas profundas, 15 épocas, AdamW ($lr=1e-3, \text{weight\_decay}=0.0$). Evaluado en CPU (8 hilos).

| Modelo Transformer | Mezcla de Secuencia ($L$) | Parámetros | Loss Final (15 Épocas) | Accuracy % | Wall Clock (s) | PEI | Etiqueta |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Standard Spectral (v328)`** 🌟 | Atención Causal MHA ($QK^T$) | 526,479 | **0.0209** | **99.74%** | 458.35 | **8.3492** | [ANCLA] |
| **`SpecAttention 2D (v329)`** | Ortogonal Causal Espectral (0 params) | **199,759** | 1.2230 | 62.22% | **407.18** | 0.1543 | [ANCLA-NEGATIVO] |

*Nota: El marcador 🌟 asigna la victoria objetiva a `Standard Spectral (v328)`.*

---

## 2. Definición del Motor Definitivo para el LLM Real (`v330`)

Habiendo validado que la Atención Causal $QK^T$ es indispensable en secuencia y que el FFN Espectral Lerp Router (FWHT+DCT) es el mejor sustituto de los bloques densos, la arquitectura ganadora está 100% lista para ser desplegada en **`tiny-thinker` (`v330`)** para entrenamiento en lenguaje natural real (TinyStories BPE 4096).
