# Ficha Técnica ONE-001: ChunkwiseComplexDeltaPhase

> 📚 **Familia:** 3. Neuronas Geométricas y de Fase Compleja  
> 🏷️ **Etiqueta de Rigor:** **[ANCLA]** (Nivel 2 - Verificado en 5 semillas iso-paramétricas en v306)

---

## 1. Formulación Matemática & Origen Histórico
Desarrollada en el proyecto *Attention-Neuron* (2026), `ChunkwiseComplexDeltaPhase` parametriza las claves y consultas de atención lineal autorregresiva sobre el círculo unitario complejo $S^1 \subset \mathbb{C}^{d_k}$:

$$K_t = e^{i \theta_{k,t}} = \cos(\theta_{k,t}) + i \sin(\theta_{k,t})$$
$$Q_t = e^{i \theta_{q,t}} = \cos(\theta_{q,t}) + i \sin(\theta_{q,t})$$

La actualización del estado de memoria asociativo $M_t \in \mathbb{C}^{d_k \times d_k}$ por bloques (chunkwise de tamaño $C=64$) sigue la regla Delta compleja:

$$v_{\text{old}} = \text{Re}(M_{t-1} K_t^*)$$
$$M_t = M_{t-1} + \beta_t (v_t - v_{\text{old}}) \otimes K_t^*$$

---

## 2. Hiperparámetros & Optimizador
- **Optimizador:** AdamW ($\beta_1=0.9, \beta_2=0.999$).
- **Learning Rate:** $4.00\text{e-}03$ (con scheduler de warmup lineal del 5%).
- **Weight Decay:** `weight_decay = 0.0` (las proyecciones angulares en $S^1$ no deben ser atenuadas por norma L2).
- **Normalización:** LayerNorm previo a la proyección + Causal Conv1D short kernel ($k=4$).

---

## 3. Presupuesto Paramétrico & Intensidad Aritmética
- **Parámetros por Capa ($d_{\text{model}}=64, n_{\text{heads}}=2, d_k=32$):** 33,922 parámetros.
- **Parámetros Totales (4 Capas + Emb + LM Head):** 144,331 parámetros.
- **Intensidad Aritmética:** Requiere aritmética compleja ($\text{Re}(\cdot)$ e $\text{Im}(\cdot)$), multiplicando por 2 la memoria estado activa respecto a reales ($2 d_k^2$ floats/head).

---

## 4. Desempeño y Métrica Principal
- **Tiny Shakespeare (LM Autorregresivo Iso-Paramétrico, 5 Semillas, $L=256$):** **Val Loss 1.7849 ± 0.0028 / Val PPL 5.96 ± 0.02** 🌟 (v306 [ANCLA]).
- **MQAR Capacidad (128 Pares, $L=1024$):** **95.61% Accuracy** (v300).
- **MQAR Vocabulario Compartido ($c=128$, $L=1088$):** **94.30% Accuracy** (v302).


---

## 5. Dominio de Tarea & Benchmarks
- **Tareas Ideales:** Recuperación asociativa de alta carga (MQAR), secuencias largas bajo interferencia de tokens repetidos.
- **Desempeño en Texto Real:** Competitivo con la atención Softmax $O(N^2)$ (PPL 6.00 vs 6.36).

---

## 6. Perfil de Hardware & Latencia Real (Wall-Clock Benchmark)
- **Hardware Evaluado:** CPU AMD Ryzen 7 8845HS / PyTorch v2.10+.
- **Latencia Forward Pass (ms per batch, $B=8$):**
  - $L=256$: 140.40 ms (0.0685 ms/token, 14,587 tok/s)
  - $L=512$: 227.56 ms (0.0555 ms/token, 18,000 tok/s)
  - $L=1024$: **425.52 ms** (0.0519 ms/token, 19,251 tok/s)
  - $L=2048$: **894.57 ms** (0.0546 ms/token, 18,314 tok/s)
- **Comparativa de Escalado:**
  - A $L=2048$, `ComplexDeltaPhase` es **3.62x más rápido que Softmax MHA** (894.57 ms vs 3,240.51 ms), demostrando escalado lineal estricto $O(L)$ frente al colapso cuadrático $O(L^2)$ de Softmax MHA.
  - Frente al control real `RealDeltaNet`, la atención compleja añade un overhead razonable de $\approx 1.36\times$ (894 ms vs 656 ms) debido al cómputo de la parte real e imaginaria en PyTorch genérico.


---

## 7. Generalización Out-of-Distribution (OOD)
Mantiene un **72.29% de precisión** en $L=2048$ (256 pares) cuando es entrenado en secuencias de longitud variable, superando el colapso masivo de baselines reales en el mismo régimen.

---

## 8. Interpretabilidad & Geometría del Espacio de Estados
- **Variedad de Estado:** Círculo unitario $S^1 \subset \mathbb{C}^{d_k}$.
- **Dinámica:** La norma $|K_t|=1$ evita puntos ciegos o cancelaciones en cero al avanzar la secuencia autorregresiva (hélice 3D).

---

## 9. Trazabilidad de Código & Scripts del Corpus
- **Implementación del Bloque:** [scratch/run_v304_tiny_lm.py](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/run_v304_tiny_lm.py#L90-L135)
- **Script de Benchmark:** `scratch/run_v306_tiny_lm_isoparam_multiseed.py`

---

## 10. Amenazas a la Validez, Anomalías & Bugs Conocidos (⚠️)
- ⚠️ **Desplome bajo Sobreescritura Activa (v303):** Cae de 99.61% a 8.40% cuando el 30% de las claves se reescriben en la secuencia, mostrando la limitación de la Delta Rule estándar para re-aprender borrados en 20 épocas.
- ⚠️ **Evaluación Iso-Paramétrica en Texto:** En v304 tuvo paridad con el control real, pero el control real contaba con un 21.7% más de parámetros (corregido en v306).
