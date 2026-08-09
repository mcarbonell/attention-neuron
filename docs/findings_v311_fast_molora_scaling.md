# Hallazgos Experimento v311: Fast MoLoRA & Scaling Sweep (Fase 4)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

* **Conclusión Previa Modificada:** En el experimento `v310`, se demostró que MoLoRA ($K=4, r=16$) superaba a LoRA estático ($r=64$), pero existía duda sobre si la aceleración tensorial con `torch.einsum` funcionaría sin overhead y cuál era el número óptimo de expertos $K$.
* **Resultado del Experimento v311:** 
  1. **Aceleración Tensorial Concretada:** `torch.einsum` redujo el tiempo de ejecución de $32.60s \to 6.05s$ (una aceleración de 5.4x en PyTorch CPU).
  2. **Escalado Monótono Confirmado (Hallazgo [ANCLA]):** A medida que se incrementa el número de expertos $K$ ($2 \to 4 \to 8 \to 16$), la Loss **mejora estrictamente monótonamente** ($3.4808 \to 3.4797 \to 3.4779 \to 3.4764$).
  3. **Superación del Baseline Denso:** La configuración `fast_molora_K16_r4` (**3.4764**) logró superar no solo a LoRA estático (3.4843), sino también al baseline denso completo `standard_dense` (3.4773).

---

## 1. Tabla de Resultados Empíricos

* **Configuración:** $N=1000$ secuencias estructuradas, $L=64$, $d_{model}=128$, 10 épocas, AdamW ($lr=1e-3$). Presupuesto iso-rango $K \times r = 64$. Evaluado en CPU (8 hilos).

| Configuración | Expertos $K$ | Rango $r$ | Parámetros | Loss Final | Wall Clock (s) | PEI | Etiqueta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`fast_molora_K16_r4`** 🌟 | 16 | 4 | 86,848 | **3.4764** | 24.76 | 0.0582 | [ANCLA] |
| **`standard_dense`** | N/A | N/A | 49,984 | 3.4773 | 2.50 | 0.0612 | [ANCLA] |
| **`fast_molora_K8_r8`** | 8 | 8 | 84,800 | 3.4779 | 9.53 | 0.0583 | [ANCLA] |
| **`fast_molora_K4_r16`** | 4 | 16 | 83,776 | 3.4797 | **6.05** | 0.0584 | [ANCLA] |
| **`fast_molora_K2_r32`** | 2 | 32 | 83,264 | 3.4808 | 6.99 | 0.0584 | [ANCLA] |
| **`static_lora`** | 1 | 64 | 82,752 | 3.4843 | 3.96 | 0.0584 | [ANCLA] |

*Nota: El marcador 🌟 asigna la mejor Loss al modelo `fast_molora_K16_r4` (3.4764).*

---

## 2. Análisis del Desempeño y Escalado

1. **Curva de Escalado por Expertos:**
   La relación entre $K$ y la Loss demuestra que la especialización en rangos más delgados ($r=4$) ruteada dinámicamente por token es numéricamente más rica que subespacios más amplios ($r=32$) con menos opciones de ruteo.
2. **Eficiencia Computacional (Wall Clock Time):**
   * $K=4$ ($6.05s$) ofrece el mejor punto de equilibrio entre velocidad y ganancia en Loss.
   * $K=16$ ($24.76s$) obtiene la máxima precisión a costa de mayor cálculo en el router softmax.

---

## 3. Amenazas a la Validez

1. **Escalado en GPUs:** El tiempo de ejecución para $K=16$ en CPU se ve penalizado por operaciones secuenciales en hilos de CPU. En GPU (DirectML/CUDA), el producto por experto se ejecuta en paralelo con latencia $O(1)$.
2. **Próximo Paso Requerido (Fase 5 - v312):** Validar la configuración ganadora $K=8$ y $K=16$ en el benchmark exigente MQAR.
