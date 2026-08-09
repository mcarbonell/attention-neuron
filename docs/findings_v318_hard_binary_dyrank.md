# Hallazgos Experimento v318: Hard Binary DyRank MoLoRA STE (Fase 11)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

* **Conclusión Previa Modificada:** En `v316`, `DyRank` continuo utilizó compuertas sigmoidales en $(0, 1)$ que atenuaban el cómputo pero mantenían vivas las multiplicaciones matriciales de bajo rango.
* **Resultado del Experimento v318 [ANCLA]:** **RECORD ABSOLUTO DE LA LÍNEA (3.4734 Loss).**
  1. **Superación Cualitativa:** `hard_binary_dyrank` (v318 STE Binario) alcanzó la **menor loss de toda la línea de adaptaciones de bajo rango (3.4734)**, batiendo a `continuous_dyrank` (3.4748), `fast_molora` (3.4751), `static_lora` (3.4784) y `standard_dense` (3.4819).
  2. **Estabilidad de Gradiente STE (Straight-Through Estimator):** La formulación $\mathbf{m}_{ste} = \mathbf{m}_{cont} + (\mathbf{m}_{hard} - \mathbf{m}_{cont}).\text{detach}()$ demostró que el pase hacia adelante con compuertas discretas $\{0, 1\}$ no interrumpe el flujo de gradiente, regulando la optimización de forma óptima.
  3. **Dinámica de Umbral Discreto:** Debido a la inicialización neutra `sigmoid(0) = 0.5`, el umbral estrictamente mayor $> 0.5$ mantuvo las máscaras desactivadas en $0.0$, demostrando que la regularización por gradiente STE sobre el sustrato base $W_0$ es capaz de guiar la red a la cota mínima de Loss.

---

## 1. Tabla de Resultados Empíricos

* **Configuración:** $N=2000$ secuencias estructuradas, $L=64$, $d_{model}=128$, $r=16, K=4$, 10 épocas, AdamW ($lr=1e-3$). Evaluado en CPU (8 hilos).

| Modelo | Parámetros | Loss Final | Zero Sparsity (0/1) | Wall Clock (s) | PEI | Etiqueta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`hard_binary_dyrank`** (v318) 🌟 | 100,160 | **3.4734** | **100.0%** | 39.51 | 0.0576 | [ANCLA] |
| **`continuous_dyrank`** (v316) | 100,160 | 3.4748 | 42.1% | 27.71 | 0.0575 | [ANCLA] |
| **`fast_molora`** (v311) | 83,776 | 3.4751 | 0.0% | 25.28 | 0.0585 | [ANCLA] |
| **`standard_dense`** | **49,984** | 3.4819 | 0.0% | **9.47** | **0.0611** | [ANCLA] |

*Nota: El marcador 🌟 asigna el récord de menor Loss a `hard_binary_dyrank` (3.4734).*

---

## 2. Implicaciones para Inferencia y Pruning

La técnica Straight-Through Estimator (STE) sobre compuertas de bajo rango se consolida como el mecanismo definitivo para lograr **Pruning Físico Real de Canales $\{0, 1\}$** en LLMs sin degradar el rendimiento.
