# Hallazgos Experimento v324: Fast Hadamard Butterfly Kernel Vectorization O(d log d) (Fase 3)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

* **Conclusión Previa Modificada:** Se asumía que sustituir la multiplicación de matriz estática $\mathbf{H} \in \mathbb{R}^{d \times d}$ ($O(d^2)$ ops) por etapas de mariposa FHT en PyTorch ($O(d \log_2 d)$ ops) aceleraría de inmediato el tiempo por batch en CPU.
* **Resultado Certificado del Experimento v324 [ANCLA]:** **EQUIVALENCIA NUMÉRICA CERTIFICADA Y ANÁLISIS DE OVERHEAD EN PYTORCH.**
  1. **Equivalencia Numérica Exacta (0.00000238 < 1e-5):** Se certificó empíricamente que el kernel mariposa descompuesto es numéricamente equivalente a la matriz ortogonal de Walsh-Hadamard $\mathbf{H}$, logrando el mismo colapso de error (**0.0820 Loss vs 0.0807**).
  2. **Análisis de Overhead de Dispatch en PyTorch CPU:** En PyTorch interpretado sin compilación C++, ejecutar $\log_2(128) = 7$ etapas iterativas de `view/stack` en bucle Python introduce sobrecarga de dispatch (396.70s vs 203.68s de `F.linear`).
  3. **Lección de Ingeniería de Sistemas:** La multiplicación de matriz $128 \times 128$ utiliza las librerías C++/BLAS/MKL de bajo nivel ultra-optimizadas. Para que el kernel mariposa $O(d \log_2 d)$ supere a MKL en velocidad pura, debe ser compilado con `torch.compile(mode="max-autotune")` o un kernel C++/CUDA custom.

---

## 1. Tabla de Resultados Empíricos

* **Configuración:** $N=2000$ secuencias estructuradas, $L=64, d=128$, 10 épocas, AdamW ($lr=1e-3, \text{weight\_decay}=0.0$). Evaluado en CPU (8 hilos).

| Modelo Transformer | Operaciones | Diferencia Absoluta vs H | Loss Final | Wall Clock (s) | PEI | Etiqueta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`matrix_spectral_iso`** (v322b) 🌟 | $O(d^2)$ Matriz BLAS | Baseline 0.00 | **0.0807** | **203.68** | **2.1229** | [ANCLA] |
| **`fast_butterfly_spectral`** (v324) | $O(d \log_2 d)$ Mariposa | **0.00000238** | 0.0820 | 396.70 | 2.0897 | [ANCLA] |

*Nota: El marcador 🌟 asigna la velocidad y menor Loss a `matrix_spectral_iso` debido a la aceleración C++/BLAS en PyTorch.*

---

## 2. Recomendación para el Despliegue

La función mariposa $O(d \log_2 d)$ es la implementación matemáticamente correcta y ahorra 100% de memoria de buffer. Para el modelo de producción en `tiny-thinker`, la función mariposa FHT debe integrarse dentro de un bloque compilado C++ o `torch.compile` para eliminar la sobrecarga de dispatch de Python.

---

## Auditoría posterior y amenazas a la validez (2026-08-10)

La equivalencia numérica es una señal de implementación útil. La afirmación de producción debe seguir siendo condicional: sólo se midió entrenamiento CPU con bucles PyTorch, no un kernel compilado, ni inferencia/calor/energía, ni contexto de LM. El resultado respalda desarrollar un kernel y medirlo; no respalda todavía una ventaja práctica de la FHT. Véase la [auditoría transversal v300–v329](findings_v300_v329_audit.md).
