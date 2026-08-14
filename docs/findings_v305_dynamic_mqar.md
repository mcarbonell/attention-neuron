# Informe de Experimento: Findings v305 - Arreglo de Harness y Evaluación Dinámica en Vivo de MQAR

**Fecha:** 2026-08-12  
**ID de Experimento:** `v305_fixed_mqar_harness`  
**Nivel de Rigor:** Nivel 1 (Sondeo de Diagnóstico y Arreglo de Harness)  
**Etiqueta de Resultado:** [SEÑAL] (Confirmación de arnés dinámico y evaluación de crosstalk)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

Este experimento **modifica e invalida explícitamente** la metodología de evaluación empleada en las versiones preliminares `v292`–`v296` y la tabla comparativa estática de `v325`.

1. **Invalidación de Dataset Congelado:** Se confirma que evaluar tareas asociativas sintéticas (MQAR) sobre un split fijo de $N=2000$ ejemplos expone el arnés a fugas de datos (*data leakage*) o memorización parcial de pares clave-valor.
2. **Corrección de Benchmark Baselines:** En `v325`, la precisión reportada para el baseline de Softmax Attention ($93.32\%$) a $L=64$ era un **artefacto de sub-optimización de hiperparámetros** (LR=$10^{-3}$ sin warmup). Al ajustar el LR del baseline a $3\times 10^{-4}$ con warmup de 50 pasos, Softmax Attention alcanza **$99.62\%$** a $L=64$, superando a las arquitecturas lineales en secuencias cortas.
3. **Reorientación del Claims Principal:** El aporte real de DeltaPhase no reside en superar a Softmax Attention en secuencias cortas de 64 tokens (donde Softmax es cercana a perfecta), sino en decodificar en $O(1)$ de memoria RAM sin KV-cache y mantener densidad asociativa en contextos extendidos ($L \ge 1024$).

---

## 1. Configuración del Experimento y Protocolo Dinámico

* **Harness:** `run_v305_fixed_mqar_harness_kaggle.py`
* **Generación de Datos:** Dinámica al vuelo por batch en GPU (`on-the-fly batch generation`), garantizando secuencias 100% inéditas por iteración sin solapamiento de splits.
* **Dispositivo:** PyTorch 2.0 (DirectML / CUDA)
* **Semillas Evaluadas:** 5 semillas independientes ($seed \in \{42, 43, 44, 45, 46\}$)
* **Hiperparámetros de Tarea:**
  * Longitud de Secuencia ($L$): 64 tokens
  * Pares Key-Value ($N_{\text{pairs}}$): 16 pares por secuencia
  * Vocabulario de Claves/Valores ($V$): 256 tokens candidatos
  * Dimensión de Memoria ($d_k$): 32 por cabeza

---

## 2. Resultados Empíricos Comparativos (Generación Dinámica)

| Modelo / Arquitectura | Tipo de Memoria / Cómputo | Presupuesto Params | Dynamic Val Acc (5 Semillas) | SE (Error Estándar) | Etiqueta |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Softmax Attention (MHA)** | Cuadrático $O(N^2)$ + KV-Cache | 1.1 M | **$99.62\%$** | $\pm 0.02\%$ | [ANCLA] |
| **Gated DeltaNet (Yang 2024)** | Lineal $O(N)$ Real ($\mathbb{R}^{d_k \times d_v}$) | 1.1 M | **$96.14\%$** | $\pm 0.18\%$ | [SEÑAL] |
| **DeltaPhase (Nuestra)** | Lineal $O(N)$ Complejo ($S^1 \subset \mathbb{C}^{d_k \times d_k}$) | 1.1 M | **$98.41\%$** | $\pm 0.12\%$ | [SEÑAL] |

*Observación en Tabla:* En la celda de DeltaPhase, la precisión en dinámica pura alcanza $98.41\%$, superando a Gated DeltaNet real en **+2.27%** de exactitud debido a la menor interferencia destructiva de los fasores unimodulares en $S^1$.

---

## 3. Curva de Capacidad de Memoria a Contexto Extendida ($L=1024$)

Para evaluar el límite de saturación del estado de memoria constante ($32 \times 32$), se varió el número de pares $KV$ ($N_{\text{pairs}}$) integrados en secuencias de 1024 tokens:

| Pares $KV$ ($N_{\text{pairs}}$) en $L=1024$ | Gated DeltaNet Real ($\mathbb{R}$) | DeltaPhase Complejo ($S^1 \subset \mathbb{C}$) | Comportamiento Observado |
| :---: | :---: | :---: | :--- |
| **16 pares** | $98.10\%$ | **$99.98\%$** | Retención limpia en ambos núcleos |
| **32 pares** | $89.45\%$ | **$99.95\%$** | DeltaNet empieza a degradar |
| **64 pares** | $64.20\%$ | **$91.30\%$** | Degradación moderada por interferencia |
| **128 pares** | $31.50\%$ | **$61.80\%$** | Colapso por saturación de capacidad |

---

## 4. Amenazas a la Validez

1. **Memory-Bound de Transformadas Espectrales en PyTorch Puro:** Aunque la reducción teórica de FLOPs en el FFN es del 48%, la implementación en PyTorch puro no incluye kernels fusionados en Triton para FWHT/DCT. En GPU, esto resulta en un incremento real de velocidad de solo el ~22% frente a matmuls densos.
2. **Evaluación Limitada a Tareas Sintéticas:** Los resultados del arnés MQAR sugieren alta densidad asociativa, pero no garantizan fluidez o coherencia sintáctica en modelado de lenguaje natural real (TinyStories / Pile).
3. **Complejidad de Gradiente en $L > 4096$:** Aunque el test de gradcheck en FP64 pasa con $7.39 \times 10^{-16}$ de error relativo L2, en precisiones reducidas (FP16/BF16) la acumulación de operaciones por chunks requiere técnicas de normalización causal estricta para evitar la propagación de errores numéricos.

---

## 5. Master Ledger Entry

```json
{"experiment_id": "v305_dynamic_mqar", "fecha": "2026-08-12", "familia": "espectral_holografico", "dataset": "MQAR Dynamic On-The-Fly (L=64/1024)", "n_eval": 2000, "metric_name": "acc", "value": 98.41, "SE": 0.12, "params": 1100000, "nivel_rigor": 1, "etiqueta": "SEÑAL"}
```
