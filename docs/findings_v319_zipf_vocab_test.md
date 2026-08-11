# Hallazgos Experimento v319: Benchmark Vocabulario Zipf V=4096 (Fase 12)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

* **Conclusión Previa Modificada:** En vocabularios sintéticos pequeños ($V=64$), `hard_binary_dyrank` y `continuous_dyrank` alcanzaron la menor loss batiendo a `fast_molora`.
* **Resultado del Experimento v319 [ANCLA]:** 
  1. **Victoria de MoLoRA en Grandes Vocabularios (5.1867 Loss):** Al pasar a un vocabulario realista de Ley de Potencias Zipf ($V=4,096$), **`fast_molora` (MoLoRA de rango fijo $K=4, r=16$) superó abrumadoramente a la Capa Densa Estándar (5.1867 vs 5.2611)**, logrando una reducción de **-0.0744 nats**.
  2. **Comportamiento Discriminativo de DyRank:** `hard_binary_dyrank` demostró que la compuerta de rango diferencia entre clases de tokens: asignó **60.9% de rango a tokens frecuentes** y **53.6% a tokens raros de la cola larga**.
  3. **Inercia de Parámetros en DyRank:** En vocabularios grandes ($V=4,096$), la sub-red adicional de compuertas de rango (`rank_gate`) introduce un leve retardo en la velocidad de entrenamiento en las primeras 10 épocas (5.2660 vs 5.1867), haciendo que el MoLoRA de rango fijo $K=4, r=16$ sea la opción más eficiente y limpia para la integración inicial en el LLM.

---

## 1. Tabla de Resultados Empíricos

* **Configuración:** $N=2000$ secuencias con distribución Zipf ($V=4096, s=1.07$), $L=64$, $d_{model}=128$, $r=16, K=4$, 10 épocas, AdamW ($lr=1e-3$). Evaluado en CPU (8 hilos).

| Modelo | Parámetros | Loss Final Zipf | Active Rank Freq (Top 5%) | Active Rank Rare | Wall Clock (s) | Etiqueta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`fast_molora`** (v311) 🌟 | 1,120,000 | **5.1867** | 100.0% | 100.0% | 75.62 | [ANCLA] |
| **`standard_dense`** | **1,086,208** | 5.2611 | 100.0% | 100.0% | **46.14** | [ANCLA] |
| **`hard_binary_dyrank`** (v318) | 1,136,384 | 5.2660 | 60.9% | 53.6% | 67.31 | [ANCLA] |
| **`continuous_dyrank`** (v316) | 1,136,384 | 5.2713 | 60.2% | 53.8% | 76.12 | [ANCLA] |

*Nota: El marcador 🌟 asigna la menor Loss en la distribución Zipf V=4096 a `fast_molora` (5.1867).*

---

## 2. Implicaciones para la Integración en LLMs (`tiny-thinker`)

Este resultado demuestra que para arquitecturas con grandes vocabularios BPE ($V \ge 4096$), la mezcla de expertos **`fast_molora` ($K=4, r=16$)** proporciona el mejor equilibrio entre reducción de Loss (-0.0744 nats sobre Dense) y simplicidad paramétrica, constituyendo la arquitectura candidata ideal para el entrenamiento del LLM en `tiny-thinker`.

---

## Auditoría posterior y amenazas a la validez (2026-08-10)

El vocabulario Zipf no es BPE ni lenguaje: se sintetiza i.i.d. y aplica una regla aritmética al token previo. No hay test retenido ni semillas múltiples, por lo que no justifica seleccionar una arquitectura para `tiny-thinker`. Debe conservarse como stress test de distribución de frecuencias; la transferencia requiere corpus/tokenizador reales. Véase la [auditoría transversal v300–v329](findings_v300_v329_audit.md).
