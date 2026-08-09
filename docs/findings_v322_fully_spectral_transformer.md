# Hallazgos Experimento v322: Fully Spectral Block (All-Spectral Transformer, Fase 1)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

* **Conclusión Previa Modificada:** En `v321` se demostró que sustituir un bloque FFN aislado por `spectral_phase_ffn` superaba a un FFN denso aislado. Se hipotetizó que unificar la mezcla de secuencia de fase con FFNs espectrales en un Transformer 100% espectral (`fully_spectral`) superaría a LLaMA estándar.
* **Resultado Certificado del Experimento v322 [ANCLA]:** **EVALUACIÓN OBJETIVA DE CAPACIDAD.**
  1. **Compresión Paramétrica del 63.6%:** `fully_spectral` redujo los parámetros totales del modelo de **412,352 $\to$ 149,952** (eliminando 262,400 parámetros de FFNs densas).
  2. **Brecha de Capacidad en Modulación Simple:** `standard_llama` alcanzó una menor loss (**2.1035 vs 3.0398**) gracias a la enorme capacidad expresiva de sus FFNs densos de 512 neuronas ocultas ($8 d^2$).
  3. **Identificación de la Vía de Mejora para `v323`:** La modulación espectral uniforme lineal de `v322` es demasiado rígida. Para igualar la capacidad de LLaMA con $2.75\times$ menos parámetros, la capa espectral requiere **modulación adaptativa de frecuencia (SpecGate)**.

---

## 1. Tabla de Resultados Empíricos

* **Configuración:** $N=2000$ secuencias estructuradas, $L=64$, $d_{model}=128$, 10 épocas, AdamW ($lr=1e-3, \text{weight\_decay}=0.0$). Evaluado en CPU (8 hilos).

| Modelo Transformer | Parámetros | Loss Final | Wall Clock (s) | PEI | Etiqueta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`standard_llama`** 🌟 | **412,352** | **2.1035** | **49.41** | **0.0847** | [ANCLA] |
| **`fully_spectral`** (v322) | 149,952 | 3.0398 | 52.18 | 0.0636 | [ANCLA] |

*Nota: El marcador 🌟 asigna la menor Loss y mayor PEI a `standard_llama` (2.1035 Loss, 412K params).*

---

## 2. Análisis Algorítmico y Siguiente Paso (`v323`)

La diferencia de 0.93 nats entre la arquitectura espectral (150K params) y LLaMA (412K params) se debe a que la modulación trigonométrica de $O(d)$ parámetros en `v322` actúa como un filtro pasabanda rígido. En el experimento `v323`, introduciremos **SpecGate (Compuertas Adaptativas de Frecuencia Espectral)** para permitir que la red abra y cierre canales de frecuencia de forma dinámica por token.
