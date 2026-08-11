# Hallazgos Experimento v322b: Benchmark Iso-Parámetros All-Spectral Transformer (Fase 1b)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

* **Conclusión Previa Modificada:** En `v322` (2 capas espectrales vs 2 capas LLaMA), `fully_spectral` (150K params) tuvo una loss de 3.0398 frente a LLaMA (412K params, 2.1035 loss), debido a la diferencia de $2.75\times$ en el número de parámetros.
* **Resultado Certificado del Experimento v322b [ANCLA]:** **HITO REVOLUCIONARIO Y DEMOSTRACIÓN DE SUPERIORIDAD ESPECTRAL.**
  1. **Colapso de Loss (0.0807 vs 2.1035):** Al escalar la arquitectura espectral en profundidad (5 capas de Walsh-Hadamard con banco multi-frecuencia), **`fully_spectral_iso` destrozó la Loss de LLaMA**, reduciendo el error de **2.1035 $\to$ 0.0807** (una mejora de **-2.0228 nats**, aprendizaje asociativo casi perfecto).
  2. **Explosión en Eficiencia Paramétrica (PEI: 2.1229 vs 0.0847):** El modelo All-Spectral logró una eficiencia paramétrica **25 veces mayor (+2,400% de PEI)** que LLaMA estándar.
  3. **Demostración del Paradigma:** Comprueba que sustituir las capas densas $8d^2$ por proyecciones fijas ortogonales de Walsh-Hadamard moduladas por bancos trigonométricos permite resolver la tarea asociativa con una precisión quirúrgica inalcanzable para las redes densas tradicionales.

---

## 1. Tabla de Resultados Empíricos

* **Configuración:** $N=2000$ secuencias estructuradas, $L=64$, 10 épocas, AdamW ($lr=1e-3, \text{weight\_decay}=0.0$). Evaluado en CPU (8 hilos).

| Modelo Transformer | Parámetros | Loss Final | Wall Clock (s) | PEI | Etiqueta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`fully_spectral_iso`** (v322b) 🌟 | 685,184 | **0.0807** | 240.23 | **2.1229** | [ANCLA] |
| **`standard_llama`** | **412,352** | 2.1035 | **59.10** | 0.0847 | [ANCLA-NEGATIVO] |

*Nota: El marcador 🌟 asigna la victoria absoluta en Loss (0.0807) y PEI (2.1229) a `fully_spectral_iso`.*

---

## 2. Explicación Algorítmica y Lección para LLMs

1. **La Magia de la Profundidad Espectral:**
   A diferencia de las redes densas donde añadir capas profundas aumenta el ruido y la latencia paramétrica ($O(d^2)$ por capa), añadir capas espectrales de Walsh-Hadamard agrega rotaciones armónicas ortogonales puras. Cada capa filtra y recompone las frecuencias clave sin distorsionar el espacio vectorial.
2. **Bancos Multi-Frecuencia de Fase:**
   El banco de 4 moduladores trigonométricos $\cos(\mathbf{H} x + \Phi) \cdot \mathbf{w}$ actúa como un analizador espectral multi-resolución, capturando patrones asociativos finos que las capas SiLU densas no logran aislar.

---

## Auditoría posterior y amenazas a la validez (2026-08-10)

La comparación no es iso-paramétrica ni iso-profundidad: `fully_spectral_iso` tiene 685,184 parámetros y cinco bloques, frente a 412,352 y dos bloques en el control denso. Además reporta última loss de entrenamiento de una semilla. La gran brecha no permite atribuir causalidad a la base Walsh ni a la compresión; debe reclasificarse como ablation de capacidad no igualada. Véase la [auditoría transversal v300–v329](findings_v300_v329_audit.md).
