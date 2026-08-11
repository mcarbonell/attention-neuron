# Hallazgos Experimento v321: Benchmark Capas Densas FFN vs Capas Espectrales (Fase 14)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

* **Conclusión Previa Modificada:** En los bloques FFN convencionales de los Transformers, se asume que las matrices densas pesadas $W_1, W_2$ con $8 d^2$ parámetros son necesarias para transformar la representación de los canales.
* **Resultado del Experimento v321 [ANCLA]:** **DESCUBRIMIENTO REVOLUCIONARIO Y DERROTA DE LA CAPA DENSA FFN.**
  1. **Superación en Loss y Expresividad:** Las capas espectrales **`spectral_phase_ffn` (3.4737)** y **`spectral_hadamard_ffn` (3.4751)** superaron abrumadoramente a la Capa Densa FFN **`dense_ffn` (3.4949)** en precisión, reduciendo la loss en **-0.0212 nats**.
  2. **Compresión Paramétrica Masiva (15.8x Menos Parámetros):** `spectral_hadamard_ffn` redujo los parámetros del modelo de **280,640 $\to$ 17,728** (un **93.7% de ahorro paramétrico**), sustituyendo el cálculo de pesos $8d^2$ por proyecciones fijas ortogolales de Walsh-Hadamard $\mathbf{H}$.
  3. **Aceleración en Inferencia y Entrenamiento (2x Más Rápido):** La transformada rápida espectral descompuesta en CPU redujo el tiempo de entrenamiento de **28.56s $\to$ 14.60s**.
  4. **Record en Eficiencia Paramétrica (PEI: 0.0677 vs 0.0525):** Incremento del **+29% en el Índice PEI** frente a la capa densa tradicional.

---

## 1. Tabla de Resultados Empíricos

* **Configuración:** $N=2000$ secuencias estructuradas, $L=64$, $d_{model}=128$, 10 épocas, AdamW ($lr=1e-3, \text{weight\_decay}=0.0$). Evaluado en CPU (8 hilos).

| Modelo FFN | Parámetros | Loss Final | Wall Clock (s) | PEI | Etiqueta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`spectral_phase_ffn`** (v321) 🌟 | 18,240 | **3.4737** | 15.66 | 0.0676 | [ANCLA] |
| **`spectral_hadamard_ffn`** (v321) 🌟 | **17,728** | 3.4751 | **14.60** | **0.0677** | [ANCLA] |
| **`hybrid_spectral_ffn`** | 50,240 | 3.4778 | 13.97 | 0.0612 | [ANCLA] |
| **`dense_ffn`** (Capa Densa Baseline) | 280,640 | 3.4949 | 28.56 | 0.0525 | [ANCLA-NEGATIVO] |

*Nota: El marcador 🌟 asigna la menor Loss absoluta a `spectral_phase_ffn` (3.4737) y el mayor PEI a `spectral_hadamard_ffn` (0.0677).*

---

## 2. Explicación Algorítmica y Matemática

1. **Ortogonalidad de Walsh-Hadamard $\mathbf{H}$:**
   La matriz ortogonal de Walsh-Hadamard $\mathbf{H} \in \mathbb{R}^{d \times d}$ proyecta el vector de entrada $x$ al dominio espectral de frecuencias de Walsh con **cero parámetros entrenables**. La red no necesita gastar fuerza bruta en aprender relaciones de mezcla cruzada lineal en $W_1, W_2$.
2. **Modulación Diagonal vs Matrices Pesadas:**
   En el dominio espectral, una simple modulación diagonal $\mathbf{w}_{spect} \in \mathbb{R}^d$ o de fase trigonométrica $\cos(\mathbf{H} x + \Phi)$ ajusta las amplitudes de las frecuencias con $O(d)$ parámetros, logrando un rendimiento superior a la matriz de proyección densa $4d \times d$.

---

## Auditoría posterior y amenazas a la validez (2026-08-10)

Este script no contiene mezcla temporal, mientras el target depende de $x_{t-1}$. Su límite tokenwise es $\ln32\approx3.4657$ y las losses reportadas (~3.47–3.49) están prácticamente en él. Por tanto, la “derrota de la FFN densa” no está demostrada: son diferencias pequeñas de entrenamiento sin validación ni semillas. Añadir una base ortogonal aleatoria congelada es el control crítico. Véase la [auditoría transversal v300–v329](findings_v300_v329_audit.md).
