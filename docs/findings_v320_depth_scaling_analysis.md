# Hallazgos Experimento v320: Análisis de Rango Capa por Capa en 8 Capas Profundas (Fase 13)

---

## 0. Sección Obligatoria de Reconciliación: Auditoría y Objetividad Científica

* **Conclusión Previa Auditada:** En experimentos previos en redes superficiales (2 capas), la arquitectura MoLoRA superó a la capa Densa.
* **Resultado Certificado del Experimento v320 [ANCLA]:** **VICTORIA CLARA Y ABSOLUTA DE LA CAPA DENSA ESTÁNDAR (`standard_dense`).**
  1. **La Capa Densa Gana en Todas las Métricas:** A 8 capas de profundidad, **`standard_dense` batió a todos los adaptadores de bajo rango**:
     * **Menor Loss:** **3.4760** vs 3.4814 (`hard_binary_dyrank`) vs 3.4872 (`fast_molora`).
     * **Mayor Eficiencia Paramétrica (PEI):** **0.0556** vs 0.0518 (`hard_binary_dyrank`).
     * **Mayor Rapidez (Wall Clock):** **23.59s** vs 152.67s (6.5x más rápida).
  2. **Explicación Algorítmica Objetiva:** En una red profunda de 8 capas residuales, la profundidad lineal pura ($8 \times d_{model}$) proporciona capacidad suficiente para resolver la tarea. Forzar un cuello de botella de bajo rango ($r=16$) y compuertas dinámicas en cada capa actúa como un **obstáculo de restricción de rango**, ralentizando el entrenamiento y aumentando el coste computacional sin aportar beneficio sobre la densidad directa.

---

## 1. Tabla de Resultados Empíricos

* **Configuración:** $N=2000$ secuencias estructuradas, 8 capas residuales profundas, $L=64$, $d_{model}=128$, $r=16, K=4$, 10 épocas, AdamW ($lr=1e-3$). Evaluado en CPU (8 hilos).

| Modelo | Parámetros | Loss Final (8 Capas) | Esparcidad Media por Capa | Wall Clock (s) | PEI | Etiqueta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`standard_dense`** 🌟 | **149,824** | **3.4760** | **0.0%** | **23.59** | **0.0556** | [ANCLA] |
| **`hard_binary_dyrank`** (v320) | 350,528 | 3.4814 | 52.6% (Zero Sparsity 0/1) | 152.67 | 0.0518 | [ANCLA] |
| **`fast_molora`** (v311) | 284,992 | 3.4872 | 0.0% | 94.48 | 0.0526 | [ANCLA] |

*Nota: El marcador 🌟 asigna la victoria objetiva en todas las dimensiones a `standard_dense` (3.4760 Loss, 149K params, 23.59s).*

---

## 2. Perfil Detallado por Capas (`hard_binary_dyrank`)

| Capa Residual | Active Rank % | Zero Sparsity (0/1) % |
| :--- | :---: | :---: |
| **Capa 1** | 48.3% | 51.7% |
| **Capa 2** | 45.9% | 54.1% |
| **Capa 3** | 45.5% | 54.5% |
| **Capa 4** | 48.3% | 51.7% |
| **Capa 5** | 48.2% | 51.8% |
| **Capa 6** | 50.1% | 49.9% |
| **Capa 7** | 47.3% | 52.7% |
| **Capa 8** | 47.0% | 53.0% |
