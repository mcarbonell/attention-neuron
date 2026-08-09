# Hallazgos Experimento v317: Conformal Spherical MoLoRA (Proyección en S^(n-1), Fase 10)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

* **Conclusión Previa Modificada:** Se hipotetizó que restringir las representaciones de bajo rango a la hiper-esfera unitaria $\mathbb{S}^{n-1}$ mediante proyectores $L_2$ aportaría una mayor precisión de representación que los adaptadores Euclídeos libres en $\mathbb{R}$.
* **Resultado del Experimento v317 [ANCLA]:** 
  1. **Estabilidad Esférica y Rendimiento:** `conformal_spherical_molora` alcanzó una loss de **3.4763**, superando a la Capa Densa Estándar (**3.4819**) y manteniéndose competitiva con `fast_molora` (**3.4751**).
  2. **Overhead de Normalización en CPU:** La doble normalización $L_2$ (`F.normalize` en la dimensión $r$ y $d_{out}$) incrementó el tiempo de cómputo en Wall Clock a **44.02s** (vs 29.70s en `fast_molora`), debido al cálculo recurrente de normas vectoriales e inversas.
  3. **Liderazgo de DyRank MoLoRA:** **DyRank MoLoRA (`v316`)** mantiene la menor loss absoluta del benchmark (**3.4748**) al combinar alta precisión con esparcidad de rango activa (57.9%).

---

## 1. Tabla de Resultados Empíricos

* **Configuración:** $N=2000$ secuencias estructuradas, $L=64$, $d_{model}=128$, $r=16, K=4$, 10 épocas, AdamW ($lr=1e-3$). Evaluado en CPU (8 hilos).

| Modelo | Parámetros | Loss Final | Wall Clock (s) | PEI | Etiqueta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`dyrank_molora`** (v316) 🌟 | 100,160 | **3.4748** | 38.95 | 0.0575 | [ANCLA] |
| **`fast_molora`** (v311) | 83,776 | 3.4751 | **29.70** | 0.0585 | [ANCLA] |
| **`conformal_spherical_molora`** (v317) | 83,776 | 3.4763 | 44.02 | 0.0584 | [ANCLA] |
| **`standard_dense`** | **49,984** | 3.4819 | **10.81** | **0.0611** | [ANCLA] |

*Nota: El marcador 🌟 asigna la menor Loss a `dyrank_molora` (3.4748).*

---

## 2. Análisis del Desempeño

1. **Invarianza de Escala:**
   Al normalizar en $\mathbb{S}^{r-1}$ y $\mathbb{S}^{d-1}$, las activaciones intermedias quedan confinadas al rango $[-1, 1]$, garantizando resistencia a la explosión de magnitud.
2. **Trade-off Computacional:**
   Aunque la geometría esférica estabiliza el mapa angular, el cómputo de `sqrt(sum(x^2))` en CPU introduce una sobrecarga del +48% en latencia frente a `fast_molora`.
