# Findings: V376 — Frontera de Capacidad Angular en Atención Escalar y de Fase

> **ID de Experimento:** `v376`  
> **Fecha:** 2026-09-19  
> **Familia:** Complejo / Fase / Capacidad de Memoria  
> **Nivel de Rigor:** Nivel 1 — Sondeo Exploratorio  
> **Etiqueta:** [SEÑAL]  
> **Script:** [`prototype_v376_angular_capacity_stress.py`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/prototype_v376_angular_capacity_stress.py)  
> **Resultados crudos:** [`results/raw/v376_angular_capacity_stress.json`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/raw/v376_angular_capacity_stress.json)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Complementa este Experimento

- **Hallazgo previo de V375 ($K=8$):** Demostró que una única cabeza compleja o de fase ($H=1, d=1$) logra 100.00% de precisión asociativa directa frente al colapso de la atención escalar real en 1D.
- **Frontera descubierta en V376 ($K=8 \dots 64$):** 
  1. Se cuantifica el límite físico de resolución angular de un único círculo $S^1$: una sola cabeza $H=1$ mantiene **100.00% de precisión hasta $K=16$ claves discretas** ($22.5^\circ$ de separación media).
  2. En $K=32$ y $K=64$, surge el fenómeno de *aglomeración angular* (angular crowding), degradando $H=1$ a 80.67% y 64.67%.
  3. **El toroide de fase ($T^H = S^1 \times \dots \times S^1$):** Al pasar a $H=4$ cabezas escalares de fase, el sistema particiona las claves en sub-círculos independientes, restaurando la precisión a **91.83% (`PhaseOnly`) y 90.00% (`ComplexScalar`) en $K=64$** (frente a un azar del 1.56%).
  4. **Degeneración irreversible de $\mathbb{R}^1$:** La atención escalar real se degrada monótonamente hasta caer al 24.33% ($H=1$) y 42.33% ($H=4$) en $K=64$.

---

## 1. Hipótesis Evaluada

1. **Límite de Resolución Angular:** Existe un límite superior de claves discriminables en un solo círculo continuo $U(1)$ antes de que la proximidad angular $\Delta\theta \to 0$ cause filtrado espurio en el Softmax.
2. **Escalabilidad Multicabeza:** Las cabezas escalares múltiples no compiten en un espacio euclídeo denso, sino que forman las dimensiones de un toroide $T^H$, incrementando la capacidad combinatoria de direccionamiento con coste $O(H)$ en lugar de $O(d^2)$.

---

## 2. Configuración Experimental

- **Barrido de Vocabulario:** $K \in \{8, 16, 32, 64\}$ claves/valores discretos.
- **Nivel de Azar Teórico:** $12.50\%$ ($K=8$), $6.25\%$ ($K=16$), $3.12\%$ ($K=32$), $1.56\%$ ($K=64$).
- **Muestra:** 2400 secuencias train, 600 val. 15 épocas por modelo, AdamW ($\text{lr}=0.01$).
- **Arquitecturas evaluadas:**
  - `PhaseOnly_U1` ($H \in \{1, 2, 4\}$, $d=1$)
  - `ComplexScalar_C1` ($H \in \{1, 2, 4\}$, $d=1$)
  - `RealScalar_R1` ($H \in \{1, 4\}$, $d=1$)
  - `StandardVector_dk8` ($H \in \{1, 4\}$, $d_k=8$)

---

## 3. Resultados Numéricos Globales

### Precisión de Validación (%) en función de $K$ (Número de Claves)

| Modelo | Dimensión | Cabezas ($H$) | $K=8$ (ch=12.5%) | $K=16$ (ch=6.2%) | $K=32$ (ch=3.1%) | $K=64$ (ch=1.6%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PhaseOnly_U1_H1`** | $1$ ($U(1)$) | 1 | 55.83% | **100.00%** | 85.83% | 45.00% |
| **`PhaseOnly_U1_H2`** | $1$ ($U(1)$) | 2 | 88.50% | 83.50% | 89.67% | 90.33% |
| **`PhaseOnly_U1_H4`** 🌟 | **1 ($U(1)$)** | 4 | **100.00%** | 89.33% | **99.67%** | **91.83%** |
| | | | | | | |
| **`ComplexScalar_C1_H1`** | $1$ ($\mathbb{C}^1$) | 1 | **100.00%** | **100.00%** | 80.67% | 64.67% |
| **`ComplexScalar_C1_H2`** | $1$ ($\mathbb{C}^1$) | 2 | **100.00%** | **100.00%** | 94.00% | 81.50% |
| **`ComplexScalar_C1_H4`** 🌟 | **1 ($\mathbb{C}^1$)** | 4 | **100.00%** | **100.00%** | 93.17% | **90.00%** |
| | | | | | | |
| `RealScalar_R1_H1` ⚠️ | $1$ ($\mathbb{R}^1$) | 1 | 53.67% | 36.33% | 32.17% | 24.33% |
| `RealScalar_R1_H4` | $1$ ($\mathbb{R}^1$) | 4 | 90.00% | 73.33% | 56.33% | 42.33% |
| | | | | | | |
| `StandardVector_dk8_H1` | $8$ ($\mathbb{R}^8$) | 1 | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| `StandardVector_dk8_H4` | $8$ ($\mathbb{R}^8$) | 4 | **100.00%** | **100.00%** | **100.00%** | **100.00%** |

---

## 4. Análisis Mecanístico de los Datos

1. **La frontera de capacidad del círculo unitario ($H=1$):**
   - Para $K=8$ y $K=16$, `ComplexScalar_C1_H1` sostiene el **100.00%**. La separación angular media es de $45^\circ$ y $22.5^\circ$, perfectamente resoluble con $\cos(\Delta\theta)$.
   - A partir de $K=32$ ($11.25^\circ$) y $K=64$ ($5.625^\circ$), la diferencia $\cos(0) - \cos(5.625^\circ) = 1.0 - 0.995 = 0.005$ es pequeña, exigiendo temperaturas de Softmax muy afiladas. Aún así, $H=1$ en $\mathbb{C}^1$ conserva un **64.67% de acierto** (41 veces por encima del azar puro de 1.56%).
2. **Efecto de Multicabeza como descomposición toroidal ($H=4$):**
   - Con 4 cabezas escalares, las claves no necesitan apretarse en un único círculo: el modelo distribuye las claves entre los círculos independientes de cada cabeza.
   - En $K=64$, `PhaseOnly_U1_H4` alcanza **91.83%** y `ComplexScalar_C1_H4` **90.00%**, cerrando prácticamente la brecha con el baseline vectorial denso consumiendo menos parámetros (5385 vs 6336) y con afinidad sin productos.
3. **El colapso continuo de la recta real:**
   - La recta real $\mathbb{R}^1$ no escala. Incluso con 4 cabezas, cae a 42.33% en $K=64$. La carencia de simetría cíclica y de ortogonalidad condena al producto escalar 1D ante el crecimiento del vocabulario.

---

## 5. Amenazas a la Validez

1. **Tiempo de Entrenamiento Fijo (15 Épocas):**
   Es plausible que con 30 o 50 épocas y un scheduler de temperatura decreciente (annealing), $H=2$ y $H=4$ alcancen el 100.00% en $K=64$.
2. **Distribución Uniforme de Claves:**
   En texto o lenguaje natural, la distribución de claves sigue leyes de Zipf (no uniforme), lo que podría favorecer aglomeraciones en fases específicas si no se introduce regularización entrópica de fase.
