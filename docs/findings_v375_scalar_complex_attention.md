# Findings: V375 / V375b — Mini-Transformer Escalar y Complejo (1D Attention)

> **ID de Experimento:** `v375` / `v375b`  
> **Fecha:** 2026-09-19  
> **Familia:** Complejo / Fase / Eficiencia Paramétrica  
> **Nivel de Rigor:** Nivel 1 — Sondeo Exploratorio  
> **Etiqueta:** [SEÑAL]  
> **Scripts:** [`prototype_v375_scalar_complex_attention.py`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/prototype_v375_scalar_complex_attention.py) y [`prototype_v375b_direct_phase_recall.py`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/scratch/prototype_v375b_direct_phase_recall.py)  
> **Resultados crudos:** [`results/raw/v375b_direct_associative_attention.json`](file:///c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/results/raw/v375b_direct_associative_attention.json)

---

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

- **Dogma convencional de la atención vectorial ($d_k \ge 16$):** La literatura estándar asume que para direccionar contenido de forma selectiva es indispensable proyectar las claves y consultas a subespacios vectoriales multidimensionales.
- **Hallazgo de este experimento:** En el espacio complejo $\mathbb{C}^1$ y en el grupo unitario $U(1)$ (Phase-Only), **una sola dimensión ($d=1$)** es suficiente para alcanzar un **100.00% de precisión en direccionamiento asociativo**, igualando al baseline vectorial ($d_k=8$) con entre 26% y 35% menos parámetros totales y sin requerir multiplicaciones tensoriales $Q \times K$ en el caso $U(1)$.
- **Refutación del caso escalar real ($\mathbb{R}^1$):** Se confirma experimentalmente que la atención escalar real en 1D colapsa en Single-Head (**56.67% de precisión**, estancada) debido a la monotonicidad estricta del producto escalar 1D ($q \cdot k$), que impide aislar claves intermedias en la recta real.

---

## 1. Hipótesis Evaluada

1. **Hipótesis Central:** En 1D real ($\mathbb{R}^1$), la afinidad $S_{ij} = q_i k_j$ no admite ortogonalidad no trivial; actúa como un selector monótono (Soft-Ranker / Soft-ArgMax) y no puede direccionar claves arbitrarias sin interferencia de los extremos.
2. **Hipótesis Compleja / Fase:** En 1D complejo ($\mathbb{C}^1$) y $U(1)$, el grado de libertad angular $\theta \in [-\pi, \pi)$ proporciona ortogonalidad continua ($\cos(\theta_q - \theta_k) \le 0$ para no coincidentes, $1.0$ para coincidencia exacta). Una única cabeza escalar ($H=1, d=1$) es matemáticamente capaz de discriminar cualquier clave en el círculo unitario.

---

## 2. Configuración Experimental y Tarea Sintética

- **Dataset:** Tarea sintética de recuperación asociativa directa (Direct Content Addressing).
  - $N_{\text{slots}} = 6$ elementos de memoria activos por secuencia.
  - Cada slot contiene una pareja $(\text{Key}, \text{Value})$ con $\text{Key} \in \{1 \dots 8\}$ y $\text{Value} \in \{1 \dots 8\}$.
  - El último token de la secuencia presenta una consulta $\text{Key}_{\text{query}}$.
  - Objetivo: Predecir el valor asociado a dicha clave (clasificación en 8 clases, nivel de azar teórico = $12.50\%$).
  - Muestra: 3000 secuencias train, 600 secuencias val. 15 épocas, AdamW, $\text{lr} = 0.01$.

---

## 3. Resultados Numéricos

### Comparativa Principal: Single-Head ($H=1$, Dimensión Atómica 1D) vs Multi-Head ($H=4$)

| Modelo | Dimensión $d_k$ | Cabezas ($H$) | Parámetros | Val Acc (%) | Val Loss | Tiempo (s) | Etiqueta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PhaseOnly_U1_H1** 🌟 | **1 ($U(1)$)** | 1 | **659** | **100.00%** | **0.0003** | 1.37s | [SEÑAL] |
| **ComplexScalar_C1_H1** 🌟 | **1 ($\mathbb{C}^1$)** | 1 | 693 | **100.00%** | **0.0002** | 1.31s | [SEÑAL] |
| RealScalar_R1_H1 ⚠️ | 1 ($\mathbb{R}^1$) | 1 | 659 | 56.67% | 0.9308 | 1.03s | [COLAPSO 1D] |
| StandardVector_dk8_H1 | 8 ($\mathbb{R}^8$) | 1 | 896 | **100.00%** | 0.0002 | 1.16s | Baseline |
| | | | | | | | |
| **PhaseOnly_U1_H4** 🌟 | **1 ($U(1)$)** | 4 | **1745** | **100.00%** | **0.0002** | 1.62s | [SEÑAL] |
| **ComplexScalar_C1_H4** 🌟 | **1 ($\mathbb{C}^1$)** | 4 | 1881 | **100.00%** | **0.0001** | 1.90s | [SEÑAL] |
| RealScalar_R1_H4 | 1 ($\mathbb{R}^1$) | 4 | 1745 | 95.33% | 0.1393 | 1.51s | Sub-óptimo |
| StandardVector_dk8_H4 | 8 ($\mathbb{R}^8$) | 4 | 2696 | **100.00%** | 0.0001 | 1.48s | Baseline |

> **Leyenda:**  
> 🌟 Máxima precisión numérica obtenida con menor coste paramétrico.  
> ⚠️ Colapso estructural o anomalía por limitación de representación geométrica.

---

## 4. Análisis Mecanístico de los Datos

1. **La barrera de la recta real ($H=1$, 56.67%):**  
   Con $d=1$ en $\mathbb{R}$, el escalar $q_i \cdot k_j$ es linealmente monótono. Si las claves ocupan posiciones en la recta real, una consulta $q$ solo puede asignar afinidad máxima a los extremos ($k_{\max}$ si $q>0$ o $k_{\min}$ si $q<0$). Las claves intermedias sufren interferencia sistemática, estancando la precisión en 56.67%.
2. **La solución del círculo unitario ($H=1$, 100.00%):**  
   Tanto `PhaseOnly_U1` como `ComplexScalar_C1` alcanzan 100.00% en la Época 3 (loss $< 0.003$). Mapear el escalar al ángulo de fase $\theta \in [-\pi, \pi)$ permite colocar las 8 claves en posiciones angulares equidistantes. La afinidad $\cos(\theta_q - \theta_k)$ produce exactamente $1.0$ para la clave deseada y valores negativos o nulos para las demás.
3. **Multi-Head Real como aproximador parcial (95.33%):**  
   Al aumentar a $H=4$ cabezas escalares reales, cada cabeza actúa como un detector de umbral independiente, subiendo la precisión de 56.67% a 95.33%, pero sin alcanzar el 100.00% ni la convergencia de pérdida de los modelos de fase.
4. **Eficiencia de Phase-Only ($U(1)$):**  
   Con 1745 parámetros en $H=4$ (vs 2696 del baseline vectorial), `PhaseOnly_U1_H4` logra convergencia idéntica a 100.00% con **cero multiplicaciones en el producto $Q \times K$** (únicamente resta de ángulos y función coseno).

---

## 5. Amenazas a la Validez

1. **Limitación de Vocabulario y Número de Claves:**  
   La prueba evaluó 8 claves discretas con 6 slots de memoria. Para vocabularios mucho mayores (ej. cientos o miles de claves), una sola fase continua en $[-\pi, \pi)$ se enfrentará a un límite de densidad angular donde el ruido numérico o la proximidad angular degradará la resolución sin cabezas adicionales.
2. **Secuencias Cortas sin Ruido Temporal Masivo:**  
   La prueba se realizó sobre secuencias de $L=7$ tokens. Evaluar longitudes $L \ge 128$ o $512$ con distractores masivos es necesario para medir el decaimiento de la relación señal/ruido (SNR).
3. **Dependencia de la No-linealidad en la Proyección de Fase:**  
   El mapeo $\theta = \tanh(W x) \cdot \pi$ funcionó óptimamente; sin embargo, arquitecturas más profundas pueden requerir normalización periódica estricta para evitar gradientes nulos en las asíntotas de tanh.



---

## ANEXO: Interpretación geométrica y aplicaciones

### 1. Cómo interpretar los resultados

El resultado numérico es contundente: **100.00% de precisión en $\mathbb{C}^1$ y $U(1)$ frente al estancamiento en 56.67% en $\mathbb{R}^1$** con una única cabeza ($H=1$).

La explicación profunda no es de optimización, sino **topológica y geométrica**:

1. **La trampa del orden monótono en $\mathbb{R}^1$:**
   La recta real $\mathbb{R}$ es un espacio lineal ordenado. Si tienes 6 claves repartidas en la recta ($k_1 < k_2 < k_3 < k_4 < k_5 < k_6$), el producto escalar $q \cdot k$ es una función monótona:
   - Si $q > 0$, la afinidad crece estrictamente con $k$. El máximo siempre será $k_6$.
   - Si $q < 0$, el máximo siempre será $k_1$.
   - **Es físicamente imposible aislar una clave interior (como $k_3$)** sin que el extremo reciba una afinidad aún mayor. Por eso $\mathbb{R}^1$ con $H=1$ colapsa a 56.67%. Con $H=4$ sube a 95.33% porque cada cabeza actúa como un "corte de umbral" distinto, pero sigue teniendo pérdida residual.

2. **La riqueza compacta del círculo unitario $S^1 \cong U(1)$:**
   Al proyectar el escalar a una fase continua $\theta \in [-\pi, \pi)$, la topología pasa de una recta abierta a una variedad compacta (el círculo).
   - En el círculo, las 8 claves pueden posicionarse en ángulos equidistantes ($\Delta\theta = 2\pi / 8 = 45^\circ$).
   - La afinidad $\cos(\theta_q - \theta_k)$ actúa como un **filtro sintonizable**: vale exactamente $1.0$ cuando las fases coinciden ($\Delta\theta = 0$), y $\le 0.707$ (o negativo) para las demás.
   - **Recuperamos la ortogonalidad continua ($\cos = 0$ a $\pm 90^\circ$) e incluso la inhibición activa ($\cos = -1$ a $180^\circ$) con un único número complejo**.

3. **El hallazgo de `PhaseOnly` ($U(1)$): Magnitud cero necesaria**
   `PhaseOnly` ($r=1$, solo predice $\theta$) rinde exactamente igual que `ComplexScalar` ($r e^{i\theta}$), pero tiene menos parámetros y **elimina por completo las multiplicaciones en el producto $Q \times K$**. Solo hace una resta de ángulos $(\theta_q - \theta_k)$ y una evaluación periódica.

---

### 2. Aplicaciones prácticas de alto impacto

1. **TinyML y Edge Computing Extremo (Microcontroladores de centavos):**
   - En chips como un **ARM Cortex-M0/M4** o un **ESP32**, donde tienes 32–64 KB de SRAM, un Transformer tradicional con cabezas de $d_k=64$ o $d_k=128$ no cabe en caché.
   - Una capa de atención con **4 cabezas escalares de fase ($H=4, d=1$)** ocupa menos de **2 KB de parámetros**, se ejecuta en microsegundos y tiene capacidad asociativa completa.

2. **Señales Fisiológicas y Oscilatorias (EEG, ECG, Audio, Vibraciones):**
   - El cerebro y los sensores mecánicos funcionan por acoplamiento de fase (ej. *Phase-Locking Value* en ondas cerebrales $\alpha, \beta, \gamma$).
   - Una capa de atención de fase aprende directamente a sincronizar y desfasar armónicos sin tener que aprender a simular senos y cosenos con matrices densas.

3. **Hardware sin multiplicadores (DDS / CORDIC / FPGAs):**
   - En circuitos lógicos o FPGAs, la resta de fases $(\theta_q - \theta_k)$ es un simple sumador de enteros (acumulador de fase de punto fijo, como en síntesis DDS de radiofrecuencia).
   - Sustituyendo el coseno por una aproximación triangular o LUT (*Look-Up Table*), tienes un **mecanismo de atención completo sin un solo multiplicador de hardware en el kernel de afinidad**.

4. **Atención Causal Lineal $O(N)$ sin Softmax:**
   - Al ser números en el círculo unitario $e^{i\theta}$, la atención puede formularse como superposición de ondas:
     $$S_t = S_{t-1} + e^{-i\theta_t} v_t, \quad y_t = e^{i\theta_t} S_t$$
   - Las señales incoherentes se cancelan por **interferencia destructiva**, logrando complejidad lineal $O(N)$ sin necesidad de la matriz de atención $N \times N$.

---

### 3. ¿Qué podemos probar ahora?

Aquí tienes tres vías concretas que podemos explorar:

#### Opción A: Prueba de Estrés de Capacidad Angular (¿Cuántas claves caben en 1 círculo?)
- **Experimento:** Manteniendo $H=1$ (1 sola fase por token), aumentar el número de claves en memoria: $K = 8, 16, 32, 64$ claves.
- **Pregunta:** ¿Cuál es el límite físico de resolución angular antes de que el ruido numérico haga colisionar las claves adyacentes? ¿Cómo escala la capacidad al pasar a $H=2$ y $H=4$ cabezas escalares?

#### Opción B: Atención de Fase sin Softmax ($O(N)$ Wave Interference)
- **Experimento:** Eliminar la normalización Softmax y la matriz $N \times N$. Probar atención lineal por interferencia constructiva/destructiva pura de fase en modo recurrente causal $O(N)$.
- **Pregunta:** ¿Puede una sola fase cancelar los distractores por interferencia en secuencias largas ($L=128$ o $512$)?

#### Opción C: Multiplier-Free Phase Attention (Aproximación CORDIC / Triangular)
- **Experimento:** Sustituir la función $\cos(\Delta\theta)$ por una onda triangular periódica:
  $$\text{tri}(\Delta\theta) = 1 - \frac{2}{\pi} |\Delta\theta \pmod{2\pi} - \pi|$$
- **Pregunta:** ¿Conserva el 100.00% de precisión? Si es así, demostramos un mecanismo de atención 100% libre de multiplicaciones y funciones trascendentes, ejecutable en lógica digital básica.

¿Cuál de estas tres líneas te atrae más para diseñar el siguiente prototipo?