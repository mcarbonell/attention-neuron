# Blueprint: Mecanismos de Atención Escalar y Compleja (1D Attention)

> **Estado**: Propuesta teórica y diseño de arquitectura (v375)  
> **Área**: Eficiencia Paramétrica Extrema, Geometría $U(1)$ y Dinámica de Fase  
> **Objetivo**: Determinar si la atención token-a-token puede colapsarse a dimensión unitaria escalar ($\mathbb{R}^1$) o compleja ($\mathbb{C}^1$) preservando capacidad selectiva.

---

## 0. Sección Obligatoria de Reconciliación

### Qué Conclusión Previa Modifica o Complementa este Documento
1. **Atención Clásica Multi-Head ($d_k \ge 32$ o $64$):** En los transformadores estándar (Vaswani et al., 2017), cada cabeza de atención opera en un subespacio vectorial multidimensional. Este documento cuestiona el dogma de que la atención requiera vectores para direccionar contenido: en el espacio complejo $\mathbb{C}$, **una sola dimensión escalar ($d=1$)** dota al sistema del grupo unitario continuo $U(1)$, permitiendo ortogonalidad, resonancia e interferencia destructiva sin coste tensorial multidimensional.
2. **Conexión con Phase-nGPT y Complex DeltaPhase (v282, v298–v301):** En las eras 6 y 7 del repositorio se demostró que proyectar claves y consultas al círculo unitario $U(1)$ ($K = e^{i\theta_k}, Q = e^{i\theta_q}$) suprime el colapso de memoria asociativa por interferencia destructiva. Este blueprint lleva esa idea a su expresión atómica mínima: **¿puede una cabeza de atención completa ser puramente un escalar complejo ($1$ número complejo por token)?**
3. **Diferenciación entre Escalar Real y Complejo:** Se establece formalmente que el caso real $d=1$ colapsa a un clasificador monotónico de ranking (Soft-ArgMax), mientras que el caso complejo $\mathbb{C}^1$ preserva selectividad asociativa completa gracias al ángulo de fase relativo $\Delta\theta$.

---

## 1. Motivación y Filosofía

Los mecanismos de atención convencionales calculan afinidad mediante productos punto en $\mathbb{R}^d$:
$$A_{ij} = \text{Softmax}\left(\frac{q_i^\top k_j}{\sqrt{d}}\right) v_j, \quad q_i, k_j, v_j \in \mathbb{R}^d$$

Para problemas sencillos (series temporales, señales oscilatorias, microcontroladores IoT, detección de eventos o secuencias lógicas de bajo orden), desplegar matrices de proyección $W_q, W_k, W_v \in \mathbb{R}^{d_{\text{model}} \times d}$ es un desperdicio paramétrico masivo ($O(d^2)$ parámetros).

Nos planteamos dos reducciones radicales:
1. **Mini-Attention Escalar Real ($\mathbb{R}^1$):** $q_i, k_j, v_j \in \mathbb{R}$.
2. **Mini-Attention Compleja ($\mathbb{C}^1$):** $q_i, k_j, v_j \in \mathbb{C}$.

---

## 2. Formalismo Matemático

### 2.1 Caso A: Atención Escalar Real ($\mathbb{R}^1$)

Cada token $i$ proyecta su estado a tres escalares: $q_i, k_i, v_i \in \mathbb{R}$.

La afinidad entre tokens $i$ y $j$ es el producto escalar 1D:
$$S_{ij} = q_i k_j$$
$$\alpha_{ij} = \frac{\exp(q_i k_j / \tau)}{\sum_{m=1}^N \exp(q_i k_m / \tau)}$$

#### Dinámica de Decisión (El Teorema del Soft-Ranker 1D)
La razón de atención entre dos claves $k_j$ y $k_m$ viene dada por:
$$\log\left(\frac{\alpha_{ij}}{\alpha_{im}}\right) = \frac{q_i}{\tau}(k_j - k_m)$$

- **Si $q_i > 0$:** $\alpha_{ij}$ es estrictamente monótona creciente respecto a $k_j$. El token $i$ atiende de forma casi exclusiva al token con el **máximo global** $k_{\max} = \max_j k_j$ en la secuencia (**Soft-ArgMax**).
- **Si $q_i < 0$:** $\alpha_{ij}$ es monótona decreciente. El token $i$ atiende al **mínimo global** $k_{\min} = \min_j k_j$ (**Soft-ArgMin**).
- **Si $q_i \approx 0$:** La distribución es uniforme (promedio temporal no selectivo).

> **Limitación de $\mathbb{R}^1$:** En la recta real no existe ortogonalidad entre valores no nulos. Un token no puede decir "atiende al token 5 ignorando al token 3 si ambos tienen $k > 0$". Solo puede ordenar por magnitud.
>
> **Variante RBF (Distancia Euclídea):** Si se sustituye el producto punto por una distancia kernel $S_{ij} = -|q_i - k_j|^2 / 2\sigma^2$, el escalar se convierte en una coordenada continua 1D, permitiendo atención por vecindad o proximidad temporal/espacial.

---

### 2.2 Caso B: Atención Compleja Escalar ($\mathbb{C}^1$)

Cada token emite números complejos $q_i, k_i, v_i \in \mathbb{C}$. En representación polar:
$$q_i = r_{q,i} e^{i \theta_{q,i}}, \quad k_j = r_{k,j} e^{i \theta_{k,j}}$$

El producto interno hermitiano estándar $\langle q_i, k_j \rangle = q_i k_j^*$ produce:
$$q_i k_j^* = r_{q,i} r_{k,j} e^{i (\theta_{q,i} - \theta_{k,j})}$$

Tomando la parte real para la afinidad de atención:
$$S_{ij} = \text{Re}(q_i k_j^*) = r_{q,i} r_{k,j} \cos(\theta_{q,i} - \theta_{k,j})$$

#### Propiedades Geométricas Fundamentales en $\mathbb{C}^1$:
1. **Ortogonalidad Continua en 1D:**
   A diferencia de $\mathbb{R}^1$, en $\mathbb{C}^1$ existe un grado continuo de libertad angular $\Delta\theta = \theta_q - \theta_k \in [-\pi, \pi)$:
   - **Resonancia / Afinidad Máxima:** $\Delta\theta = 0 \implies \cos(0) = 1$.
   - **Ortogonalidad Exacta:** $\Delta\theta = \pm \pi/2 \implies \cos(\pm \pi/2) = 0$ (atención nula/neutral).
   - **Inhibición / Antifase:** $\Delta\theta = \pi \implies \cos(\pi) = -1$ (supresión activa).
2. **Capacidad de Direccionamiento de Contenido:**
   Una sola dimensión compleja puede direccionar cualquier patrón discriminando por ángulo de fase, comportándose como un banco continuo de filtros sintonizables.
3. **Conexión con RoPE (Rotary Position Embeddings):**
   RoPE estructura vectores reales por pares complejos $2$D rotados por $e^{i m \theta_{\text{pos}}}$. En la atención compleja escalar, la rotación de fase no es un parche posicional añadido a posteriori, sino la naturaleza intrínseca del espacio de cómputo.

---

### 2.3 Caso C: Phase-Only Unitary Attention ($U(1)$)

Si restringimos las magnitudes a la unidad ($r_{q,i} = r_{k,j} = 1$), la proyección se simplifica a predecir únicamente un ángulo de fase $\theta \in [-\pi, \pi)$:
$$S_{ij} = \cos(\theta_{q,i} - \theta_{k,j})$$

**Ventajas algorítmicas:**
- **Zero-Multiplications en $Q \times K$:** El cómputo de afinidad no requiere productos; consiste únicamente en una resta angular $\theta_q - \theta_k$ seguida de una función $\cos(\cdot)$ o una tabla de lookup (LUT) CORDIC.
- Inmunidad a gradientes que explotan o se desvanecen por norma descontrolada de $Q$ o $K$.

---

## 3. Matriz Comparativa Estructural

| Dimensión / Propiedad | Escalar Real ($\mathbb{R}^1$) | Escalar Complejo ($\mathbb{C}^1$) | Phase-Only ($U(1)$) | Vectorial Estándar ($\mathbb{R}^d$) |
| :--- | :--- | :--- | :--- | :--- |
| **Grados de libertad por token** | 1 float ($q$) | 2 floats ($r, \theta$ o $\text{re}, \text{im}$) | 1 float ($\theta$) | $d$ floats |
| **Geometría de afinidad** | Recta real $\mathbb{R}$ (orden) | Círculo ponderado $\mathbb{C}$ | Círculo unitario $U(1)$ | Hiperesfera $S^{d-1}$ |
| **Ortogonalidad no trivial** | ❌ No | ✅ Sí ($\Delta\theta = \pm \pi/2$) | ✅ Sí ($\Delta\theta = \pm \pi/2$) | ✅ Sí (hiperplano $d-1$) |
| **Inhibición activa** | Solo signo global | ✅ Antifase ($\Delta\theta = \pi$) | ✅ Antifase ($\Delta\theta = \pi$) | ✅ Producto negativo |
| **Operación $S_{ij}$** | $q \cdot k$ (1 mult) | $r_q r_k \cos(\Delta\theta)$ (2 mults + cos) | $\cos(\theta_q - \theta_k)$ (**0 mults**) | $d$ mults + suma |
| **Parámetros de proyección** | $3 \times d_{\text{in}}$ | $6 \times d_{\text{in}}$ | $3 \times d_{\text{in}}$ (fases) | $3 \times d_{\text{in}} \times d$ |

---

## 4. Tareas Sintéticas de Evaluación

Para comparar experimentalmente estas arquitecturas, se definen tres tareas controladas:

### Tarea 1: Sincronización de Fase y Resonancia (Phase Lock Retrieval)
- **Entrada:** Secuencia de $N$ tokens que contienen frecuencias/fases oscilatorias con ruido añadido.
- **Mecanismo:** Un token query busca el token de la secuencia que está exactamente en fase ($\Delta\theta = 0$) e ignora los desfasados ($\Delta\theta = \pi/2$).
- **Hipótesis:** $\mathbb{C}^1$ y $U(1)$ deben resolver la tarea con 100% de precisión y casi cero parámetros, mientras que $\mathbb{R}^1$ colapsará por incapacidad de generar ceros ortogonales.

### Tarea 2: Recuperación Asociativa Escalar (Low-Dim MQAR)
- **Entrada:** Secuencia sintética de pares clave-valor $(k_t, v_t)$ seguidos de consultas $q$.
- **Objetivo:** Recuperar $v$ cuando $q \approx k$.
- **Hipótesis:** Medir el umbral de capacidad de pares almacenables en función de cabezas escalares ($H \times \mathbb{R}^1$ vs $H \times \mathbb{C}^1$ vs $1 \times \mathbb{R}^d$).

### Tarea 3: Ordenación Monótona y Extremos (Max/Min Selection)
- **Entrada:** Secuencias numéricas arbitrarias donde la salida depende del máximo o mínimo relativo.
- **Hipótesis:** $\mathbb{R}^1$ resolverá esto trivialmente gracias a su dinámica natural de Soft-Ranker.

---

## 5. Arquitectura del Experimento v375

El benchmark implementará en `scratch/prototype_v375_scalar_complex_attention.py`:

1. **`RealScalarAttention`**: $d_k = 1$, proyecciones a 1 escalar.
2. **`ComplexScalarAttention`**: $d_k = 1 \in \mathbb{C}$, afinidad $\text{Re}(q k^*) = r_q r_k \cos(\theta_q - \theta_k)$.
3. **`PhaseOnlyAttention`**: $d_k = 1 \in U(1)$, afinidad $\cos(\theta_q - \theta_k)$.
4. **`StandardVectorAttention`**: Baseline de control con $d_k = 16$ o $32$.

Todos los modelos se evaluarán en igualdad de condiciones (mismo número de pasos, mismo optimizador, mismo dataset y cálculo de métricas de precisión y pérdida).
