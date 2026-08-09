# Idea & Brainstorming: CORDIC-Inspired Unitary Phase Attention

> 💡 **DOCUMENTO DE EXPLORACIÓN CONCEPTUAL:** Conexión entre la atención de fase compleja (`DeltaPhase`) y el algoritmo CORDIC (*Coordinate Rotation Digital Computer*) para aceleración en hardware y cuantización discreta de fase.

---

## 1. Motivación e Intuición

En el algoritmo CORDIC (inventado por Jack Volder en 1959), cualquier rotación de un vector complejo $z = x + i y = r e^{i\theta}$ se realiza sin multiplicaciones, utilizando únicamente **sumas y desplazamientos de bits (bit-shifts)**.

Un ángulo $\theta$ se descompone como una serie de micro-rotaciones binarias:
$$\theta = \sum_{i=0}^{B-1} d_i \alpha_i, \quad \text{donde } \alpha_i = \arctan(2^{-i}) \quad \text{y } d_i \in \{-1, +1\}$$

En el plano complejo, la actualización en el paso $i$ viene dada por:
$$x_{i+1} = x_i - d_i \cdot (y_i \gg i)$$
$$y_{i+1} = y_i + d_i \cdot (x_i \gg i)$$

Donde $\gg i$ representa un desplazamiento a la derecha de $i$ bits (multiplicación por $2^{-i}$). La escala total $K = \prod_{i=0}^{B-1} \sqrt{1 + 2^{-2i}} \approx 1.64676$ es un factor constante que se aplica una sola vez al final.

---

## 2. Aplicación a la Atención de Fase Compleja (`DeltaPhase`)

Actualmente, en la arquitectura `ChunkwiseComplexDeltaPhase`, las fases complejas se calculan mediante funciones trascendentales:
$$K = e^{i \theta_k} = \cos(\theta_k) + i \sin(\theta_k)$$

Esto requiere cómputo de coma flotante denso (`torch.cos`, `torch.sin`, `torch.polar`). La perspectiva CORDIC ofrece tres ventajas clave:

### 2.1 Cuantización Extrema de Fase (Bit-Shift Attention)
Si representamos el vector de fases de cada cabeza de atención como un vector de decisiones binarias $d = (d_0, d_1, \dots, d_{B-1}) \in \{-1, +1\}^B$:
- La rotación de las claves y consultas no requiere multiplicadores complejas de punto flotante.
- Se ejecuta mediante **$B$ pasos de acumulación con desplazamiento de bits**, ideal para aceleradores TPU/FPGA/ASIC o GPUs con INT8/INT4.

### 2.2 Preservación Nativa del Círculo Unitario $S^1$
Al igual que en CORDIC, las micro-rotaciones preservan la norma del vector estado (salvo la constante $K$). Esto garantiza que la actualización del estado de memoria asociativo $M_t$ permanezca acotada en el círculo unitario $S^1 \subset \mathbb{C}^{d_k}$, evitando la explosión o desvanecimiento del gradiente sin necesidad de normalizaciones L2 costosas por paso.

### 2.3 Jerarquía de Ángulos: Semántica Gruesa vs Posición Fina
La secuencia de ángulos $\alpha_i = \arctan(2^{-i})$ sigue una jerarquía geométrica:
- **Pasos iniciales ($i = 0, 1, 2$):** $\alpha_0 = 45^\circ, \alpha_1 \approx 26.57^\circ, \alpha_2 \approx 14.04^\circ$. Determinan la orientación semántica global del token.
- **Pasos avanzados ($i \ge 4$):** Ajustes angulares finos que codifican la posición relativa y previenen la interferencia asociativa en secuencias de largo contexto.

---

## 3. Hoja de Ruta de Pruebas Futuras

1. **Simulación PyTorch (CORDIC-Quantized DeltaPhase):** Implementar un módulo prototype que reemplace `torch.polar` por $B=8$ iteraciones CORDIC discretas en PyTorch y evaluar la pérdida de perplejidad respecto a $\mathbb{C}$ continuo.
2. **Eficiencia en Inferencia:** Medir el ahorro de energía/latencia teórica en hardware entero (INT8) al sustituir multiplicaciones complejas por desplazamientos de bits.
