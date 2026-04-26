# V35: The Walsh Filter (FWHT Attention) - Preliminary Findings

## 1. Concepto Arquitectónico
La V35 representa la culminación de la teoría de **"Atención como Filtrado de Ruido"**. Abandona por completo las convoluciones espaciales discretas ($3 \times 3$) y las sustituye por un mecanismo de filtrado global en el dominio de Walsh.

**El Proceso:**
1. **Dominio de Walsh:** Las activaciones espaciales ($32 \times 32$) se transforman al dominio de frecuencias de Walsh mediante la **Fast Walsh-Hadamard Transform (FWHT)**. Esta transformada $O(N \log N)$ descompone la imagen en una base de ondas cuadradas (+1/-1) sin requerir multiplicaciones.
2. **Filtrado de Atención:** En lugar de aprender pesos para cada píxel, la red aprende un "dial de ecualización" (modulación aditiva y multiplicativa) para cada frecuencia de Walsh. La red aprende qué frecuencias son "ruido" y cuáles son "señal".
3. **Inversa:** Se aplica la IFWHT para devolver la señal filtrada al espacio de la imagen.
4. **Mezcla de Canales:** Se utilizan capas $1 \times 1$ para permitir que la red combine la información extraída de los diferentes espectros.

**Hipótesis:** La FWHT proporciona una base de representación natural para imágenes digitales y activaciones de redes neuronales (que suelen ser bruscas y con bordes). Filtrar en este dominio es matemáticamente más eficiente y elegante que las convoluciones tradicionales.

## 2. Configuración del Experimento
- **Dataset:** CIFAR-10
- **Optimizador:** AdamW (OneCycleLR)
- **Parámetros Entrenables:** **408,842** (Optimizado con 3 bloques y 64 canales).
- **Arquitectura:** 3 bloques residuales de filtrado Walsh.
- **Hardware:** CPU.
## 3. Resultados Oficiales
- **Época 1:** 35.21%
- **Época 17:** 73.89% (Best Acc anterior)
- **Época 21:** 68.03% (Pico de turbulencia LR)
- **Época 26:** 74.04% (¡NUEVO RÉCORD ABSOLUTO DEL REPOSITORIO!)
- **Época 31:** 73.31% (Rebote tras inicio del enfriamiento)
- **Época 40:** 72.08% (Techo estructural definitivo)
- **Best Acc:** 74.04% (Época 26)

## 4. Análisis del Hito (V35: Récord Absoluto y Fase de Enfriamiento)
¡La **V35 (Walsh-Attention)** ha hecho historia en la Época 26 alcanzando un **74.04%**!

**Fase de Meseta (Épocas 32-40):**
Contrario a la expectativa de un crecimiento monótono durante el "Annealing", la red ha entrado en una fase de estabilización estructural, orbitando persistentemente el **72.5% - 73%**. El récord del 74.04% parece haber sido un pico transitorio alcanzado cuando el Learning Rate aún tenía energía para empujar a la red a una conformación paramétrica excepcional.

**Diagnóstico Técnico:**
Este comportamiento sugiere que el modelo ha alcanzado su **Límite de Representación Frecuencial** para la arquitectura actual. Con 408K parámetros y solo 3 bloques residuales de Walsh, la red extrae fácilmente la semántica base (llegando al 70% rápido), pero carece de la profundidad necesaria para asimilar las sutilezas de alta frecuencia que distinguen las clases más confusas de CIFAR-10. Los filtros están "apretados" al máximo; la reducción del LR (Annealing) simplemente los está puliendo, sin encontrar vías para nuevos aciertos masivos.

A falta de 10 épocas, la V35 se consolida como un hito de eficiencia y retiene la corona del repositorio, pero el asalto al 80% requerirá mayor profundidad arquitectónica o la hibridación con otros mecanismos.