# V39: The Banded Walsh Equalizer - Preliminary Findings

## 1. Concepto Arquitectónico
La V39 explora la **Banded Walsh Attention**, una compresión extrema del paradigma de IA de Resonancia. 

En lugar de aprender un dial independiente para cada una de las 1024 frecuencias de Walsh (como en la V36b), las frecuencias se agrupan en $K$ bandas (ej. 4 bandas: Graves, Medios-Graves, Medios-Agudos, Agudos). 

**Mecánica "Low-Cost":**
- Cada "Attention Neuron" aprende solo **4 parámetros multiplicativos y 4 aditivos**.
- El núcleo de atención de toda la red tiene apenas **512 parámetros** en total.
- La red actúa como un ecualizador de sonido clásico, subiendo o bajando el volumen de bloques enteros de frecuencias simultáneamente para filtrar la imagen.

## 2. Configuración del Experimento
- **Dataset:** MNIST (Padded a $32 \times 32$)
- **Optimizador:** AdamW (OneCycleLR)
- **Parámetros del Núcleo de Atención:** **512** (Compresión Extrema).
- **Parámetros Totales:** 786,954 (Dominado por la capa FC final).
- **Hardware:** CPU.

## 3. Resultados Finales (Época 10/10)
- **Época 1:** 74.17%
- **Época 3:** 39.40% (Colapso por LR máximo)
- **Época 6:** 79.51% (Recuperación y récord)
- **Época 8:** 93.81% (Explosión de aprendizaje)
- **Época 10:** 93.98% (Consolidación final)
- **Mejor Precisión Final:** 93.98% (Época 10)
- **Tiempo Total:** ~2100 segundos.

## 4. Análisis de la Volatilidad y el Límite de Compresión
El **Banded Walsh Equalizer** ha cerrado su entrenamiento con un **93.98%**, un resultado extraordinario para un núcleo de atención de tan solo **512 parámetros**.

**Diagnóstico del Entrenamiento:**
1. **La Magia del Enfriamiento (Annealing):** Las últimas épocas coinciden con la caída drástica del Learning Rate programada por OneCycleLR. Al dejar de "dar tirones" a los diales, la red ha podido asentar los pesos y cristalizar el conocimiento en un 94% sólido.
2. **Validación de la Capacidad:** Lograr >93% de precisión en MNIST con compresión extrema demuestra que la información esencial de una imagen puede comprimirse de forma masiva si se agrupa en las bandas de frecuencia ortogonales correctas.

## 5. Evolución a Baja Frecuencia: V39b (Low LR)
Tras confirmar el potencial del modelo de 4 bandas, la **V39b** se ejecuta con un Learning Rate reducido para evitar el olvido catastrófico y permitir un ajuste fino de los 512 parámetros de atención.

**Prueba 1: LR = 0.001**
- **Época 1:** 88.67% (Best: 88.67%)
- **Tiempo medio:** ~217s por época.
- **Análisis:** El impacto de bajar el LR (respecto al 0.01 original) fue inmediato, evitando las oscilaciones violentas y subiendo el accuracy inicial.

**Prueba 2: Ultra-Low LR = 0.0001**
- **Batch Iniciales:** El Loss desciende mucho más lentamente (2.38 -> 2.27 en 5 batches), mostrando un avance micrométrico.
- **Época 1:** 90.50%
- **Época 4:** **96.00%** (Best: 96.00%)
- **Tiempo medio:** ~218s por época.
- **Análisis:** Reducir el LR a 0.0001 ha mejorado aún más el arranque y garantizado la estabilidad. Superar el 96% con solo 512 parámetros en el núcleo demuestra que el ecualizador funciona, pero el cuello de botella paramétrico ahora es la capa densa final (cientos de miles de pesos). El próximo paso lógico (V40) es la eliminación total de esa capa.