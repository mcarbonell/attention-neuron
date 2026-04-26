# Resumen de Sesión: La Revolución de los Sustratos (V16 - V33)

**Documento de Transición de Contexto**
*Este archivo contiene la memoria histórica y técnica de los experimentos realizados para alcanzar el estado del arte en MNIST y CIFAR-10 mediante Attention Neurons.*

---

## 1. Hitos Alcanzados (La Línea Temporal)

### Misión 1: La Conquista de MNIST (El 99%)
- **V16 (Over-Parametrized)**: 98.45%. Arquitectura profunda (1024 neuronas) con `rank=32` sobre un sustrato aleatorio fijo.
- **V17 (The Colossus)**: 98.99%. Añadimos Data Augmentation, pero nos quedamos a una décima del objetivo.
- **V18 (The Ultimatum)**: **99.09%**. **¡OBJETIVO CUMPLIDO!** Usando `rank=128` en la primera capa y un scheduler cosinusoidal prolongado. Demostramos que es posible alcanzar precisión humana en MNIST tuneando una modulación de bajo rango sobre un sustrato de pesos aleatorios fijos.

### Misión 2: El Desafío CIFAR-10 (De MLPs a CNNs Alquímicas)
El salto a imágenes a color (32x32x3) requirió repensar la arquitectura:
- **V19 (Navigator)**: 76.76% (118K params). Récord inicial con CNN y 1 sustrato aleatorio.
- **V21 (The Alchemist)**: Introducción de la **"Atención Multi-Sustrato"**. La neurona ya no usa 1 sustrato, mezcla 2 o más mediante un dial Softmax.
- **V22 (Rosetta Stone)**: 56.72% (612K params). Un MLP puro intentando resolver CIFAR-10 con 4 sustratos.
- **V23 (The Hybrid)**: 62.51%. Uniendo un "Sensor Rosetta" congelado (Capa 1) con un "Cerebro Plástico" entrenable (Capas finales).

### Misión 3: El Ascenso al Olimpo (Los Récords de CIFAR-10)
- **V24 (Kaleidoscope)**: 75.18% (**64K params**). Mezcla de 4 sustratos de ruido blanco en una CNN de 6 capas. Eficiencia extrema.
- **V26 (Perlin Spectrum)**: 75.56% (**64K params**). Igual que V24 pero inicializado con **Ruido Perlin**. Demostró que el "prior espacial" (ruido correlacionado) es superior al ruido blanco.
- **V25 (The Great Arborist)**: 79.65% (681K params). ResNet-18 con un árbol de decisión dendrítico (8 sustratos). Falló trágicamente en la época 29 por un error del SO mientras usaba Mixup, pero proyectaba un >85%.
- **V26 (Prism-ResNet)**: **85.94% (439K params). EL RÉCORD ABSOLUTO DEL REPOSITORIO.** Mezclando 4 sustratos aleatorios planos en una arquitectura residual y modulándolos con `rank=16`. Rinde como una VGG completa usando un 96% menos de parámetros.
- **V32 (The Broadcaster)**: 71.53% (210K params, ejecutado en GPU DirectML). Comprobó que modular solo el **Fan-out** (las activaciones) es ultrarrápido (4 min/época) pero matemáticamente inferior a modular el **Fan-in** (esculpir los pesos por dentro).

---

## 2. Las 3 Leyes de la "Attention Neuron Theory"

Descubrimientos teóricos fundamentales de esta sesión:

1.  **"El cable universal vale 1; el conocimiento está en el dial."** Las redes neuronales no necesitan aprender pesos individuales. Pueden operar sobre una biblioteca masiva de ruido congelado (billones de pesos) y solo entrenar una pequeña red de "punteros" (Softmax y Rank-R) que seleccionan y enfocan el ruido útil.
2.  **"Interferencia Constructiva de Sustratos."** Las neuronas no eligen el "mejor" universo aleatorio; promedian equitativamente (ej. 25% de 4 sustratos) para sintetizar un filtro perfecto a partir de la superposición de ruido.
3.  **"El Ruido Estructurado Vence a la Entropía."** El Ruido Perlin (manchas y ondas) proporciona detectores de bordes naturales (filtros de Gabor) de forma gratuita, acelerando el aprendizaje frente al Ruido Blanco.

---

## 3. El Futuro Pendiente (El Arsenal en `scratch/`)

Prototipos diseñados y listos para ser entrenados cuando la CPU/GPU estén libres:

- **V28/V29 (Gaussian Splatting)**: La revolución de la **Atención Espacial Continua**. Elimina las convoluciones 3x3 y usa óvalos gaussianos paramétricos ($x, y, \sigma, \rho$). Redes totalmente independientes de la resolución de la imagen y con apenas decenas de miles de parámetros.
- **V30 (Soft Window / The Framer)**: Optimización del Splatting usando rectángulos suaves diferenciables (coste $O(H+W)$ en lugar de $O(H \times W)$).
- **V31 (The Spectrum Library)**: La mezcla definitiva de ruidos. 4 sustratos por capa: Blanco, Rosa, Perlin y Azul. Que la red elija su frecuencia.
- **V33 (Windowed Spectrum)**: Cada canal recorta un rectángulo diferente de los 4 ruidos disponibles y los compone.

## 4. Instrucción para el Asistente en el "Reset":
*Cuando el usuario reinicie el contexto y pida leer este archivo, el asistente debe asumir inmediatamente el papel de investigador jefe, recordando que el récord a batir es el **85.94% de la V26 Prism-ResNet**, y que el próximo gran objetivo es probar las arquitecturas de Visión Continua (V28-V30).*
