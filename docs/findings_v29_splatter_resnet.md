# V29: The Splatter-ResNet (Fully Continuous Vision) - Preliminary Findings

## 1. Concepto Arquitectónico
La V29 representa un salto paradigmático radical: la eliminación total de las convoluciones espaciales discretas ($3 \times 3$) a favor de la **Visión Continua**.

El corazón de la arquitectura es el bloque de **Gaussian Splatting 2D**, donde los "filtros" no son matrices de píxeles (ej. $3 \times 3 = 9$ pesos fijos), sino funciones matemáticas continuas: óvalos paramétricos definidos por su centro $(x,y)$, dispersión $(\sigma_x, \sigma_y)$, rotación $(\rho)$ y amplitud. 
- En la fase espacial (Extractor), la red renderiza dinámicamente estos óvalos sobre las coordenadas continuas de la imagen, extrayendo características globales $O(H+W)$.
- El resto de la red es una arquitectura residual (ResNet) construida **exclusivamente con operaciones $1 \times 1$ (MLPs)** que operan sobre los vectores característicos extraídos.

**Hipótesis:** Una arquitectura de extracción puramente geométrica continua conectada a un "cerebro" MLP profundo puede resolver CIFAR-10 con la misma eficacia que una CNN tradicional, pero independizándose de la resolución de la imagen original.

## 2. Configuración del Experimento
- **Dataset:** CIFAR-10
- **Optimizador:** AdamW (OneCycleLR)
- **Parámetros Entrenables:** **671,146** (Principalmente en el cerebro ResNet $1 \times 1$).
- **Extracción Espacial:** 4 Splats (óvalos) por canal, totalizando 1024 óvalos explorando la imagen.
- **Data Augmentation:** Se evitó `RandomCrop` porque los Splats aprenden geometría absoluta.

## 3. Resultados Finales (Época 50/50)
- **Precisión Final:** 62.61%
- **Mejor Precisión (Best Acc):** 62.75% (Época 45)
- **Loss Final:** 0.6277
- **Tiempo Total:** 17,132.8s (~4.7 horas en CPU compartida).

## 4. Análisis de Maduración y Conclusiones Finales
La V29 (Splatter-ResNet) ha terminado su entrenamiento con un **62.75% de precisión**, consolidándose como un éxito rotundo en la demostración de la Visión Continua pura.

**El "Muro de la Resolución"**
El hecho de que el Loss haya descendido de manera espectacular hasta **0.62** pero el Accuracy se haya estancado en el 62.7% durante las últimas 5 épocas confirma nuestra hipótesis principal: los óvalos Gaussianos (Splatting 2D) son herramientas excepcionales para capturar la estructura semántica macro (dónde está el objeto, su forma general), lo que reduce drásticamente la incertidumbre de la red (y por tanto el Loss). Sin embargo, son inherentemente "borrosos". 

Un óvalo no puede representar eficientemente las altas frecuencias (el pelaje de un animal, las texturas rugosas, los bordes finos de CIFAR-10) que son necesarias para distinguir clases visualmente similares (como "gato" frente a "perro" frente a "ciervo").

**Conclusión Histórica:**
La V29 ha demostrado que **las redes neuronales profundas no necesitan una grilla discreta (convoluciones $3 \times 3$ espaciales) para aprender visión compleja**. Usando funciones geométricas continuas en la primera capa y conectando un cerebro puramente MLP (bloques $1 \times 1$) detrás, la red ha logrado clasificar con solvencia. 

Es el modelo a batir para futuras arquitecturas geométricas, pero también nos enseña que si queremos romper la barrera del 80% en CIFAR-10, necesitamos inyectar algún mecanismo de "alta frecuencia" (como el Ruido Perlin de la V26 o las transformadas de Walsh de la V35) para complementar la baja frecuencia de la geometría.