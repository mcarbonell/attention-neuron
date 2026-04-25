# Attention Neuron como Formato de Compresión

*Documento conceptual y operativo sobre una nueva línea de trabajo: usar la Attention Neuron no solo como arquitectura entrenable, sino como representación compacta de redes neuronales.*

---

## 1. Motivación

Hasta ahora, la Attention Neuron se ha estudiado principalmente como:

- arquitectura ultraligera
- parametrización neuron-centric de bajo rango
- posible alternativa al entrenamiento clásico peso a peso
- base para aprendizaje eficiente y hardware-friendly

Pero la propia formulación abre una segunda vía muy potente:

> una red puede no almacenarse como una colección explícita de millones de pesos, sino como una semilla reproducible + un pequeño conjunto de parámetros neuronales de modulación

En otras palabras:

- una red densa clásica almacena directamente todas sus conexiones
- una red Attention Neuron almacena **las reglas para reconstruir esas conexiones**

Eso convierte a la arquitectura en un posible **formato de compresión de redes neuronales**.

---

## 2. Idea Central

En una capa Attention Neuron, la matriz efectiva de pesos no se guarda explícitamente como un tensor libre. Se reconstruye a partir de:

1. un sustrato pseudoaleatorio reproducible `W_init`
2. una modulación multiplicativa de bajo rango
3. una corrección aditiva de bajo rango
4. un bias o fase por neurona

Por ejemplo:

$$
W_{eff} = W_{init} \odot M + A
$$

o en la variante residual:

$$
W_{eff} = W_{init} + W_{init} \odot M + A
$$

donde:

- `W_init` no necesita almacenarse completo si se puede regenerar con una semilla fija
- `M` y `A` se construyen a partir de tensores pequeños ligados a las neuronas

La consecuencia es fuerte:

> ya no es necesario almacenar `O(in * out)` pesos por capa, sino solo `O(r * (in + out))` más metadatos mínimos

---

## 3. Dos Formas de Entender la Compresión

### 3.1 Compresión nativa

La red se entrena directamente en formato comprimido.

En este caso:

- nunca existe una matriz densa libre entrenable
- el modelo final ya nace comprimido

Esta es la línea que el repositorio ya ha estado explorando de forma natural.

### 3.2 Compresión post-hoc

Una red densa ya entrenada se aproxima a posteriori mediante una representación Attention Neuron.

En este caso:

- partimos de una MLP o CNN densa entrenada
- fijamos un sustrato reproducible
- optimizamos solo parámetros neuronales pequeños para imitar la red original

Esta línea sería más cercana a la compresión clásica de modelos o a un "codec neuronal".

---

## 4. Conteo Paramétrico

### 4.1 Capa densa clásica

Para una capa lineal de `in_features = N` y `out_features = M`:

- pesos: `N * M`
- bias: `M`

Total:

$$
P_{dense} = N \cdot M + M
$$

### 4.2 Attention Neuron de rango `r`

Con dos ramas de bajo rango:

- multiplicativa:
  - `delta_in_m`: `M * r`
  - `delta_out_m`: `r * N`
- aditiva:
  - `delta_in_a`: `M * r`
  - `delta_out_a`: `r * N`
- bias o fase:
  - `M`

Total:

$$
P_{attn} = 2Mr + 2Nr + M = 2r(M+N) + M
$$

La semilla del sustrato tiene coste despreciable frente al tamaño de la matriz.

---

## 5. Ejemplos Numéricos

### Ejemplo A: Capa 784 -> 512 con rank=2

#### Densa

$$
784 \cdot 512 + 512 = 401,920
$$

#### Attention Neuron

$$
2 \cdot 2 \cdot (784 + 512) + 512 = 5,696
$$

**Compresión aproximada**

$$
401,920 / 5,696 \approx 70.6x
$$

### Ejemplo B: Capa 1024 -> 1024 con rank=2

#### Densa

$$
1024 \cdot 1024 + 1024 = 1,049,600
$$

#### Attention Neuron

$$
2 \cdot 2 \cdot (1024 + 1024) + 1024 = 9,216
$$

**Compresión aproximada**

$$
1,049,600 / 9,216 \approx 113.9x
$$

### Ejemplo C: Capa 3072 -> 1024 con rank=2

#### Densa

$$
3072 \cdot 1024 + 1024 = 3,146,752
$$

#### Attention Neuron

$$
2 \cdot 2 \cdot (3072 + 1024) + 1024 = 17,408
$$

**Compresión aproximada**

$$
3,146,752 / 17,408 \approx 180.8x
$$

---

## 6. Qué Se Almacenaría Realmente

Un checkpoint Attention Neuron comprimido no necesita guardar cada matriz completa. Bastaría con almacenar:

1. arquitectura del modelo
2. semillas por capa o por bloque
3. tensores pequeños de modulación
4. bias/fase por neurona
5. hiperparámetros de reconstrucción
   - rank
   - variante (`residual`, `phase`, `sparse`, etc.)
   - tipo de inicialización pseudoaleatoria

En formato conceptual:

```text
model/
  layer_1/
    seed = 12345
    rank = 2
    delta_in_m
    delta_out_m
    delta_in_a
    delta_out_a
    theta_bias
  layer_2/
    seed = 67890
    ...
```

Esto convierte el modelo en algo parecido a:

- una receta de reconstrucción
- más que una tabla explícita de pesos

---

## 7. Tipos de Estudio de Compresión

### C1. Estudio estructural

Objetivo:

- comparar parámetros almacenados teóricos
- comparar tamaño de checkpoint real en disco

Medidas:

- número de parámetros entrenables
- tamaño serializado del modelo
- ratio de compresión frente a MLP/CNN densas equivalentes

### C2. Estudio funcional desde cero

Objetivo:

- comparar una red densa entrenada desde cero frente a una Attention Neuron entrenada desde cero con el mismo ancho/profundidad

Medidas:

- accuracy
- tamaño de modelo
- accuracy por KB
- accuracy por parámetro almacenado

### C3. Compresión post-hoc de una red densa

Objetivo:

- entrenar una Attention Neuron para imitar una red densa ya entrenada

Objetivos de imitación posibles:

- logits
- activaciones intermedias
- predicciones finales

Medidas:

- accuracy retenida
- error de reconstrucción
- ratio de compresión

### C4. Estudio de semillas

Objetivo:

- ver cuánto depende la compresión de la semilla del sustrato

Preguntas:

- ¿hay semillas "buenas" y "malas"?
- ¿cuántos bits reales de información está aportando la semilla?
- ¿conviene usar una semilla fija universal o una semilla por capa?

---

## 8. Métricas Recomendadas

Para que esta línea sea seria, no basta con decir "usa menos parámetros". Hay que medir:

### Métricas de compresión

- `stored_parameters`
- `checkpoint_size_bytes`
- `compression_ratio`

### Métricas funcionales

- `final_accuracy`
- `accuracy_retained`
- `logit_mse` o `feature_mse` en compresión post-hoc

### Métricas compuestas

- `accuracy_per_kb`
- `accuracy_per_stored_parameter`
- `compression_vs_accuracy_drop`

### Métricas de coste

- tiempo de reconstrucción de pesos
- tiempo de inferencia
- memoria pico durante inferencia

Esto es importante porque puede ocurrir que:

- el modelo sea muy pequeño en disco
- pero más caro de reconstruir en tiempo real

Eso no invalida la idea, pero define su nicho.

---

## 9. Casos de Uso Potenciales

### 9.1 Distribución de modelos

Enviar checkpoints muy pequeños:

- seed + parámetros neuronales
- en lugar de megabytes o gigabytes de pesos

### 9.2 Edge AI

Dispositivos con muy poca memoria pueden almacenar:

- solo la descripción comprimida
- y regenerar pesos localmente

### 9.3 Fine-tuning ligero

Un modelo base podría compartirse como sustrato fijo, y las adaptaciones especializadas serían solo:

- deltas neuronales pequeños
- más que adaptadores matriciales completos

### 9.4 Neural codec

Una red entrenada podría convertirse a un "código" más compacto basado en:

- semilla
- bajo rango neuronal
- reconstrucción determinista

Esta es quizá la lectura más ambiciosa.

---

## 10. Riesgos y Preguntas Abiertas

### Riesgo 1: Compresión paramétrica no implica compresión funcional

Puede ocurrir que la representación sea pequeña, pero retenga menos comportamiento útil del esperado.

### Riesgo 2: Reconstrucción cara

Si reconstruir `W_eff` cuesta mucho, la ventaja de almacenamiento puede no traducirse bien a inferencia.

### Riesgo 3: Dependencia de semilla

La calidad del sustrato pseudoaleatorio puede ser un cuello oculto.

### Riesgo 4: Comparación injusta

Una red densa y una Attention Neuron no deben compararse solo en número de parámetros, sino en:

- accuracy
- coste de inferencia
- tamaño de checkpoint

---

## 11. Experimentos Iniciales Recomendados

### E1. Tabla de compresión teórica por capa

Construir una tabla para las arquitecturas ya usadas en el repo:

- MLP MNIST 784 -> 512 -> 10
- MLP CIFAR 3072 -> 1024 -> 10
- CNNs factorizadas

Comparar:

- parámetros densos
- parámetros Attention Neuron
- ratio de compresión

### E2. Tamaño real de checkpoint

Guardar en disco:

- una MLP densa
- una Attention Neuron equivalente

Comparar tamaño real del archivo serializado.

### E3. Accuracy por KB

Tomar resultados ya existentes y calcular:

- accuracy / tamaño de modelo

Esto puede ser una de las métricas más llamativas del proyecto.

### E4. Compresión post-hoc mínima

Primera prueba sencilla:

1. entrenar una MLP densa pequeña en MNIST
2. congelarla
3. entrenar una Attention Neuron para imitar sus logits
4. medir cuánta accuracy se retiene con una fracción pequeña del tamaño

---

## 12. Hipótesis Fuerte

La hipótesis fuerte de esta línea sería:

> Una red neuronal puede representarse de forma mucho más compacta como un sustrato pseudoaleatorio reproducible más modulaciones neuronales de bajo rango, en lugar de almacenar explícitamente todos sus pesos.

Si esta hipótesis se valida, la Attention Neuron deja de ser solo:

- una arquitectura alternativa

y pasa a ser también:

- un formato de compresión
- una parametrización compacta
- y potencialmente un codec neuronal

---

## 13. Conclusión

La compresión no es una aplicación lateral de la Attention Neuron; puede convertirse en una de sus consecuencias más potentes.

Lo interesante aquí no es únicamente que "use menos parámetros entrenables", sino que permite reformular qué significa almacenar una red:

- no como una tabla exhaustiva de pesos
- sino como una estructura generativa compacta

Eso abre una vía de investigación nueva, distinta de la pura accuracy, y con mucha fuerza conceptual propia.
