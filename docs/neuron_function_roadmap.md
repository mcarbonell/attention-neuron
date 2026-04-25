# Roadmap de Funciones Neuronales

*Catalogo de familias de funciones de agregacion candidatas para la linea polimorfica de la Attention Neuron.*

---

## 1. Objetivo

Hasta ahora, la exploracion polimorfica ha probado tres ideas principales:

- `SUM` como agregador clasico
- `MAX` como detector estricto de rasgo dominante
- `L2` como proxy suave, vectorizable y eficiente del comportamiento tipo `MAX`

Los hallazgos mas recientes sugieren una conclusion importante:

- el polimorfismo neuronal **si es una idea fertile**
- la red **si utiliza distintos modos de agregacion**
- pero no todas las familias matematicas son igual de estables o eficientes

Este documento recopila las siguientes familias de funciones candidatas para ampliar esa linea de trabajo.

---

## 2. Principio de Diseño

No interesa probar funciones "raras" solo por novedad. La prioridad debe darse a funciones que cumplan una o varias de estas propiedades:

1. **Semantica clara**
   La funcion debe corresponder a una "personalidad" interpretable de la neurona.

2. **Compatibilidad computacional**
   Debe poder implementarse con operaciones vectorizables, idealmente basadas en matmul o reducciones simples.

3. **Estabilidad numerica**
   No debe introducir gradientes caoticos o rupturas artificiales del signo.

4. **Diferencia funcional real**
   Debe ofrecer un comportamiento cualitativamente distinto a `SUM`.

---

## 3. Familias de Funciones

### F1. SUM

**Formula**

$$
y = \sum_i w_i x_i
$$

**Semantica**

Acumulador de evidencia.

**Interpretacion**

La neurona responde cuando muchas entradas empujan en una direccion compatible.

**Ventajas**

- baseline perfecto
- barato
- estable
- signado de forma natural

**Desventajas**

- no distingue bien entre evidencia repartida y evidencia muy concentrada

**Estado**

Ya validado. Debe seguir siendo el punto de referencia universal.

---

### F2. MAX

**Formula**

$$
y = \max_i (w_i x_i)
$$

**Semantica**

Detector estricto de rasgo dominante.

**Interpretacion**

La neurona se activa si una entrada concreta domina claramente al resto.

**Ventajas**

- semantica muy clara
- induce especializacion fuerte

**Desventajas**

- coste alto en implementacion exacta
- rompe eficiencia matricial
- gradiente menos amigable

**Estado**

Ya explorado en `V13`. Util como idea, pero no como implementacion eficiente.

---

### F3. L2 Energy

**Formula**

$$
y = \sqrt{\sum_i (w_i x_i)^2 + \epsilon}
$$

**Semantica**

Detector de energia o intensidad dominante.

**Interpretacion**

Amplifica entradas grandes y atenúa cancelaciones.

**Ventajas**

- vectorizable
- suave
- proxy muy bueno de `MAX`
- compatible con hardware

**Desventajas**

- pierde parte del caracter "winner-takes-all" del max exacto

**Estado**

Ya validado en `V14`. Es hoy la mejor extension practica de la neurona polimorfica.

---

### F4. RMS

**Formula**

$$
y = \sqrt{\frac{1}{N}\sum_i (w_i x_i)^2 + \epsilon}
$$

**Semantica**

Detector de energia normalizada.

**Interpretacion**

Parecida a `L2`, pero desacopla mas la respuesta de la cantidad total de conexiones.

**Ventajas**

- muy estable
- comparable entre neuronas con distinto fan-in
- semantica limpia

**Desventajas**

- puede parecerse demasiado a `L2` en algunas configuraciones

**Prioridad**

Alta. Es una de las siguientes pruebas mas razonables.

---

### F5. MEAN

**Formula**

$$
y = \frac{1}{N} \sum_i w_i x_i
$$

**Semantica**

Detector de consenso.

**Interpretacion**

La neurona responde a la evidencia media, no a la suma total.

**Ventajas**

- muy barata
- muy interpretable
- desacopla respuesta y fan-in

**Desventajas**

- puede ser demasiado parecida a `SUM` si el fan-in es fijo

**Prioridad**

Media-Alta. Muy util si se quiere separar "evidencia total" de "densidad de evidencia".

---

### F6. L1 Magnitude / ABS-SUM

**Formula**

$$
y_{mag} = \sum_i |w_i x_i|
$$

con alguna forma de restaurar signo o mezclarla con `SUM`.

**Semantica**

Detector de intensidad total.

**Interpretacion**

Mide cuanta señal hay sin permitir que se cancele algebraicamente.

**Ventajas**

- muy interpretable
- barata
- distinta de `SUM`

**Desventajas**

- el signo no sale de forma natural
- puede ser mejor como magnitud auxiliar que como salida directa

**Prioridad**

Media. Buena candidata si se diseña una restauracion de signo suave y robusta.

---

### F7. LogSumExp

**Formula**

$$
y = \frac{1}{\beta}\log\left(\sum_i exp(\beta \cdot w_i x_i)\right)
$$

**Semantica**

Detector suave de rasgo dominante.

**Interpretacion**

Con `beta` pequeño se comporta de forma mas difusa; con `beta` grande se acerca a `MAX`.

**Ventajas**

- diferenciable
- interpretable
- control continuo del "caracter dominante"

**Desventajas**

- riesgo de overflow si no se implementa con cuidado
- mas caro que `SUM` o `L2`

**Prioridad**

Alta. Probablemente la mejor familia siguiente despues de `L2`.

---

### F8. Soft Top-k

**Idea**

No usar el maximo exacto, sino una agregacion donde las entradas grandes pesan mucho mas que las pequeñas.

**Semantica**

Detector de pocos rasgos clave.

**Ventajas**

- mas rica que `MAX`
- evita depender de una sola conexion

**Desventajas**

- implementacion mas compleja
- riesgo de acercarse demasiado a mecanismos de atencion locales mas caros

**Prioridad**

Media-Baja. Interesante, pero no la probaria antes que `RMS` o `LogSumExp`.

---

### F9. Multiplicative / Co-occurrence Aggregation

**Idea**

Agregar entradas de forma que la neurona responda a co-ocurrencia de rasgos y no solo a suma de evidencia.

**Posibles aproximaciones**

- suma de log-magnitudes
- producto suavizado
- mezcla entre `SUM` y una via multiplicativa

**Semantica**

Detector tipo "AND": la neurona necesita varias pistas simultaneas.

**Ventajas**

- funcionalmente muy distinta de `SUM`
- muy interesante para rasgos composicionales

**Desventajas**

- numericamente delicada
- facil de destabilizar
- el signo es problematico

**Prioridad**

Baja por ahora. Conceptualmente muy valiosa, pero no para la siguiente ronda inmediata.

---

### F10. Variance / Contrast

**Idea**

La neurona no mide solo la magnitud media, sino la dispersion o desacuerdo entre las entradas.

**Semantica**

Detector de contraste.

**Ventajas**

- muy distinta de `SUM`
- puede tener sentido en vision o capas tempranas

**Desventajas**

- menos obvia como operador principal
- mas facil que funcione como señal auxiliar que como agregador dominante

**Prioridad**

Baja-Media. Interesante en imagen, menos urgente en MNIST.

---

### F11. Competitive Softmax Pooling

**Idea**

La neurona redistribuye internamente el peso entre entradas usando una normalizacion suave.

**Semantica**

Atencion local dentro de la neurona.

**Ventajas**

- muy interpretable
- fuerza especializacion

**Desventajas**

- mas compleja
- puede salirse de la elegancia actual del metodo
- se acerca a una micro-atencion por neurona

**Prioridad**

Baja por ahora. Muy interesante, pero ya entra en otra familia de arquitectura.

---

## 4. Tabla Resumen

| Familia | Personalidad Neuronal | Coste | Riesgo Numerico | Prioridad |
| :--- | :--- | :--- | :--- | :--- |
| `SUM` | Acumulador de evidencia | Muy bajo | Muy bajo | Referencia |
| `MAX` | Detector estricto | Alto | Medio | Ya explorado |
| `L2` | Detector de energia | Bajo | Bajo | Ya validado |
| `RMS` | Energia normalizada | Bajo | Bajo | Alta |
| `MEAN` | Consenso | Muy bajo | Muy bajo | Media-Alta |
| `ABS-SUM` | Intensidad total | Bajo | Medio | Media |
| `LogSumExp` | Dominante suave | Medio | Medio | Alta |
| `Soft Top-k` | Pocos rasgos clave | Medio-Alto | Medio | Media-Baja |
| `Multiplicative` | Co-ocurrencia | Medio-Alto | Alto | Baja |
| `Variance` | Contraste | Medio | Medio | Baja-Media |
| `Softmax Pooling` | Atencion local | Alto | Medio | Baja |

---

## 5. Secuencia Recomendada

Si el objetivo es avanzar con rapidez y bajo riesgo, la secuencia recomendada es:

### Fase 1

1. `SUM` vs `RMS`
2. `SUM` vs `LogSumExp`
3. `SUM` vs `MEAN`

### Fase 2

4. `SUM` vs `ABS-SUM`
5. mezclas de `SUM + L2 + RMS`

### Fase 3

6. `Soft Top-k`
7. funciones de co-ocurrencia
8. competencia intra-neuronal o softmax pooling

---

## 6. Qué Pregunta Responde Cada Familia

| Familia | Pregunta |
| :--- | :--- |
| `RMS` | ¿Importa la energia dominante o la energia normalizada? |
| `MEAN` | ¿La neurona debe medir evidencia total o consenso medio? |
| `ABS-SUM` | ¿La cancelacion algebraica de signos esta ocultando señal util? |
| `LogSumExp` | ¿Se puede recuperar el comportamiento de `MAX` de forma continua y estable? |
| `Multiplicative` | ¿Hay neuronas que deban detectar co-ocurrencia y no suma? |
| `Variance` | ¿Algunas neuronas deberian detectar contraste o desacuerdo? |

---

## 7. Recomendación Operativa

La recomendacion mas realista, viendo los findings actuales, es:

1. mantener `V1` como baseline principal
2. mantener `V14` como baseline polimorfico eficiente
3. probar a continuacion:
   - `RMS`
   - `LogSumExp`
   - `MEAN`

Estas tres familias tienen el mejor equilibrio entre:

- novedad funcional
- coste razonable
- estabilidad probable
- facilidad de interpretar los resultados

---

## 8. Cierre

La linea polimorfica parece prometedora no porque de mas accuracy inmediata, sino porque abre la posibilidad de que cada neurona aprenda no solo "que conexiones usa", sino "que tipo de computacion realiza".

La clave ahora es explorar esa familia con funciones que sean:

- matematicamente sanas
- computacionalmente baratas
- y conceptualmente distintas entre si

En ese sentido, la siguiente frontera natural no parece ser una algebra totalmente libre, sino una pequeña biblioteca de agregadores neuronales bien escogidos.
