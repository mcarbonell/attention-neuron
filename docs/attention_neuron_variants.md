# Attention Neuron: Variantes de Arquitectura a Explorar

*Documento de brainstorming operativo. No es una hoja de resultados, sino un catalogo de variantes plausibles de la Attention Neuron para futuras iteraciones en `scratch/`.*

---

## 1. Objetivo

La version actual de la Attention Neuron ya ha encontrado una formulacion con muy buena señal inicial:

- sustrato aleatorio congelado
- modulacion multiplicativa de bajo rango
- correccion aditiva de bajo rango
- bias de fase acotado

Este documento propone variantes para explorar el espacio de diseño alrededor de esa formulacion base. La idea no es ejecutarlas todas de golpe, sino tener un mapa claro de extensiones posibles y de la intuicion detras de cada una.

---

## 2. Variante Base de Referencia

Tomamos como referencia conceptual esta forma:

$$
W_{eff} = W_{init} \odot M + A
$$

donde:

- $W_{init}$ es el sustrato aleatorio congelado
- $M$ es una modulacion multiplicativa de bajo rango
- $A$ es una correccion aditiva de bajo rango
- el bias se expresa como $sin(\theta)$

Las variantes de abajo modifican uno o varios de esos componentes.

---

## 3. Catalogo de Variantes

### V1. Residual Attention Neuron

**Formula**

$$
W_{eff} = W_{init} + W_{init} \odot M + A
$$

**Idea**

En vez de pensar la modulacion como reemplazo completo del sustrato, se trata como una correccion residual sobre la conectividad base.

**Hipotesis**

Puede estabilizar el entrenamiento y facilitar comparaciones con LoRA, porque el modelo parte de una identidad mas explicita respecto al sustrato aleatorio.

**Lo que esperaria**

- mejor estabilidad al inicio
- menor sensibilidad a inicializacion
- curva de aprendizaje mas suave

**Riesgo**

Puede hacer que la red dependa demasiado del sustrato base y reduzca la fuerza del gating estructural.

---

### V2. Log-Gated Attention Neuron

**Formula**

$$
M = exp(S)
$$

o bien

$$
M = 1 + \alpha \cdot tanh(S)
$$

**Idea**

La parte multiplicativa no se aprende como factor libre, sino como una parametrizacion geometrica mas estable.

**Hipotesis**

La mayor parte del valor de la arquitectura puede venir del gating multiplicativo, pero ese gating quizas necesita una geometria mejor condicionada.

**Lo que esperaria**

- menos explosiones o colapsos de escala
- optimizacion mas estable
- posibilidad de usar learning rates mas agresivos

**Riesgo**

Si se restringe demasiado el rango efectivo, la modulacion puede perder expresividad.

---

### V3. Sparse Attention Neuron

**Idea**

Introducir regularizacion para empujar la modulacion a seleccionar pocos patrones utiles en lugar de redistribuirlo todo suavemente.

**Mecanismos posibles**

- L1 sobre la parte aditiva
- penalizacion sobre desviacion de la parte multiplicativa respecto a `1`
- top-k suave o thresholding differentiable

**Hipotesis**

Si la neurona realmente "atiende", deberia tender a una modulacion selectiva y no puramente difusa.

**Lo que esperaria**

- mas interpretabilidad
- mejor compresion efectiva
- potencialmente mejor generalizacion

**Riesgo**

Regularizar demasiado pronto puede matar una arquitectura que todavia necesita flexibilidad.

---

### V4. Pure Multiplicative Attention Neuron

**Formula**

$$
W_{eff} = W_{init} \odot M
$$

**Idea**

Eliminar la correccion aditiva y comprobar hasta donde llega la red solo con gating estructural.

**Hipotesis**

La parte multiplicativa podria contener la intuicion central del metodo, y la parte aditiva podria ser mas correctora que esencial.

**Lo que esperaria**

- menos parametros
- mayor limpieza conceptual
- rendimiento decente pero posiblemente techo inferior

**Riesgo**

El sistema puede quedarse demasiado pegado al sustrato y no conseguir escapar de malas orientaciones iniciales.

---

### V5. Pure Additive Attention Neuron

**Formula**

$$
W_{eff} = W_{init} + A
$$

**Idea**

Usar esta variante como baseline fuerte y cercano a LoRA.

**Hipotesis**

Si esta variante se acerca mucho al modelo completo, entonces la "novedad funcional" estaria menos en el gating y mas en el simple low-rank residual.

**Lo que esperaria**

- baseline imprescindible para comparar
- rendimiento razonable
- posible peor eficiencia por parametro que la version completa

**Riesgo**

Que el modelo completo gane solo marginalmente y obligue a reformular la tesis central.

---

### V6. Dual-Speed Attention Neuron

**Idea**

Entrenar la parte multiplicativa y la aditiva con dinamicas distintas.

**Ejemplos**

- learning rate menor para la parte multiplicativa
- congelar la parte aditiva durante las primeras epocas
- abrir la parte aditiva despues de que el gating ya este colocado

**Hipotesis**

La parte multiplicativa ordena la topologia global, mientras que la aditiva refina detalles. Si ambas aprenden al mismo ritmo, pueden interferirse.

**Lo que esperaria**

- mejor estabilidad
- menos dependencia del azar inicial
- mejor rendimiento final

**Riesgo**

Complica el entrenamiento y añade mas hiperparametros.

---

### V7. Layerwise Rank Attention Neuron

**Idea**

No usar el mismo rango en todas las capas.

**Ejemplos**

- capas iniciales con `rank=4`
- capas profundas con `rank=2`
- clasificador final con `rank=1`

**Hipotesis**

Las capas no necesitan la misma capacidad de reconfiguracion. El rango optimo podria depender del papel de cada capa.

**Lo que esperaria**

- mejor precision con presupuesto parecido
- mejor asignacion de capacidad

**Riesgo**

Es facil sobrecomplicar antes de entender bien el baseline uniforme.

---

### V8. Group Attention Neuron

**Idea**

La unidad de modulacion no es la neurona individual, sino grupos pequenos de neuronas que comparten parte de la "personalidad".

**Ejemplos**

- grupos de 4, 8 o 16 neuronas
- parte multiplicativa compartida, parte aditiva individual

**Hipotesis**

Puede ser una buena transicion entre la modulacion por neurona y la modulacion por bloque, reduciendo aun mas coste y mejorando estabilidad.

**Lo que esperaria**

- menos parametros
- regularizacion estructural fuerte
- posible beneficio al escalar a redes grandes

**Riesgo**

Si se comparte demasiado, se pierde la asimetria fina que hace interesante al metodo.

---

### V9. Evolving Substrate Attention Neuron

**Idea**

Separar claramente una segunda familia: el sustrato deja de ser completamente fijo y se actualiza lentamente.

**Posibles reglas**

- `W_init` se actualiza con EMA muy lenta
- solo una fraccion del sustrato puede moverse
- solo se actualiza si una senal global lo justifica

**Hipotesis**

Quizas el mejor escenario no sea un sustrato totalmente fijo, sino casi fijo, con plasticidad lenta y controlada.

**Lo que esperaria**

- mas capacidad expresiva
- menos dependencia del azar inicial
- posible puente con ideas mas biologicas

**Riesgo**

Pierde parte de la elegancia original y complica mucho la narrativa hardware.

---

### V10. Phase Everywhere Attention Neuron

**Idea**

Extender la idea angular mas alla del bias.

**Ejemplos**

- parametros multiplicativos definidos desde fases
- parte aditiva acotada por `sin` o `tanh`
- fases separadas para escuchar y emitir

**Hipotesis**

Si la tesis fuerte del metodo es estabilidad fisica y cuantizacion amigable, puede tener sentido que no solo el bias viva en una geometria acotada.

**Lo que esperaria**

- comportamiento muy estable
- representacion mas interpretable geometricamente
- fuerte coherencia conceptual

**Riesgo**

Demasiadas restricciones a la vez pueden matar capacidad.

---

### V11. Competitive Attention Neuron

**Idea**

Introducir competencia explicita entre entradas o entre neuronas, en vez de dejar que toda la modulacion sea cooperativa.

**Ejemplos**

- softmax parcial sobre ciertas componentes de modulacion
- normalizacion competitiva entre neuronas de una misma capa
- inhibicion lateral suave

**Hipotesis**

La arquitectura podria mejorar si fuerza a decidir "quien atiende a que", en vez de permitir que todo se amplifique a la vez.

**Lo que esperaria**

- mas especializacion
- menos redundancia interna
- posible mejora en interpretabilidad

**Riesgo**

Puede reintroducir problemas de cancelacion o apagar demasiado la señal.

---

### V12. Hybrid Attention Neuron for CNNs

**Idea**

En convolucion, modular no solo canales de entrada y salida, sino tambien una pequena componente espacial.

**Ejemplo**

$$
W_{eff}(o, i, x, y) = W_{init}(o, i, x, y) \cdot M_{chan}(o, i) \cdot M_{spatial}(x, y) + A_{chan}(o, i)
$$

**Hipotesis**

Quizas en CNNs el cuello de capacidad no este solo en canales, sino tambien en permitir una ligera reconfiguracion espacial sin volver a entrenar el kernel completo.

**Lo que esperaria**

- mejora clara en CIFAR-10 y datasets visuales
- coste moderado si el termino espacial es muy pequeno

**Riesgo**

Se acerca demasiado a entrenar kernels normales y diluye la pureza del metodo.

---

## 4. Variantes que Priorizaría Primero

Si hubiera que elegir pocas variantes con buena relacion valor/esfuerzo, priorizaria estas:

### Prioridad Alta

1. **Residual Attention Neuron**
   Porque es una extension natural, limpia y muy comparable con baselines low-rank.

2. **Log-Gated Attention Neuron**
   Porque puede resolver problemas de estabilidad sin cambiar la tesis central.

3. **Pure Additive Attention Neuron**
   Porque es el baseline conceptual mas importante para no enganaros con la narrativa.

4. **Pure Multiplicative Attention Neuron**
   Porque ayuda a descubrir si el corazon real de la idea es el gating estructural.

### Prioridad Media

5. **Dual-Speed Attention Neuron**
   Porque puede desbloquear bastante rendimiento si la parte multiplicativa y aditiva tienen papeles distintos.

6. **Sparse Attention Neuron**
   Porque puede convertir la idea en algo aun mas interpretable y elegante.

7. **Hybrid Attention Neuron for CNNs**
   Porque la via visual probablemente os dara una de las mejores validaciones tempranas.

---

## 5. Variantes que Trataría con Cuidado

Estas me parecen prometedoras, pero las dejaria para mas adelante:

- **Evolving Substrate Attention Neuron**
  Porque abre una segunda tesis distinta.

- **Phase Everywhere Attention Neuron**
  Porque puede ser muy bonita conceptualmente, pero corre el riesgo de sobre-restringir demasiado pronto.

- **Competitive Attention Neuron**
  Porque puede introducir interacciones muy dificiles de depurar sin antes tener baselines mas estables.

- **Group Attention Neuron**
  Porque tiene sentido al escalar, pero puede ocultar lo que ocurre realmente a nivel neurona si se introduce demasiado pronto.

---

## 6. Recomendación Operativa

Si el objetivo es mantener la velocidad de descubrimiento sin dispersarse, la secuencia recomendada seria:

1. convertir la arquitectura actual en baseline robusto
2. ejecutar `Pure Additive` y `Pure Multiplicative`
3. probar `Residual`
4. probar `Log-Gated`
5. solo entonces abrir variantes mas exoticas

En otras palabras:

- primero descubrir **que parte del metodo importa**
- despues descubrir **como estabilizarla**
- y solo despues explorar variantes mas creativas

---

## 7. Cierre

La Attention Neuron ya tiene una forma bastante propia. Lo importante ahora no es producir diez mutaciones sin orden, sino preservar la intuicion central mientras se exploran cambios que puedan:

- aclarar la fuente real de la mejora
- mejorar la estabilidad
- facilitar la escalabilidad
- reforzar la identidad del metodo

Este documento sirve como mapa de ese espacio de diseño.

---

## 8. Progreso y Resultados Actuales

**Estado de las Variantes de Prioridad Alta y Media (Evaluadas en MNIST/CIFAR-10 con Adam):**

| Variante | Estado | Resultado / Conclusión |
| :--- | :--- | :--- |
| **V4 (Pure Multiplicative)** | ✅ Completado | **80.74%**. Demuestra que el gating multiplicativo es el motor principal. |
| **V5 (Pure Additive)** | ✅ Completado | **39.17%**. La corrección aditiva sobre ruido (estilo LoRA) es insuficiente por sí sola. |
| **V1 (Residual)** | ✅ Completado | **87.61%**. Elegido como **Baseline Robusto**. Curva estable, gran expresividad. |
| **V2 (Log-Gated - Exp)** | ✅ Completado | **87.07%**. Geometría estricta no mejora V1 y penaliza tiempo computacional. |
| **V2b (Bounded - Tanh)** | ✅ Completado | **83.48%**. Acotar la modulación reduce fuertemente la capacidad expresiva. |
| **V6 (Dual-Speed)** | ✅ Completado | **86.75%**. Separar LR retrasa la convergencia; Adam prefiere optimizar ambas partes juntas. |
| **V3 (Sparse - L1)** | ✅ Completado | **85.87%**. Fuerte interpretabilidad confirmada: la red usa pocas conexiones del sustrato. |
| **V12 (Hybrid CNN - MNIST)** | ✅ Completado | **83.42%** con solo **6,648** parámetros. El paradigma escala a visión espacial 2D. |
| **V12b (Hybrid CNN - CIFAR10)** | ✅ Completado | **40.06%** en 10 épocas (`rank=16`, 76k param). Escala a color; el `rank` es la perilla de capacidad. |
| **V13 (Polymorphic SUM/MAX)** | ✅ Completado | **86.51%**. La red aprende su álgebra: la oculta mezcla SUM/MAX, la final elige 100% SUM. Lento (~38s). |
| **V14 (Polymorphic SUM/L2)** | ✅ Completado | **86.46%**. Reemplazar MAX por L2 vectorizada recupera la velocidad (~13s) manteniendo la auto-organización. |

*Nota: Habiendo estabilizado la idea central con la variante Residual (V1) y validado su aplicabilidad en visión con V12, el siguiente paso es explorar el aumento de capacidad por época o probar las variantes "exóticas" (Phase Everywhere, Evolving Substrate, etc.).*
