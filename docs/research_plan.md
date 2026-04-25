# Plan de Investigacion: Attention Neuron

*Hoja de ruta de corto plazo para convertir la exploracion inicial del repositorio en un programa de investigacion disciplinado, comparable y acumulativo.*

---

## 1. Objetivo del Plan

El repositorio ya ha validado una intuicion importante: una red puede conservar una parte muy alta de su capacidad predictiva si, en lugar de entrenar todos los pesos espaciales, aprende una modulacion topologica de muy bajo rango asociada a las neuronas.

El siguiente paso ya no es "probar ideas sueltas", sino responder de forma ordenada a cuatro preguntas:

1. **Que componentes de la Attention Neuron son realmente esenciales?**
2. **Que parte de la mejora viene del bajo rango en general y que parte viene de la formulacion neuron-centric?**
3. **Hasta donde escala la arquitectura fuera de MNIST?**
4. **Puede DGE optimizar esta arquitectura de forma competitiva en coste real, no solo conceptual?**

Este documento define la matriz de experimentos recomendada para las proximas iteraciones.

---

## 2. Estado Actual del Repositorio

### 2.1 Hallazgos ya explorados

Segun [dge_findings_attention_neuron.md](C:\Users\mrcm_\Local\proj\algorithms\attention-neuron\docs\dge_findings_attention_neuron.md), el repositorio ya ha cubierto estas etapas:

- `v1`: normalizacion estricta del fan-in + delta escalar + bias angular. Fallo por cancelacion matematica.
- `v2`: neurona como variable con 2 parametros entrenables.
- `v3`: mascara estocastica aplicada a subconjuntos de cables.
- `v4`: optimizacion greedy sin backpropagation.
- `v5-v5b`: escalado bruto de anchura, con resultados pobres.
- `v6-v6b`: formulacion dual-neuron con factorizacion rank-1 y rank-2.
- `v7`: rank-2 + phase bias.
- `v8`: traslado a CIFAR-10 con MLP.
- `v9`: traslado a CNN factorizada.

### 2.2 Punto fuerte actual

La evidencia mas fuerte del repositorio, a dia de hoy, esta en la **arquitectura**:

- La factorizacion rank-2 parece capturar una gran parte del rendimiento con una fraccion minima de parametros entrenables.
- El phase bias parece preservar estabilidad sin destruir precision.
- La idea parece trasladarse desde MLPs a CNNs.

### 2.3 Punto todavia incierto

La evidencia sobre **DGE + Attention Neuron** todavia necesita consolidacion experimental. Existe una narrativa muy prometedora en [findings_v10.md](C:\Users\mrcm_\Local\proj\algorithms\attention-neuron\docs\findings_v10.md), pero antes de elevar esa linea a conclusion estable conviene reconciliarla con los resultados crudos guardados en `results/raw/`.

Por tanto, este plan separa claramente:

- experimentos de **confirmacion de arquitectura**
- experimentos de **comparacion justa**
- experimentos de **escalado**
- experimentos de **validacion de DGE**

---

## 3. Principios de Priorizacion

Durante las proximas iteraciones, la prioridad no debe ser "abrir mas ideas", sino reducir incertidumbre cientifica.

### 3.1 Orden de ataque recomendado

1. **Confirmar y ablar la arquitectura**
2. **Comparar contra baselines cercanos**
3. **Revisar robustez y sensibilidad**
4. **Escalar a datasets y arquitecturas algo mas duras**
5. **Volver a DGE con una arquitectura ya fijada**

### 3.2 Regla estrategica

La tesis principal a demostrar primero no debe ser:

> "hemos sustituido backpropagation"

Sino esta otra, mas alcanzable y mas fuerte en el corto plazo:

> "podemos entrenar redes utiles optimizando solo un espacio neuron-centric de muy baja dimension, manteniendo gran parte del rendimiento con una fraccion minima de parametros entrenables"

Si esa tesis queda robusta, entonces DGE pasa de ser una promesa a una extension natural.

---

## 4. Matriz de Experimentos

### 4.1 Bloque A: Confirmacion de arquitectura

Objetivo: aislar que piezas de la Attention Neuron aportan realmente rendimiento y estabilidad.

| ID | Prioridad | Experimento | Comparacion | Dataset | Decision que desbloquea |
| :--- | :--- | :--- | :--- | :--- | :--- |
| A1 | P0 | Repetir baseline `rank-2 + additive + multiplicative + phase bias` | `v6b` vs `v7` | MNIST | Fijar baseline dorado |
| A2 | P0 | `phase bias` vs bias lineal | misma arquitectura y mismo rank | MNIST | Saber si `sin(theta)` aporta evidencia real |
| A3 | P0 | solo termino aditivo vs completo | mismo presupuesto aproximado | MNIST | Medir cuanto aporta el gating multiplicativo |
| A4 | P0 | solo termino multiplicativo vs completo | mismo presupuesto aproximado | MNIST | Medir si el termino aditivo es esencial |
| A5 | P0 | barrido de rango `1/2/4/8` | mismo setup | MNIST | Encontrar punto de saturacion expresiva |
| A6 | P0 | repetir mejores variantes con 5 seeds | mejor `rank-2` y mejor `rank-4` | MNIST | Pasar de anecdotas a señal robusta |

### 4.2 Bloque B: Comparacion justa con baselines cercanos

Objetivo: evitar que la mejora se confunda con "cualquier low-rank pequeno funciona".

| ID | Prioridad | Experimento | Comparacion | Dataset | Decision que desbloquea |
| :--- | :--- | :--- | :--- | :--- | :--- |
| B1 | P1 | MLP denso estandar | attention neuron | MNIST | Cuanto rendimiento se cede por compresion |
| B2 | P1 | matriz congelada + solo bias entrenable | attention neuron | MNIST | Cuanto aporta modular cables y no solo offsets |
| B3 | P1 | matriz congelada + low-rank aditivo tipo LoRA | attention neuron | MNIST | Separar "bajo rango general" de "neuron-centric" |
| B4 | P1 | mejor Adam/backprop vs mejor DGE | misma arquitectura | MNIST | Saber si DGE es competitivo en coste real |

### 4.3 Bloque C: Sensibilidad y dinamica de entrenamiento

Objetivo: entender que hiperparametros gobiernan la estabilidad del metodo.

| ID | Prioridad | Experimento | Comparacion | Dataset | Decision que desbloquea |
| :--- | :--- | :--- | :--- | :--- | :--- |
| C1 | P1 | barrido de `mask_prob` | `1.0 / 0.8 / 0.5 / 0.2` | MNIST | Ver si la mascara ayuda o solo introduce ruido |
| C2 | P1 | mascara fija vs schedule a `1.0` | mismo modelo | MNIST | Validar la hipotesis de "clean polish" |
| C3 | P1 | sensibilidad a batch size | `128 / 512 / 2048 / 8192` | MNIST | Medir cuanta señal estadistica necesita el metodo |
| C4 | P1 | sensibilidad a anchura | hidden `256 / 512 / 1024` | MNIST | Ver si el sustrato aleatorio exige sobreanchura |

### 4.4 Bloque D: Escalado y generalizacion

Objetivo: probar que la arquitectura no es un truco restringido a MNIST.

| ID | Prioridad | Experimento | Comparacion | Dataset | Decision que desbloquea |
| :--- | :--- | :--- | :--- | :--- | :--- |
| D1 | P2 | mejor variante en Fashion-MNIST | baselines de B | Fashion-MNIST | Ver si la señal sale de digitos limpios |
| D2 | P2 | revalidacion de MLP attention en CIFAR-10 | MLP denso pequeno | CIFAR-10 | Confirmar que el resultado de `v8` aguanta limpio |
| D3 | P2 | revalidacion de CNN attention | CNN densa pequena | CIFAR-10 | Confirmar que la idea sirve en dominio espacial |

### 4.5 Bloque E: DGE sobre arquitectura fijada

Objetivo: volver a la hipotesis de sinergia solo cuando la arquitectura ya este estabilizada.

| ID | Prioridad | Experimento | Comparacion | Dataset | Decision que desbloquea |
| :--- | :--- | :--- | :--- | :--- | :--- |
| E1 | P2 | DGE limpio sobre la mejor attention neuron | Adam en misma arquitectura | MNIST | Saber si DGE compensa en eficiencia |
| E2 | P2 | `rank-2` vs `rank-4` bajo DGE | mismo schedule | MNIST | Medir si comprimir menos ayuda a DGE |
| E3 | P2 | batch fijo vs batch incremental con patience | mismo modelo | MNIST | Ver si la heuristica adaptativa es real |
| E4 | P2 | SPSA/DGE simple vs ruido estructurado | misma arquitectura | MNIST | Preparar una posible `v11` o `v12` seria |

---

## 5. Matriz Minima para las Proximas Horas

Si el objetivo es avanzar rapido sin dispersarse, esta es la **submatriz minima** recomendada:

| ID | Experimento | Seeds | Prioridad inmediata |
| :--- | :--- | :--- | :--- |
| M1 | `rank-2 full + phase` | 5 | Muy alta |
| M2 | `rank-2 full - phase` | 5 | Muy alta |
| M3 | `rank-2 multiplicative only` | 5 | Muy alta |
| M4 | `rank-2 additive only` | 5 | Muy alta |
| M5 | `rank-1 full + phase` | 5 | Alta |
| M6 | `rank-4 full + phase` | 5 | Alta |
| M7 | frozen random + low-rank aditivo equivalente | 5 | Alta |
| M8 | mejor variante resultante en Fashion-MNIST o CIFAR-10 | 3-5 | Alta |

Con esta submatriz ya se pueden responder preguntas cientificas fuertes:

- si el `phase bias` importa de verdad
- si la mejora depende del termino multiplicativo, del aditivo o de ambos
- si `rank-2` es un punto especial o solo un escalon intermedio
- si el metodo supera a un baseline de bajo rango mas convencional
- si la señal sale de MNIST

---

## 6. Secuencia Recomendada de Ejecucion

### Fase 1: Consolidar arquitectura

Ejecutar primero:

1. `A1`
2. `A2`
3. `A3`
4. `A4`
5. `A5`
6. `A6`

Resultado esperado:

- una variante baseline claramente definida
- una conclusion clara sobre el valor del phase bias
- una conclusion clara sobre aditivo vs multiplicativo
- un rango recomendado por defecto

### Fase 2: Comparacion justa

Ejecutar despues:

1. `B2`
2. `B3`
3. `B1`

Resultado esperado:

- demostrar que la mejora no es reducible a "solo bias"
- demostrar que la mejora no es reducible a "cualquier low-rank"
- cuantificar cuanto cuesta la compresion frente a una densa normal

### Fase 3: Robustez y sensibilidad

Ejecutar despues:

1. `C1`
2. `C2`
3. `C3`
4. `C4`

Resultado esperado:

- una receta de entrenamiento mas estable
- un rango razonable de batch size y mascara
- una intuicion mas clara sobre la anchura necesaria

### Fase 4: Escalado

Ejecutar despues:

1. `D1`
2. `D2`
3. `D3`

Resultado esperado:

- demostrar generalizacion fuera de MNIST
- confirmar si el metodo aguanta en imagenes mas complejas

### Fase 5: Vuelta a DGE

Solo cuando todo lo anterior este firme:

1. `E1`
2. `E2`
3. `E3`
4. `E4`

Resultado esperado:

- decidir si DGE es una linea principal o una linea secundaria
- evitar enterrar una buena arquitectura dentro de un optimizador todavia inmaduro

---

## 7. Criterios de Decision

Para evitar iterar sin cierre, cada bloque debe terminar con una decision explicita.

### 7.1 Criterios para arquitectura

- **Mantener phase bias** si mejora estabilidad o no pierde precision de forma significativa frente a bias lineal.
- **Mantener termino multiplicativo** si el gap frente a `additive only` es consistente.
- **Mantener termino aditivo** si evita techos de precision claramente inferiores.
- **Fijar rank por defecto** donde la mejora marginal por parametro adicional deje de compensar.

### 7.2 Criterios para comparacion

- Si el metodo no supera a un low-rank aditivo equivalente, la tesis central debe reformularse.
- Si el metodo mantiene mejor precision por parametro entrenable, la tesis neuron-centric gana fuerza.

### 7.3 Criterios para DGE

DGE solo deberia escalar a linea principal si demuestra al menos una de estas tres ventajas:

- mejor precision por numero de evaluaciones
- mejor uso de memoria total
- mejor comportamiento en variantes donde backprop sea inestable

Si no cumple ninguna, debe permanecer como exploracion paralela y no como narrativa central.

---

## 8. Riesgos a Vigilar

### Riesgo 1: Confundir novedad con combinacion afortunada

El metodo puede ser muy prometedor sin necesidad de reclamar una ruptura total con LoRA o low-rank clasico. Conviene medir antes de narrar.

### Riesgo 2: Sobreajustar la historia a MNIST

MNIST sirve para descubrir mecanica, no para cerrar tesis ambiciosas.

### Riesgo 3: Dar por validado DGE demasiado pronto

La sinergia puede ser real, pero la carga de prueba es mucho mayor en tiempo real, overhead y reproducibilidad.

### Riesgo 4: Mezclar resultados narrativos y crudos

Antes de reclamar hitos estables, toda afirmacion importante debe apoyarse en JSONs, semillas y configuraciones recuperables.

---

## 9. Tarea Administrativa Recomendada

Antes o durante la siguiente ronda de experimentos, conviene hacer una mini-auditoria:

1. Enumerar todos los resultados crudos disponibles en `results/raw/`.
2. Mapear cada JSON a su script exacto y a su version documental.
3. Marcar que resultados estan:
   - confirmados por JSON
   - presentes solo en docs narrativos
   - pendientes de repeticion

Esto reducira mucho el riesgo de construir la hoja de ruta sobre resultados mezclados o dificilmente reproducibles.

---

## 10. Conclusión Operativa

El repositorio ya tiene una intuicion arquitectonica fuerte. La prioridad ahora no es inventar mas variantes a ciegas, sino convertir esa intuicion en una secuencia de decisiones verificables.

La recomendacion central de este plan es sencilla:

- **primero consolidar la arquitectura**
- **despues compararla justamente**
- **luego escalarla**
- **y solo entonces decidir si DGE merece ser la narrativa principal**

Si esta secuencia sale bien, la Attention Neuron puede pasar muy rapido de "idea prometedora de unas horas" a "programa de investigacion con tesis clara".
