# Diseños Propuestos: V14, V15 y V16

*Extensiones directas del hallazgo de `V13 (Polymorphic Attention Neuron)`, orientadas a preservar la riqueza funcional del "dial neuronal" sin pagar el coste prohibitivo del operador `max` exacto.*

---

## 1. Contexto

El experimento [findings_v13_polymorphic.md](C:\Users\mrcm_\Local\proj\algorithms\attention-neuron\docs\findings_v13_polymorphic.md) ha dejado una conclusion importante:

- el polimorfismo neuronal **si aporta estructura interna interesante**
- la red **si se autoorganiza** en roles distintos por capa
- pero el uso de `max` exacto **rompe la eficiencia computacional**

Por tanto, la siguiente generacion no deberia insistir en el `max` duro, sino buscar **aproximaciones suaves, vectorizables y compatibles con matmul**.

---

## 2. V14: L2 Polymorphic Attention Neuron

### Idea

Sustituir el dial `SUM/MAX` por un dial entre:

- agregacion lineal clasica: `SUM`
- agregacion energetica: `L2`

La intuicion es que la norma `L2` ya favorece valores grandes y rasgos dominantes, pero sin requerir el `max` exacto ni romper la vectorizacion.

### Formula

Para una capa:

$$
y_{sum} = X \cdot W^T
$$

$$
y_{l2} = \sqrt{(X^2) \cdot (W^2)^T + \epsilon}
$$

$$
y_{eff} = \alpha \cdot y_{sum} + (1-\alpha) \cdot y_{l2}
$$

donde `alpha` es entrenable por neurona.

### Hipotesis

`L2` puede actuar como detector de evidencia fuerte sin perder compatibilidad con hardware optimizado para multiplicacion de matrices.

### Lo que esperamos

- accuracy similar o ligeramente superior a `V13`
- tiempo por epoca mucho menor que `SUM/MAX`
- autoorganizacion interpretable de `alpha`

### Criterio de exito

- mantener o superar `86.5%` en MNIST
- reducir claramente el tiempo por epoca respecto a `V13`
- observar distribucion no trivial de `alpha` en capas ocultas

### Riesgo

La `L2` puede parecerse demasiado a una suma suavizada y no separar roles neuronales con tanta claridad como el `max`.

---

## 3. V15: Learnable-Lp Attention Neuron

### Idea

Generalizar el experimento polimorfico: en vez de mezclar dos agregadores fijos, permitir que cada capa o neurona aprenda un exponente `p` que defina su propia algebra de agregacion.

### Formula

Una formulacion estable por capa podria ser:

$$
y_{lp} = \left((|X|^p) \cdot (|W|^p)^T + \epsilon \right)^{1/p}
$$

con `p` parametrizado como:

$$
p = 1 + softplus(\rho)
$$

De ese modo:

- `p` cercano a `1` se parece a una suma
- `p` grande se acerca a un comportamiento tipo detector dominante

### Version recomendada

Primero entrenable **por capa**, no por neurona. Eso reduce ruido, complejidad y coste de depuracion.

### Hipotesis

La red puede descubrir por si sola la "algebra adecuada" para cada capa:

- capas ocultas con `p > 1`
- capa final con `p ~ 1`

### Lo que esperamos

- mejor compromiso entre expresividad y coste
- una lectura muy clara de la funcion de cada capa
- un puente limpio entre `SUM` y `MAX` sin operadores duros

### Criterio de exito

- accuracy competitiva con `V1`
- valores de `p` divergentes por capa con sentido funcional
- coste razonable frente a `V13`

### Riesgo

Si `p` se vuelve demasiado grande o demasiado inestable, la optimizacion puede hacerse ruidosa o la capa puede colapsar a un comportamiento poco util.

---

## 4. V16: Competitive-Lp Attention Neuron

### Idea

Combinar el hallazgo de `V13` con una forma ligera de competencia neuronal.

No se trata de usar `max` exacto, sino de permitir que una neurona:

- agregue con una metrica `Lp`
- y ademas compita con otras neuronas de la capa a traves de una normalizacion suave

### Mecanismo sugerido

1. Calcular la salida polimorfica `Lp`.
2. Aplicar una compuerta competitiva ligera entre neuronas de la misma capa.

Por ejemplo:

$$
g = sigmoid(\beta \cdot (y - mean(y_{layer})))
$$

$$
y_{out} = g \odot y
$$

o una normalizacion estilo softmax local muy suave.

### Hipotesis

El valor del polimorfismo puede emerger mejor si las neuronas no solo eligen como agregar, sino tambien si compiten por representar rasgos distintos.

### Lo que esperamos

- mayor especializacion
- menor redundancia interna
- potencial mejora de generalizacion

### Criterio de exito

- distribuciones neuronales mas diferenciadas
- mejora en interpretabilidad
- accuracy no inferior de forma severa al baseline residual

### Riesgo

Es el diseño mas arriesgado de los tres. Mete dos fuentes nuevas de comportamiento a la vez:

- algebra variable
- competencia entre neuronas

Puede ser muy interesante, pero tambien mas dificil de estabilizar.

---

## 5. Orden Recomendado

### 1. V14 primero

Es la extension mas limpia de `V13`.

- cambia solo el agregador costoso
- mantiene la intuicion del dial neuronal
- maximiza la probabilidad de obtener una mejora rapida

### 2. V15 despues

Es la evolucion conceptual natural.

- mas elegante matematicamente
- probablemente mas general
- mejor candidata si `V14` confirma que `L2` ya captura buena parte del efecto

### 3. V16 al final

Es la apuesta mas ambiciosa.

- mas novedad
- mas riesgo
- mas dificil de depurar

---

## 6. Recomendación Operativa para Implementación

### V14

- implementar sobre la baseline residual
- `alpha` por neurona
- comparar directamente contra `V1` y `V13`

### V15

- empezar con `p` por capa
- acotar implicitamente con `p = 1 + softplus(rho)`
- loggear el valor final de `p` por capa

### V16

- construir solo despues de tener claro el comportamiento de `V14` o `V15`
- mantener la competencia suave, no hard
- evitar introducir varias restricciones fuertes a la vez

---

## 7. Qué Pregunta Responde Cada Variante

| Variante | Pregunta principal |
| :--- | :--- |
| `V14` | ¿El valor de `V13` era el polimorfismo, o solo el uso de `max` duro? |
| `V15` | ¿Puede la red aprender su propia algebra de agregacion de forma continua y eficiente? |
| `V16` | ¿La especializacion mejora si las neuronas no solo agregan distinto, sino que tambien compiten? |

---

## 8. Cierre

`V13` ha abierto una linea muy sugerente: la neurona no solo puede aprender sus pesos efectivos, sino tambien su forma de computar evidencia.

La mejor continuacion de esa idea no parece ser insistir en el `max` exacto, sino moverse hacia una familia `Lp`:

- mas suave
- mas vectorizable
- mas interpretable
- y probablemente mas escalable

Por eso, la secuencia recomendada es:

1. `V14` para capturar el beneficio inmediato
2. `V15` para convertirlo en principio general
3. `V16` para explorar una fase mas biologica y competitiva
