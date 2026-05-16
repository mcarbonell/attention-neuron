# Los Experimentos que Haría

## La pregunta central que quiero responder

```
¿Es el peso individual w_i la unidad mínima de aprendizaje
o es un artefacto histórico de cómo Rosenblatt formuló el problema?
```

Todo lo demás deriva de ahí.

---

## Experimento 1: La Red de Pesos Aleatorios Fijos

```python
# Pesos w_i generados aleatoriamente y CONGELADOS
# Solo se entrena un parámetro por neurona: su ganancia global λ
y = σ(λ · Σ w_i_random · x_i + b)
```

**Pregunta**: ¿Cuánto del aprendizaje viene de ajustar los pesos relativos entre sí, y cuánto viene simplemente de que cada neurona aprende su nivel de activación global?

Si esto funciona razonablemente bien, es demoledor. Significaría que la mayor parte del aprendizaje no está en las sinapsis sino en la modulación neuronal.

---

## Experimento 2: Pesos como Función, no como Tabla

En lugar de w_i como lookup table de n valores libres:

```python
# Los pesos son generados por una función con k parámetros, k << n
w_i = f(i, θ)  # θ son los parámetros entrenables

# Ejemplos de f:
w_i = sin(θ_1 · i + θ_2)           # 2 parámetros
w_i = θ_1 · exp(-θ_2 · (i - θ_3)²) # 3 parámetros gaussiana
w_i = Bezier(i, P_0, P_1, P_2)      # como V50 pero generalizado
```

**Pregunta**: ¿Qué familia de funciones generadoras de pesos es universalmente útil más allá del dominio visual?

---

## Experimento 3: El Límite de Compresión Real

Barrido sistemático:

```
784 parámetros por neurona  → accuracy X
100 parámetros por neurona  → accuracy ?
10 parámetros por neurona   → accuracy ?
5 parámetros por neurona    → accuracy ?
2 parámetros por neurona    → accuracy ?
1 parámetro por neurona     → accuracy ?
```

**Pregunta**: ¿Existe un límite de compresión duro o la curva es suave hasta el final? ¿Cuánta información sobre la tarea está realmente en los pesos vs en la arquitectura?

---

## Experimento 4: Transferencia del Sesgo Inductivo

Tomar las neuronas de trazos entrenadas en MNIST y aplicarlas a:
- MNIST con ruido extremo
- MNIST rotado
- MNIST a resolución 128x128 sin reentrenar
- Dígitos de otro dataset completamente diferente

**Pregunta**: ¿La representación geométrica generaliza de formas que la representación píxel no puede? ¿Es más robusta a distributional shift?

---

## Experimento 5: El más radical

```python
# Red sin pesos en absoluto en la primera capa
# Solo distancias, relaciones, estructuras topológicas

# En lugar de:
y = σ(Σ w_i · x_i)

# Algo como:
y = σ(f_geometrica(x, θ_neurona))

# Donde f_geometrica puede ser:
# - distancia a un prototipo
# - correlación con una forma
# - resonancia con una frecuencia
```

**Pregunta**: ¿Puede una red aprender sin multiplicaciones de matrices? ¿Qué operaciones son realmente necesarias?

---

## La pregunta que más me inquieta

Y creo que es la que tú ya estás haciendo:

**Si el modelo de Rosenblatt es una elección arbitraria que se fosilizó por inercia histórica, y si con estructuras alternativas conseguimos 99% de compresión sin coste en accuracy...**

```
¿Cuánta de la inteligencia actual de los LLMs está
en los parámetros
y cuánta está en la estructura de la atención?

¿Qué pasaría con un transformer donde cada cabeza de atención
fuera una neurona paramétrica estructurada
en lugar de matrices densas libres?
```

---

## Lo que creo que estás viendo

Y dime si voy desencaminado:

El dogma actual dice que la inteligencia emerge de **muchos parámetros libres + suficientes datos + gradiente**.

Tu hipótesis es que la inteligencia emerge de **estructura correcta + pocos parámetros + gradiente**.

Y que llevamos 70 años usando parámetros libres para aproximar estructura que podríamos haber codificado directamente.

¿Es eso?