# V101 Cone Attention: Extreme Parameter Efficiency & Biological Inhibition

## Resumen del Experimento (V98 - V101)
El objetivo de estos experimentos fue demostrar que la atención espacial estructurada puede reemplazar a las conexiones densas tradicionales en tareas de visión, reduciendo drásticamente el número de parámetros sin sacrificar expresividad.

En una red neuronal estándar, cada neurona densa conectada a una imagen de 28x28 (MNIST) requiere 784 pesos libres, lo que genera problemas de optimización en espacios de alta dimensionalidad y dificulta la convergencia. 

Nuestro enfoque propone una **"Neurona de Atención Cónica 2D"**, que requiere únicamente **4 parámetros** ($C_x, C_y, Radio, Amplitud$) para definir su campo receptivo, imitando a las células ganglionares de la retina.

## Hitos de Rendimiento
- **V98 (Triángulo 1D):** 1,064 parámetros. Precisión: ~18.5%. Demostró que las derivadas fluyen por operaciones geométricas, pero evidenció que aplanar la imagen (perder la topología 2D) rompe el campo receptivo.
- **V99 (Cono 2D):** 1,320 parámetros. Precisión: ~89.36%. Al usar la distancia Euclidiana real (X, Y) y restaurar la topología 2D de la imagen, la precisión saltó más de 70 puntos.
- **V100 (Cono 2D + Capa de Salida Densa):** 3,850 parámetros. Precisión: ~91.75%. Conectar los mapas de características espaciales generados por los conos a una capa densa de 10 clases rompió la barrera del 90%.
- **V101 (Cono 2D + Inhibición + LR bajo):** 3,850 parámetros. Precisión: **94.30%**. Al permitir amplitudes negativas (inhibición) inicializando entre -1.0 y 1.0, y bajando el LR a 0.001, la convergencia fue extraordinariamente rápida (90.44% en Epoch 1).
- **V102 (Cono 2D + Salida Triangular 1D):** 1,320 parámetros. Precisión: **89.34%**. Volviendo a la extrema eficiencia, se forzó un cuello de botella geométrico. Con un presupuesto minúsculo (~25 conos por clase), la red logró auto-organizar las 256 características ocultas en un espacio topológico 1D continuo para satisfacer al clasificador triangular.

## Auto-Organización Topológica (El caso V102)
El experimento V102 reveló un comportamiento profundo: la interpretabilidad forzada.
Al usar una **Capa Triangular 1D** como clasificador final, obligamos a la red a que las características necesarias para detectar un dígito específico deban estar **físicamente juntas** en el vector oculto de 256 dimensiones. 
La red, al aprender a optimizar la *Loss* usando solo distancias, actuó como un **Autoencoder Topológico Supervisado**. Creó un espacio latente estructurado y continuo en lugar del típico caos no interpretable de una capa Densa clásica, logrando casi un 90% de precisión bajo una restricción paramétrica severa.

## Formulación Matemática del Cono 2D
La generación de los pesos dinámicos para cada uno de los 784 píxeles $(P_x, P_y)$ se realiza en tiempo real a partir de solo 4 parámetros entrenables por neurona:

1. **Distancia Euclidiana:** Se mide la separación física entre el píxel y el centro de atención de la neurona $(C_x, C_y)$.
   $$Distancia = \sqrt{(P_x - C_x)^2 + (P_y - C_y)^2}$$

2. **Normalización por Radio:** El campo atencional tiene forma de cono, alcanzando su pico máximo en el centro exacto y cayendo hasta cero en el borde delimitado por el $Radio$.
   $$Base = 1 - \frac{Distancia}{Radio}$$

3. **Corte Espacial (ReLU):** Para evitar que el cono tenga un pozo infinito negativo y actúe verdaderamente como un campo receptivo local, se truncan a cero los valores fuera del radio.
   $$Base\_Real = max\left(0, 1 - \frac{Distancia}{Radio}\right)$$

4. **Amplitud (Excitación/Inhibición):** Se multiplica la forma del cono por la fuerza de la neurona. Una `Amplitud` positiva significa que la neurona se activa al ver luz (Centro ON). Una `Amplitud` negativa significa que la neurona resta activación general al ver luz (Centro OFF, Inhibición).
   $$Peso\_Final = Base\_Real \times Amplitud$$

## Conclusión
La reducción del espacio de búsqueda a **4 dimensiones por neurona** hace que el paisaje de pérdida (*Loss Landscape*) sea liso y directo. El descenso de gradiente simplemente "mueve" el cono $(C_x, C_y)$ o "ajusta" su tamaño ($Radio$) e intensidad ($Amplitud$), logrando una convergencia acelerada y evitando el sobreajuste (*overfitting*) en el ruido intrínseco del dataset MNIST.
