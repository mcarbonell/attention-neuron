# Findings V55: Symmetry Break & Geometric Blindness

## Resumen del Experimento
En la versión v55, realizamos una prueba de "Symmetry Breaking" inicializando las 256 neuronas como un segmento vertical idéntico en el centro de la imagen (14, 14). El objetivo era ver si el gradiente de la capa clasificadora era suficiente para dispersar las neuronas y cubrir el espacio de MNIST.

## Resultados (100 Épocas)
- **Best Accuracy**: **95.71%** (Epoch 97).
- **Comportamiento**: La red **logró romper la simetría** pero de forma extremadamente lenta.
- **Visualización**: La galería muestra ahora una diversidad real de ángulos y posiciones, aunque con una organización menos "limpia" que partiendo de azar.

## Análisis: El Triunfo de la Persistencia
Este experimento extendido nos da una visión más profunda de la dinámica de las Matchstick Neurons:

1.  **El Ciclo de "Búsqueda"**: 
    - En las primeras 20 épocas, la red se estanca en el ~85-90%. Durante este tiempo, los gradientes son débiles, pero constantes.
    - Los **sigmas crecen** primero. Al ensancharse, la neurona empieza a "sentir" gradientes de píxeles más lejanos.
    - Una vez que la neurona "capta" una estructura importante fuera del centro, las coordenadas $(x, y)$ empiezan a migrar.
2.  **Eficiencia vs. Posibilidad**: 
    - Partiendo de azar (v51), llegamos al 98% en **10 épocas**.
    - Partiendo del centro (v55), necesitamos **100 épocas** para llegar al 95%.
    - Esto demuestra que la inicialización aleatoria en este modelo no es solo para evitar redundancia (como en MLPs), sino que actúa como una **infraestructura de transporte de gradiente**. Sin ella, la red pasa el 90% del tiempo simplemente intentando "encontrar" dónde están los datos.
3.  **Techo de Precisión**: El hecho de que no llegue al 98.3% sugiere que algunas neuronas se quedan atrapadas en mínimos locales o colapsan entre sí durante la migración masiva desde el centro hacia la periferia.

## Conclusión Actualizada
La simetría **se rompe**, pero el coste computacional es 10 veces mayor y el resultado final es inferior. La "Ceguera Geométrica" es real, pero no absoluta; es una barrera de gradiente que se puede saltar con fuerza bruta (épocas) o con inteligencia (inicialización aleatoria o schedulers de sigma).

## Lecciones para el Futuro
- La **cobertura inicial** es obligatoria.
- Si quisiéramos que se movieran desde el centro, necesitaríamos un "gradiente de largo alcance" o una fase inicial donde los sigmas sean muy grandes (visión borrosa global) y se vayan estrechando (foco local).
