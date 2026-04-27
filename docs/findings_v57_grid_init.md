# Findings V57: Grid Initialization (Structured Prior)

## Resumen del Experimento
En la versión v57, probamos la **inicialización en rejilla (Grid Initialization)**. Colocamos las 256 neuronas Matchstick en una cuadrícula perfecta de 16x16 cubriendo toda la imagen de 28x28. Cada neurona comenzó como un pequeño segmento vertical.

## Resultados
- **Best Accuracy**: **97.16%** (Epoch 9).
- **First Epoch Accuracy**: **95.84%**.
- **Comparativa**: Superior a la inicialización central (v55: 95.71% en 100 épocas) pero ligeramente inferior a la inicialización aleatoria optimizada (v51: 98.30%).

## Análisis: Orden vs. Caos
1.  **Arranque Explosivo**: La precisión de la primera época (95.84%) es una de las más altas registradas. Esto confirma que la cobertura total del lienzo elimina la "ceguera geométrica" desde el primer batch.
2.  **Estabilidad**: El entrenamiento fue muy estable, manteniéndose en el rango del 96-97% casi todo el tiempo. La rejilla actúa como un "prior" de muestreo uniforme que garantiza que ninguna característica del dígito sea ignorada.
3.  **El Techo del Orden**: El hecho de que no supere el 98.3% sugiere que la rigidez de la rejilla inicial podría ser un obstáculo. En la inicialización aleatoria, algunas neuronas empiezan por azar en ángulos o posiciones muy ventajosas que la rejilla tarda en alcanzar. El "caos" de la v51 proporciona una diversidad de búsqueda más rica desde el inicio.

## Conclusión
La inicialización en rejilla es la forma más **fiable y rápida** de obtener un modelo decente (>97%) sin depender de la suerte del muestreo aleatorio. Sin embargo, para alcanzar precisiones de estado del arte en este modelo, la diversidad estocástica (azar) parece ser superior a largo plazo.

## Próximos Pasos
- ¿Y si combinamos ambas? Una rejilla base con un pequeño desplazamiento aleatorio (Jittered Grid).
- Evaluar si la rejilla es más robusta ante rotaciones.
