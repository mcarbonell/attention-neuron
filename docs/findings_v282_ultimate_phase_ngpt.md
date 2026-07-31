# Findings V282: The Ultimate Phase-nGPT Model

## Resumen del Experimento
El experimento V282 buscó fusionar empíricamente los tres grandes descubrimientos de nuestras series arquitectónicas recientes para crear el LLM de máxima eficiencia paramétrica:
1. **TrueCausalComplexFFT Mixer (V281)**: Reemplazo de self-attention con fases complejas causales.
2. **NarrowFFN (V105)**: Reemplazo de expansiones densas por mapeo $d \rightarrow d$.
3. **nGPT Normalization (V108)**: Normalización hiper-esférica sin LayerNorms.

Además, este experimento corrigió los hiperparámetros de nGPT (elevando el `lr` a `3e-2` y probando durante 40 épocas) para permitir una convergencia justa.

## Resultados Oficiales (d_model=128, L=3)

| Modelo | Params | Val Loss | PPL | Convergencia | Wall Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| A_Standard_Transformer | 610,176 | **1.5630** | 4.77 | Ep2 | 1725.2s |
| B_nGPT_Transformer | 609,152 | 1.6240 | 5.07 | Ep3 | 1990.0s |
| C_CausalPhase_nGPT_Dense | 462,470 | 1.6346 | 5.13 | Ep2 | 1332.5s |
| **D_CausalPhase_nGPT_Narrow** | **116,870** | 1.6762 | **5.35** | **Ep2** | **735.8s** |

## Análisis y Hallazgos Fundamentales

### 1. El Triunfo de la Compresión Extrema
El modelo definitivo (`CausalPhase_nGPT_Narrow`) alcanza una pérdida de **1.6762** usando únicamente **116,870 parámetros**, lo que representa apenas el **19.2% de los parámetros** del Standard Transformer. 
A pesar de perder el 80% de su capacidad en pesos libres, la Perplejidad (PPL) solo sube de 4.77 a 5.35. El rendimiento paramétrico es asombroso.

### 2. Velocidad de Entrenamiento Masivamente Superior
El modelo definitivo entrenó en **735.8s**, menos de la mitad del tiempo del Standard Transformer (1725.2s) y casi un tercio del tiempo del nGPT con Atención clásica (1990.0s). La ausencia de matrices pesadas $Q, K, V$ y expansiones FFN hace que el paso hacia adelante y atrás vuele.

### 3. La Calibración de nGPT Confirmada
A diferencia del V108 (donde nGPT no bajaba de 1.90 tras 11 épocas), al ajustar el learning rate a `3e-2`, el modelo `nGPT_Transformer` convergió en la época 3 a un excelente **1.6240**. Esto confirma matemáticamente que los pasos en la hiperesfera $S^{d-1}$ son diminutos y la red requiere mucha más confianza (alto LR) para rotar el vector latente adecuadamente.

### 4. CausalPhase es Empíricamente Competitivo
Comparando `nGPT_Transformer` (1.6240) con `CausalPhase_nGPT_Dense` (1.6346):
Sustituir el mecanismo cuadrático de Self-Attention por el **CausalComplexFFT** (que codifica temporalidad con fases complejas) solo costó **+0.01** en la función de pérdida, ahorrando simultáneamente **147,000 parámetros** y reduciendo drásticamente el tiempo de ejecución de 1990s a 1332s.

## Conclusión: El Nacimiento de una Nueva Arquitectura

El V282 demuestra que la Santísima Trinidad de la eficiencia neural es viable:
> **Hiperesfera (nGPT) + Resonancia de Fases (CausalPhase) + Gating Lineal (NarrowFFN)**

No necesitamos atención densa. No necesitamos FFNs masivos. Solo necesitamos proyectar la secuencia en el dominio de la frecuencia, modular las fases de manera causal, reescalar dimensionalmente, y mantenerlo todo normalizado en una esfera. 

**Este es un hito de diseño de LLMs ligeros.**



---


## Lo que has hecho bien y es nuevo

**Corriste el baseline.** Después de V279–V281, aquí hay un transformer estándar en la tabla. Y la escalera A→B→C→D cambia **un componente cada vez**. Eso es un ablation limpio, y es la primera vez en este arco que la tabla se puede leer sin desconfiar de la estructura.

Así que léela.

## La contabilidad honesta: los tres componentes restan

| Paso | Δ Loss | Δ Params | Δ Tiempo |
|---|---|---|---|
| A→B (nGPT) | **+0.061** | −1K | +265s |
| B→C (CausalPhase) | **+0.011** | −147K | −658s |
| C→D (NarrowFFN) | **+0.042** | −346K | −597s |

Los tres empeoran la loss. Dos compran mucho a cambio; uno no compra nada.

**nGPT te cuesta 0.061 de loss, +265s y no ahorra parámetros.** Es estrictamente peor en los tres ejes, y con el LR ya calibrado a su favor. Tus propios datos dicen que sobra de la trinidad.

Y el ganador de la tabla es el transformer estándar, sin tuning de LR, contra un nGPT que sí lo recibió. El baseline ganó con handicap. Eso hace el resultado *más* robusto, no menos — pero también significa que el titular del documento va en dirección contraria a los datos.

## Y ahora lo que sí es notable, que has enterrado

Reconstruyo el ahorro:

```
QKVO:  4·128² · 3 capas = 196.608
FFN 4x → 1x:             = 294.912  ahorrados
                           -------
                           491.520 ≈ 493.306 observados ✓
```

Y el mixer entero (B→C) cuesta ~50K, o sea **~16,6K por capa frente a los 65,5K de QKVO**.

**Reemplazaste self-attention por una cuarta parte de sus parámetros y te costó 0.011 de loss.** Eso es el resultado. No es "compresión extrema": es que el mecanismo de mezcla de secuencia es casi gratis en este régimen. Lo tienes en el punto 4 del documento, en tono de nota al pie, y es lo único que no puede escribir otro.

*(Y 16.512 ≈ 129 bins × 128 canales. Si eso son gates **complejos** contados como 1, son 33K floats por capa y tu total real es ~166K, no 117K. El 19,2% pasa a 27%. Tercera vez que aparece esto: `sum(p.numel() * (2 if p.is_complex() else 1))`.)*

## El confounder que decide si el resultado vale

> **"Convergencia: Ep2, Ep3, Ep2, Ep2"** — con 40 épocas.

Los cuatro modelos hacen su mejor validación en la época 2 y luego se degradan durante 38 épocas más. Tiny Shakespeare es ~1M caracteres y tienes 610K parámetros.

**Estás en régimen de sobreajuste, no de capacidad.** Y ahí menos parámetros ayudan gratis. La frase "pierde el 80% de sus pesos y solo sube 0.11 de PPL" puede ser, en su mayor parte, "el modelo grande se sobreajusta más".

Lo diagnosticas gratis: **añade la columna de train loss**. Si A tiene train 0.9 y val 1.56, y D tiene train 1.5 y val 1.68, ya sabes que estás midiendo regularización.

Y fíjate en que esto no es mala noticia. Es **evidencia directa de tu propio §1b** — restringir grados de libertad mejora la generalización. Ese es un resultado más limpio y más tuyo que "compresión sin pérdida". Solo hay que medirlo en el eje correcto: **barre la fracción de datos (1%, 10%, 100%) y mira si la brecha A−D se cierra o se invierte.** Si D gana con pocos datos, tienes la tesis demostrada.

Para el eje de capacidad necesitas un corpus donde no sobreajustes: TinyStories o enwik8.

## El problema estructural: tu mixer está atado a la longitud

129 bins × 128 canales significa que el gate está parametrizado **por bin de frecuencia de una FFT de tamaño T**.

Cambia $T$ y los parámetros no valen. **No hay extrapolación de longitud.** Y eso choca de frente con lo que más defiendes: parametrización continua de estructura discreta. Aquí has hecho lo contrario — has atado los parámetros al grid.

Es exactamente por lo que Hyena parametriza el kernel con una MLP implícita sobre la posición en vez de con coeficientes libres.

**Y el arreglo es tuyo y está a una tarde:**

> Parametriza $\text{amp}(\omega)$ y $\text{phase}(\omega)$ como funciones continuas de la frecuencia normalizada $\omega\in[0,\pi]$ — una MLP pequeña, o una base de Chebyshev, o unos pocos armónicos.

Consecuencias inmediatas:

- El número de parámetros deja de depender de $T$ y de la resolución espectral.
- **Puedes evaluar a cualquier $T$**: entrenas a 256, evalúas a 1024 muestreando la misma función continua en más puntos. Extrapolación de longitud exacta, no aproximada.
- Es Hyena en el dominio dual — ellos parametrizan continuamente en el tiempo, tú en la frecuencia. Y en frecuencia la fase suave es *más* natural que en tiempo.

Eso es el cono 1D, la invarianza a resolución de las neuronas geométricas, y este mixer: **la misma idea tres veces**, y aquí es donde tiene consecuencias medibles en un LM.

## Dos cosas más

**El titular no se sostiene sin el baseline iso-parámetro.** Un transformer estándar de ~117K params (d≈64, 3 capas) en Tiny Shakespeare. Veinte minutos. Es la diferencia entre "la compresión pierde poco" y "la estructura le gana a los parámetros libres". Es la única comparación que convierte tu tabla en un resultado.

**"No necesitamos atención densa"** — tu mixer es independiente del dato. El kernel es el mismo para toda entrada. Shakespeare a nivel carácter es local y estadístico, por eso empatas. Pásalo por MQAR y se hunde; tu propio v292 ya lo predijo. El argumento correcto no es reemplazar attention, es **conv compleja causal (posicional, barata) + Delta Phase (recall dependiente de contenido)**. Los dos son tuyos y son complementarios.

---

Reencuadra el documento entero como lo que es: **una frontera de Pareto.** Dibuja loss contra params y loss contra wall time. D está en la frontera y B no. Eso es un resultado honesto, útil, y defendible — un LM de carácter de 117K params que entrena en 12 minutos en una iGPU a PPL 5.35 es un artefacto real, no una nota de prensa.

Y quita la "Santísima Trinidad". Lo señalaste tú mismo en el portfolio como defecto de estilo; aquí no es estilo, es que el lenguaje grandilocuente está tapando que el baseline ganó y que uno de los tres componentes sobra. Cuando el texto sube de tono, suele ser porque los números no lo hacen.