# Hallazgos del Experimento: Compresión Espectral Zero-Shot "BMP a JPG" (v288)

Este documento resume los resultados obtenidos en el experimento **v288**, que evalúa la analogía de compresión "BMP a JPG" para los pesos de una red neuronal pre-entrenada (**GPT-2**), comparando la estabilidad en la perplejidad bajo tres métodos de poda de coeficientes.

---

## 1. Configuración Experimental
* **Modelo Evaluado**: GPT-2 Small (124M parámetros, Hugging Face).
* **Capas Afectadas**: Proyecciones lineales de todos los bloques de atención (`c_attn`, `c_proj`) y MLP (`c_fc`, `c_proj`).
* **Dataset de Evaluación**: Tiny Shakespeare (20 secuencias de longitud 512, totalizando 10,240 tokens).
* **Parámetro de Estabilización**: **Variance Rescaling** aplicado en la reconstrucción del peso.
* **Dispositivo**: CPU.

---

## 2. Resultados Oficiales (Perplejidad en WikiText-2 / Tiny Shakespeare)

* **Perplejidad Baseline (Sin comprimir)**: **89.5758**

A continuación se muestra la comparación de perplejidad (PPL) para cada método según el porcentaje de parámetros eliminados (ratio de compresión):

| Ratio de Compresión | Poda Espacial (Baseline) | Paso Bajo DCT (JPG Slice) | Umbral de Energía DCT (JPG Coefs) |
| :---: | :---: | :---: | :---: |
| **0% (Base)** | 89.58 | 89.58 | 89.58 |
| **10%** | **89.42** | 2832.09 | 95.41 |
| **30%** | 97.08 | 7657.95 | **93.88** (Supera a Espacial) |
| **50%** | **342.84** | Explosión | 1625.29 |
| **70%** | **2923.19** | Explosión | 3385.14 |
| **80%** | 5453.53 | Explosión | **2944.36** (Supera a Espacial) |
| **90%** | **4897.35** | Explosión | 8468.90 |

*Nota: "Explosión" indica una perplejidad superior a 10,000, reflejando un colapso completo del lenguaje.*

---

## 3. Hallazgos Fundamentales

### A. Fallo Catastrófico del Paso Bajo DCT (JPG Slice)
Al conservar estrictamente solo la zona de bajas frecuencias (el cuadrante superior izquierdo del espectro DCT), la red sufre una degradación de PPL instantánea (2832.09 a solo 10% de compresión).
* **Insight**: A diferencia de las imágenes donde las altas frecuencias representan ruido visual o detalles que el ojo humano ignora, en los pesos de un LLM **las altas frecuencias son esenciales**. Codifican las diferencias sutiles entre cabezas y dimensiones necesarias para la correcta distribución de la atención.

### B. Éxito de la Compresión por Umbral de Energía DCT
Al ordenar los coeficientes DCT por magnitud absoluta y conservar solo el top $(1 - \text{ratio})$, el modelo demuestra una resiliencia excepcional en comparación con el corte por frecuencias.
* **Insight 1 (Compresión Baja)**: Con un **30% de compresión**, el Umbral de Energía DCT logra **93.88 PPL**, superando a la poda espacial estándar (**97.08 PPL**).
* **Insight 2 (Compresión Alta)**: Con un **80% de compresión**, el Umbral de Energía DCT retiene **2944.36 PPL**, superando sustancialmente a la poda espacial (**5453.53 PPL**).
* **Conclusión**: Excluir los componentes de baja magnitud en el dominio frecuencial es un método de compresión superior a la poda espacial a ratios moderados y altos. La base DCT concentra la energía semántica en componentes clave, actuando como un excelente regularizador.

---

## 4. Próximos Pasos Recomendados

1. **Entrenamiento Nativo con Regularización Espectral (L1 en DCT)**: Dado que el Paso Bajo zero-shot falla por la dependencia en altas frecuencias, proponer un entrenamiento (pre-training) donde se penalice la norma L1 del espectro. Esto forzará al modelo a encontrar una solución de baja frecuencia que permita compresión de paso bajo sin pérdida de perplejidad.
2. **Cuantización Espectral Jerárquica**: Evaluar la cuantización de los coeficientes de secuencialidad (de v229) sobre este benchmark de GPT-2 para determinar si una asignación variable de bits en el dominio de frecuencia es superior a la poda de coeficientes.



---


## Lo primero: la mitad de la tabla no contiene información

Mira las columnas verticalmente:

| | Espacial | DCT Energía |
|---|---|---|
| 70% | 2923 | 3385 |
| 80% | **5453** | **2944** ↓ |
| 90% | **4897** ↓ | 8469 |

**Más compresión, mejor perplejidad.** Dos veces, en las dos columnas.

Eso no es un efecto: es el diagnóstico de que por encima de ~200 PPL el modelo está muerto y los números miden **cómo de muerto**, que es una variable sin orden. Un modelo colapsado puede dar 3.000 o 8.000 según hacia qué distribución degenerada caiga, y eso no guarda relación con cuánta estructura conservaste.

Así que el "Insight 2" —*"a 80% el DCT supera a espacial, 2944 vs 5453"*— compara dos cadáveres. Y el orden se invierte diez puntos más allá.

**Filas con contenido: 10% y 30%. Nada más.**

Y ahí:

| | 10% | 30% |
|---|---|---|
| Espacial | **89.42** | 97.08 |
| DCT Energía | 95.41 | **93.88** |

A 10% gana espacial claramente. A 30% gana DCT por 3.2. Y la columna DCT es **no monótona en la zona viva** (95.41 → 93.88 al comprimir más), lo que sugiere que ±3 PPL está dentro del ruido de un eval de 10K tokens.

Tu resultado real es: *un cruce entre 10% y 30%, de magnitud incierta*. Eso es defendible y pequeño.

*(Crédito donde toca: la fila del 0% da 89.58 exacto en las tres columnas. Eso valida el round-trip DCT→IDCT. Es un perft, y está bien que esté.)*

---

## Por qué falla el paso bajo — y tu explicación no es la correcta

Dices: *"las altas frecuencias son esenciales, codifican diferencias sutiles entre cabezas"*.

La razón real es más simple y más útil: **no hay ninguna razón para que un peso tenga estructura de baja frecuencia.**

Las neuronas ocultas son **permutación-invariantes**. El orden de filas y columnas de $W$ es una convención arbitraria del proceso de entrenamiento. La "frecuencia" a lo largo de un eje sin orden intrínseco es ruido: el espectro está **plano**. Un filtro paso bajo tira el 90% de la energía elegida esencialmente al azar.

No es que las altas frecuencias sean importantes. Es que **la etiqueta "alta frecuencia" no significa nada aquí.**

Y tú ya tienes la demostración: **V290**. Reordenar canales por TSP mueve la PPL de 163.95 a 88.36. Un factor 2 solo por permutar. Si el espectro tuviera significado semántico, permutar no cambiaría nada.

**V288 y V290 son el mismo hallazgo:**

> V288: la base DCT no está alineada con el orden arbitrario de canales → el paso bajo destruye.
> V290: si buscas la permutación que la alinea, el paso bajo funciona.

Presentados juntos son un arco coherente y una contribución real. Por separado, V288 parece un fracaso y V290 un truco.

---

## Distinción que te debo: podar y cuantizar quieren bases opuestas

Antes te dije que tu línea DCT era pariente de QuaRot/QuIP. Hay que matizarlo, porque el matiz es útil:

| | Quiere del espectro |
|---|---|
| **Cuantizar** | energía **repartida**, sin outliers → rotaciones aleatorias/Hadamard hacen los pesos más gaussianos (TCL) y matan los outliers. **Eso es QuIP/QuaRot.** |
| **Podar** | energía **concentrada**, colas pesadas → quieres que pocos coeficientes lleven casi todo |

**Son objetivos opuestos.** Una rotación aleatoria *empeora* la poda por magnitud, precisamente porque gaussianiza.

Así que tu V288 (poda) y tu V289 (cuantización) buscan cosas contrarias en la misma maquinaria, y conviene que estén separados en tu cabeza y en el ledger.

Y de ahí sale el baseline que falta: **rotación ortogonal aleatoria + umbral de energía.** Si empata con DCT, no es frecuencia; si pierde claramente, la DCT sí concentra. Una línea, y es la quinta vez que aparece esta misma pregunta en tu corpus.

*(La respuesta óptima teórica a "qué base concentra la energía" es la **KLT/PCA** — la eigenbase de la covarianza de filas. La DCT es su aproximación asintótica para procesos AR(1) muy correlados, que un peso no es. Calcular la KLT real es barato y es el techo contra el que medir. Y tu TSP es, en el fondo, un intento de acercar la DCT a la KLT reordenando.)*

---

## El baseline espacial está muy por debajo del estado del arte

50% de poda por magnitud → 342 PPL. Eso es lo esperable de la magnitud desnuda, y por eso nadie la usa:

- **SparseGPT** (Frantar & Alistarh 2023): 50% one-shot en modelos GPT con degradación casi nula. Usa información de segundo orden de un set de calibración.
- **Wanda** (Sun et al. 2023): saliencia = $|W_{ij}| \cdot \|X_j\|_2$. Una línea de código, sin retraining, y aplasta a la magnitud pura.

La lección de los dos: **la magnitud del peso es un criterio de saliencia malo; necesitas estadística de activaciones.** Un peso pequeño multiplicado por una activación grande importa más que un peso grande sobre un canal muerto.

Eso te da la mejora obvia y barata: **umbral de energía DCT ponderado por la norma de activación del canal.** Es Wanda en el dominio transformado, y probablemente mueve tu cruce del 30% a bastante más.

*(Y ojo: tu método es zero-shot y eso es una virtud real que debes reivindicar — pero entonces el competidor honesto es Wanda, que casi lo es, no SparseGPT.)*

---

## El eval

**La cabecera dice "WikiText-2 / Tiny Shakespeare" y la config dice Shakespeare, 10.240 tokens.** Arregla eso antes de que lo vea otro; es el mismo tipo de contaminación que el 3.5 de nanoGPT.

Y luego córrelo de verdad en WikiText-2:

- Es **in-distribution** para GPT-2. Shakespeare es OOD, y en OOD la poda actúa como suavizado y **mejora** artificialmente. Tu 89.42 < 89.58 a 10% es exactamente ese efecto, y probablemente desaparece en distribución.
- El test estándar son ~280K tokens: 27× más muestra.
- Y es la referencia que permite comparar con cualquier paper de poda.

Si tu cruce del 30% sobrevive en WikiText-2, es un resultado. Si desaparece, has aprendido que era un artefacto de evaluar fuera de distribución — y eso también vale, porque V289 y V290 cuelgan de aquí.

*(Y documenta qué hace exactamente el "Variance Rescaling". Reescalar la varianza reconstruida puede estar haciendo buena parte del trabajo, y no está ablacionado.)*

---

## Lo que este experimento cierra, y es bueno

Tu "próximo paso 1" es: *entrenar con L1 sobre el espectro DCT para forzar una solución de baja frecuencia*.

**Ya lo hiciste. Es V63 y V283.**

Y ahí está el arco completo, que es un hallazgo de verdad:

> **No puedes proyectar un modelo denso ya entrenado a un subespacio de baja frecuencia (V288: colapsa a 2832 con solo 10%). Pero sí puedes entrenar directamente dentro de ese subespacio (V283: 42K params, mejor loss que el denso equivalente).**

Es exactamente la distinción que te señalé en V63 —"no descartaste grados de libertad, es que nunca existieron"— y ahora la tienes demostrada empíricamente por los dos lados.

Eso no es una nota al pie. Es una afirmación sobre optimización, no sobre compresión: **el subespacio contiene buenas soluciones, pero la solución densa no está cerca de él.** Predice el fracaso del paso bajo post-hoc, predice el éxito de V283, y sugiere lo obvio: entrenamiento con proyección progresiva al subespacio, o partir de un modelo denso y hacer *fine-tuning* dentro de la restricción en vez de proyectar de golpe.

Ese párrafo vale más que toda la tabla.