# Auditoría transversal v300–v329: evidencia, límites y reclasificación

> **Fecha de auditoría:** 2026-08-10. Este documento registra una revisión posterior de scripts, logs y *findings*. No modifica resultados numéricos ni elimina hallazgos: corrige el alcance inferencial que permiten los arneses implementados.

## Resumen ejecutivo

La evidencia más sólida de la serie es **v306**: en Tiny Shakespeare real, con *split* de validación, presupuesto iso-paramétrico y cinco semillas, `ChunkwiseComplexDeltaPhase` obtiene una mejora media frente al control real iso-paramétrico. Usando las pérdidas por semilla registradas, el contraste pareado da aproximadamente $p=0.005$ (Welch, $p\approx0.003$). Es evidencia positiva de magnitud modesta; la etiqueta previa $p<0.001$ es demasiado fuerte para $n=5$.

**v305** también conserva valor como diagnóstico: identificó que arneses MQAR estáticos permitían memorizar lotes y documentó la corrección mediante generación *on-the-fly*. Por esa razón, las conclusiones sintéticas de v300–v303 deben considerarse exploratorias y no comparables directamente con el harness corregido.

El resto de la serie mezcla dos familias que deben permanecer separadas:

1. **DeltaPhase en LM real:** v304 es exploratorio y v306 es la evidencia principal; v307 no es una evaluación en TinyStories/BPE real.
2. **FFN/adaptadores espectrales en patrones sintéticos:** v308–v329 generan una regla aritmética fija. Son útiles para depurar mecanismos y formular hipótesis, pero aún no demuestran transferencia a lenguaje ni superioridad arquitectónica general.

## Hallazgos de auditoría

### 1. v300–v303: el cambio de harness rompe la continuidad histórica

El script de v300 pre-genera los lotes de entrenamiento y evaluación. v305 documenta que este enfoque estático permitía memorización y que el benchmark se corrigió con lotes nuevos en cada paso. Por tanto, las ventajas, colapsos y curvas de escalado de v300–v303 no deben promocionarse ni agregarse como evidencia de rendimiento relativo hasta ser repetidas en el harness *on-the-fly* certificado.

El resultado negativo de overwrite en v303 sigue siendo informativo como hipótesis de fragilidad, pero no permite atribuir con seguridad el efecto a la Delta Rule antes de repetirlo en el harness corregido y con varias semillas.

### 2. v307 no contiene TinyStories ni BPE real

`run_v307_tinystories_bpe_lm.py` no carga un corpus ni aplica un tokenizador BPE. `generate_subword_dataset` muestrea cada token de forma independiente de una distribución Zipf y usa el desplazamiento como target. Así, el siguiente token no contiene información contextual predecible; el experimento mide principalmente ajuste de frecuencia marginal y estado de optimización, no modelado de lenguaje, recuperación ni uso de contexto.

La loss de entropía de la distribución Zipf usada ($V=4096$, exponente $0.8$) es aproximadamente $7.188$ nats. Las losses observadas ($\sim7.69$) están aún por encima de ese predictor unigram óptimo. v307 debe reclasificarse como **stress test sintético Zipf i.i.d.**, no como validación BPE/TinyStories ni confirmación de transferencia de v306.

### 3. v308–v321: suelo de entropía y ausencia de evaluación retenida

Los scripts de v309–v321 usan la misma regla:

$$y_t=(3x_{t-1}+x_t+7)\bmod64, \qquad x_t\sim\mathrm{Uniform}\{0,\ldots,31\}.$$

En los modelos tokenwise (en particular v309–v321 sin mecanismo de mezcla temporal), $x_{t-1}$ no está disponible. Dado $x_t$, quedan 32 respuestas equiprobables; el óptimo teórico es $H(Y\mid X_t)=\ln32\approx3.4657$ nats. Las losses reportadas de aproximadamente 3.47–3.49 están en dicho suelo. Las diferencias de milésimas no prueban expresividad, sparsity o superioridad de una base sin test retenido y análisis multi-semilla.

Además, v308–v320 reportan normalmente la última loss de un batch de entrenamiento; v321 y posteriores agregan por época, pero continúan midiendo el mismo conjunto usado para optimizar. Ninguno de esos números es una métrica de generalización.

### 4. v322–v329: señal sintética de mezcla temporal, no validación de LLM

Desde v322 hay atención causal, de modo que el modelo sí puede consultar $x_{t-1}$ y resolver la regla fija. Esta transición explica el descenso de loss y las accuracies altas: muestra capacidad de aprender una dependencia de un paso en distribución, no memoria asociativa de largo alcance ni desempeño en lenguaje natural.

No hay split train/valid/test, evaluación *on-the-fly*, semillas múltiples ni presupuestos de entrenamiento seleccionados independientemente. Las conclusiones deben restringirse a: “en esta regla sintética y este presupuesto, el candidato optimiza mejor en entrenamiento”.

La etiqueta **“iso-paramétrico”** es incorrecta en v322b (685,184 parámetros y 5 bloques espectrales frente a 412,352 parámetros y 2 bloques densos) y sólo aproximada en v325, donde también cambian profundidad y parámetros entre lados. v327 compara una fusión con mayor capacidad de combinación que las bases puras. Estos experimentos son ablations de configuración, no pruebas limpias de eficiencia intrínseca.

### 5. Interpretaciones de implementación que requieren acotación

- El “Learnable Substrate Lerp Router” de v328 usa tres logits globales por capa. No enruta por token ni por canal; sus porcentajes describen una mezcla global de capa.
- El mezclador de secuencia de v329 no es estrictamente “0 parámetros”: incluye parámetros de fase y amplitud de longitud de secuencia. El resultado sí respalda que **ese mezclador estático concreto** es inferior a QK$^T$ en esta tarea; no demuestra que toda alternativa sin atención sea imposible.
- PEI, definido como $1/(\mathrm{loss}\cdot\log_{10}(\mathrm{params}))$, diverge al acercarse la loss a cero y mezcla una métrica de entrenamiento con tamaño del modelo. No debe usarse para afirmar superioridad. Conviene sustituirlo por un frente de Pareto validado: loss/accuracy de validación, parámetros, FLOPs y latencia medida.

## Reclasificación de evidencia

| Bloque | Estado recomendado | Qué se puede afirmar hoy |
| :--- | :--- | :--- |
| v300–v303 | Nivel 1, supersedido por cambio de harness | Hipótesis y diagnóstico histórico; repetir *on-the-fly*. |
| v304 | Nivel 1 | Primera señal en texto; presupuesto no igualado. |
| v305 | Diagnóstico válido | El harness estático era vulnerable a memorización; MHA fue certificado tras la corrección. |
| v306 | Nivel 2, evidencia positiva moderada | DeltaPhase mejora al control real iso-paramétrico en Tiny Shakespeare de caracteres, con $n=5$. |
| v307 | Nivel 1 sintético, renombrar | Comparación en Zipf i.i.d.; no es TinyStories ni BPE real. |
| v308–v321 | Nivel 0/1 de mecanismo | Comportamiento cerca de un suelo de entropía tokenwise; no hay generalización medida. |
| v322–v329 | Nivel 1 de mecanismo | Aprendizaje en entrenamiento de una regla causal local; no extrapolar a LLMs. |

## Requisitos mínimos antes de una nueva promoción

1. Corpus real, tokenizer y hashes/versiones explícitos; separar por completo train/valid/test.
2. Cinco semillas como mínimo, idealmente diez para efectos pequeños, con diferencias por semilla y contraste estadístico predefinido.
3. Misma profundidad, parámetros, FLOPs, optimizador, schedule y número de tokens de entrenamiento; informar cualquier desigualdad como ablation, no como iso-presupuesto.
4. Para bases espectrales, incluir una matriz ortogonal aleatoria congelada y permutaciones de base como controles.
5. Para memoria, evaluar MQAR *on-the-fly* con longitud y distancia de consulta fuera de distribución, además de LM real.
6. Reportar el mejor checkpoint elegido exclusivamente por validación y medir la latencia tras calentamiento, con la misma implementación/dispositivo.

