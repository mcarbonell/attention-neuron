# Findings v331 — Ablación causal de dos ramas y bases espectrales

> **Estatus:** Nivel 2, cinco semillas de entrenamiento y 1,024 secuencias retenidas no solapadas en validación y test por semilla. Resultado central: [RUIDO-SOSPECHA]. Ninguna comparación causal alcanza el criterio pre-registrado de magnitud `2 × SE`.

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

v330b dejó una tendencia no concluyente de `lerp_fwht_dct` frente a una **sola** base ortogonal aleatoria (`Δ=-0.01536`, `SE=0.01041`). Esa comparación confundía la estructura de bases con la topología: Lerp tiene dos ramas, parámetros de modulación por rama y un combinador, mientras `random_orthogonal` tenía una rama.

v331 mantiene constante la topología Lerp de dos ramas y cambia únicamente el par de matrices congeladas. El control crítico `lerp_random_pair` usa dos matrices ortogonales aleatorias independientes, con exactamente los mismos 59,205 parámetros entrenables. El resultado modifica la lectura previa:

1. La tendencia v330b no se convierte en una ventaja específica de FWHT+DCT: `FWHT+DCT − RandomPair = -0.00137`, con `2×SE=0.00893`.
2. Tampoco hay evidencia de que dos bases aleatorias distintas superen a dos copias de una misma base: `RandomPair − RandomTied = -0.00562`, con `2×SE=0.03182`.
3. Por tanto, v330/v330b no sostienen una atribución causal a geometría espectral ni a diversidad de dos bases; quedan como [RUIDO-SOSPECHA] bajo este alcance.

## 1. Pregunta y protocolo

**Pregunta primaria:** a igual arquitectura Lerp de dos ramas, router global, combinador, datos, entrenamiento y parámetros, ¿la pareja fija FWHT+DCT-II mejora a dos bases ortogonales aleatorias independientes?

- **Corpus y split:** Tiny Shakespeare real a nivel de carácter; split temporal 70/15/15.
- **Backbone:** embedding, posición seno/coseno fija, dos bloques de atención causal QK$^T$, LayerNorm y head lineal.
- **Entrenamiento:** AdamW, `d_model=64`, 4 cabezas, 2 bloques, secuencia 128, batch 16, 30 épocas, 150 pasos/época, LR $3\times10^{-3}$, `weight_decay=0`, clip 1.0.
- **Semillas de entrenamiento:** `[10, 20, 30, 42, 100]`.
- **Evaluación:** 64 batches fijos, retenidos y no solapados por partición (1,024 secuencias); checkpoint de mínima `valid_loss`; test ejecutado una única vez después de seleccionar checkpoint.
- **Bases aleatorias:** R0/R1 se generan de forma determinista por semilla de entrenamiento y capa. Dentro de una semilla, las condiciones que usan R0/R1 comparten exactamente esas matrices; el JSON conserva semilla y SHA-256 de cada buffer.

## 2. Igualación de capacidad y condiciones

| Variante | Parámetros | Condición |
| :--- | ---: | :--- |
| `lerp_fwht_dct` | 59,205 | Pareja estructurada FWHT+DCT-II. |
| `lerp_random_pair` | 59,205 | R0 y R1 independientes; control crítico. |
| `lerp_random_tied` | 59,205 | R0 y R0; mismo router y dos ramas, sin diversidad de base. |
| `lerp_fwht_random` | 59,205 | FWHT y R0. |
| `lerp_dct_random` | 59,205 | DCT-II y R0. |
| `dense_ffn` | 59,207 | Referencia externa; +2 parámetros respecto a Lerp. |

Las cinco condiciones Lerp son exactamente iso-paramétricas y tienen el mismo código de FFN; sólo cambian buffers fijos. Dense no forma parte de la atribución causal entre pares, pero mantiene la referencia práctica de calidad y coste.

## 3. Resultados Nivel 2

| Modelo | Test loss media ± SE | Test PPL media ± SE | Wall-clock medio | Etiqueta |
| :--- | ---: | ---: | ---: | :--- |
| `lerp_dct_random` | **1.94472 ± 0.00900** | **6.9928 ± 0.0633** | 214.68 s | [RUIDO-SOSPECHA] |
| `dense_ffn` | 1.95270 ± 0.00635 | 7.0482 ± 0.0448 | **157.14 s** | [SEÑAL] |
| `lerp_fwht_dct` | 1.95366 ± 0.00663 | 7.0551 ± 0.0467 | 204.39 s | [RUIDO-SOSPECHA] |
| `lerp_random_pair` | 1.95504 ± 0.00987 | 7.0655 ± 0.0699 | 183.05 s | [SEÑAL] |
| `lerp_fwht_random` | 1.95930 ± 0.00720 | 7.0951 ± 0.0512 | 217.77 s | [RUIDO-SOSPECHA] |
| `lerp_random_tied` | 1.96065 ± 0.01108 | 7.1057 ± 0.0785 | 183.34 s | [RUIDO-SOSPECHA] |

El orden numérico no equivale a una mejora: la variante DCT+Random es primera por media, pero su margen frente al control crítico no satisface el umbral predefinido. Dense tiene el menor coste; `lerp_fwht_dct` tarda 30.1% más y `lerp_random_pair` 16.5% más.

## 4. Comparaciones emparejadas pre-registradas

Valores negativos favorecen la primera variante. El criterio de lectura fue fijado antes de ejecutar: una magnitud inferior a `2×SE` no se considera evidencia de mejora.

| Comparación | Δ loss media | SE emparejado | 2 × SE | Lectura |
| :--- | ---: | ---: | ---: | :--- |
| FWHT+DCT − RandomPair | -0.00137 | 0.00446 | 0.00893 | Primaria: no distingue estructura espectral de dos bases aleatorias. |
| RandomPair − RandomTied | -0.00562 | 0.01591 | 0.03182 | No hay evidencia de un efecto de diversidad de bases. |
| FWHT+Random − RandomPair | +0.00427 | 0.00480 | 0.00960 | No hay señal favorable atribuible a FWHT aislado. |
| DCT+Random − RandomPair | -0.01031 | 0.00905 | 0.01811 | Tendencia favorable a DCT+Random, no suficiente. |
| FWHT+DCT − Dense | +0.00097 | 0.00871 | 0.01742 | Prácticamente empate de calidad; Dense queda nominalmente delante y es más rápido. |
| DCT+Random − Dense | -0.00798 | 0.01043 | 0.02087 | Primera media nominal, no distinguible de Dense. |

## 5. Convergencia

Los checkpoints seleccionados están entre las épocas 27 y 30. Denso selecciona la época 30 en las cinco semillas; las variantes Lerp seleccionan mayoritariamente 28–30. La validación media de la época 30 aún es comparable o algo menor que la del checkpoint en varias condiciones, de modo que no se debe interpretar este presupuesto como rendimiento asintótico.

Esta limitación no altera la lectura causal principal: las condiciones comparadas tienen el mismo presupuesto y la diferencia primaria FWHT+DCT–RandomPair es mucho menor que su incertidumbre emparejada.

## 6. Conclusión

En este protocolo, las bases FWHT+DCT-II no muestran una mejora verificable sobre dos bases ortogonales aleatorias independientes dentro de la misma arquitectura Lerp. La señal previa de v330b frente a una rama aleatoria no admite ya una interpretación específica de base espectral. Tampoco se confirma que la diversidad de dos bases sea la explicación.

La referencia práctica sigue siendo Dense: está indistinguible de FWHT+DCT en calidad (`+0.00097 ± 0.00871` para Lerp−Dense) y cuesta menos tiempo. `lerp_dct_random` merece anotarse como tendencia exploratoria, no como dirección de explotación, pues no supera `2×SE` frente a RandomPair ni Dense.

## 7. Auditoría posterior y amenazas a la validez (2026-08-11)

1. **Varianza de optimización y de base están mezcladas.** Cada semilla de entrenamiento induce una realización R0/R1 distinta. Esto evita depender de una única matriz aleatoria, pero no permite descomponer por separado ambas fuentes de varianza. Si DCT+Random vuelve a sugerir señal, ejecutar un diseño cruzado con varias semillas de base por cada semilla de entrenamiento.
2. **Convergencia incompleta.** Checkpoints tardíos sugieren que 30 épocas no garantizan meseta. Una extensión sólo se justifica si una hipótesis nueva concreta predice que cambie el contraste DCT+Random–RandomPair; prolongar todas las condiciones no resuelve por sí mismo la ambigüedad actual.
3. **Alcance externo reducido.** Tiny Shakespeare char, `d_model=64` y dos bloques no representan BPE, contextos largos ni escalas LLM. Cualquier réplica positiva debe pasar primero a un corpus/tokenizador real de mayor escala.
4. **Coste de implementación.** Las bases son matrices materializadas y se aplican con `F.linear` en CPU. Los tiempos comparan esta implementación, no kernels FWHT/DCT compilados ni FLOPs algorítmicos.
5. **Cinco semillas aún tienen resolución limitada.** La tendencia DCT+Random–RandomPair (-0.01031) no llega a `2×SE=0.01811`. Debe permanecer [RUIDO-SOSPECHA], aunque sea la menor loss media de la tabla.

## 8. Artefactos reproducibles

- Script: `scratch/prototype_v331_two_branch_basis_ablation.py`
- Plan pre-registrado: `docs/experiment_plan_v331_two_branch_basis_ablation.md`
- JSON Nivel 2: `results/raw/v331_two_branch_basis_ablation_20260811T113633Z.json`
