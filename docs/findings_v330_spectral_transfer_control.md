# Findings v330 — Transferencia espectral controlada en Tiny Shakespeare

> **Estatus:** evaluación comparativa de Nivel 2 ($n=5$ semillas, 1,024 ventanas retenidas no solapadas en validación y test por semilla). El resultado no promueve ninguna variante espectral a [ANCLA]. Las señales favorables de Lerp se mantienen como [RUIDO-SOSPECHA] hasta una extensión de convergencia.

## 0. Sección Obligatoria de Reconciliación: Qué Conclusión Previa Modifica o Invalida este Experimento

v321–v329 sugerían que FWHT, DCT-II y sus fusiones podían ser sustitutos superiores de FFNs densos y recomendaban su despliegue en un LLM. Aquellos resultados se obtuvieron en entrenamiento de una regla sintética, sin test retenido, sin control ortogonal aleatorio y, en varios casos, con presupuestos no igualados.

v330 reevalúa la hipótesis en texto real con un único backbone causal, datos reales y tres controles críticos: FFN denso de tamaño emparejado, base ortogonal aleatoria congelada y bases puras FWHT/DCT-II. El resultado modifica la lectura anterior:

1. No se observa evidencia de que FWHT o DCT-II aporten una ventaja específica frente a una rotación ortogonal aleatoria.
2. `lerp_fwht_dct` no mejora de forma estadísticamente distinguible al FFN denso ni al control ortogonal aleatorio.
3. El FFN denso obtiene la menor test loss media bajo este protocolo.

## 1. Pregunta, protocolo y controles

**Pregunta primaria:** ¿FWHT/DCT-II aportan una ventaja específica sobre una rotación ortogonal aleatoria fija cuando se mantienen constantes la mezcla causal, datos, profundidad, tokens de entrenamiento y presupuesto de parámetros?

- **Corpus:** Tiny Shakespeare real a nivel de carácter, split temporal 70% train / 15% valid / 15% test.
- **Backbone compartido:** embedding, posición seno/coseno fija, dos bloques de atención causal QK$^T$, LayerNorm y head lineal.
- **Selección de checkpoint:** mínima `valid_loss`; el test se ejecuta una vez sobre el checkpoint elegido.
- **Entrenamiento:** AdamW, $d_{model}=64$, 4 cabezas, secuencia 128, batch 16, 15 épocas, 150 pasos/época, LR $3\times10^{-3}$, `weight_decay=0` y clip de gradiente 1.0.
- **Semillas:** `[10, 20, 30, 42, 100]`.
- **Datos de comparación:** ventanas de evaluación fijas y no solapadas; para una misma semilla, todos los modelos reciben los mismos batches de entrenamiento y evaluación.

### ¿Es iso-paramétrico?

**Sí, a efectos prácticos de número de parámetros entrenables.** La diferencia máxima es de 6 parámetros sobre aproximadamente 59.2k, un $0.010\%$:

| Modelo | Parámetros | Diferencia vs. mínimo |
| :--- | ---: | ---: |
| `fwht` | 59,201 | 0 |
| `dct` | 59,201 | 0 |
| `random_orthogonal` | 59,201 | 0 |
| `lerp_fwht_dct` | 59,205 | +4 |
| `dense_ffn` | 59,207 | +6 |

La igualdad es paramétrica, **no computacional**. Lerp evalúa dos transformadas y dos ramas, mientras el FFN denso ejecuta sus proyecciones densas convencionales. Por ello, cualquier conclusión de eficiencia debe usar latencia/FLOPs medidos, no sólo parámetros.

## 2. Resultados de test

| Modelo | Test loss media ± SE entre semillas | Test PPL media ± SE | Wall-clock medio | Etiqueta |
| :--- | ---: | ---: | ---: | :--- |
| **`dense_ffn`** | **1.99637 ± 0.00809** | **7.3633 ± 0.0601** | **74.30 s** | [SEÑAL] |
| `lerp_fwht_dct` | 2.00148 ± 0.00658 | 7.4006 ± 0.0486 | 106.22 s | [RUIDO-SOSPECHA] |
| `random_orthogonal` | 2.01073 ± 0.00820 | 7.4698 ± 0.0613 | 114.14 s | [SEÑAL] |
| `fwht` | 2.01313 ± 0.00474 | 7.4871 ± 0.0355 | 107.28 s | [SEÑAL] |
| `dct` | 2.01698 ± 0.00383 | 7.5158 ± 0.0288 | 103.64 s | [SEÑAL ADVERSA] |

El candidato Lerp usa $42.9\%$ más wall-clock que denso ($106.22$ s frente a $74.30$ s) y $50.9\%$ más tiempo de evaluación de función ($47.24$ s frente a $31.31$ s). No hay ventaja de coste en la implementación actual de PyTorch/CPU.

## 3. Comparaciones emparejadas por semilla

La unidad de comparación es la diferencia de test loss de cada semilla, no la superposición visual de SEs agregados. Valores negativos favorecen la primera variante nombrada.

| Comparación | $\Delta$ loss media | SE emparejado | Umbral $2\times SE$ | Lectura |
| :--- | ---: | ---: | ---: | :--- |
| Lerp − Denso | +0.00511 | 0.00940 | 0.01881 | No distinguible; denso queda nominalmente delante. |
| Lerp − Aleatorio | -0.00925 | 0.00759 | 0.01518 | Tendencia favorable a Lerp, no distinguible. |
| FWHT − Aleatorio | +0.00240 | 0.00864 | 0.01728 | No hay evidencia de efecto específico FWHT. |
| DCT − Aleatorio | +0.00625 | 0.00912 | 0.01824 | No hay evidencia de efecto específico DCT. |
| DCT − Denso | +0.02060 | 0.00749 | 0.01497 | Señal adversa para DCT bajo este setup; requiere controles antes de cierre definitivo. |

## 4. Curvas de convergencia

Los mejores checkpoints aparecen al final del presupuesto: época 15 para todas las semillas de DCT, FWHT y Lerp; denso y aleatorio seleccionan época 14 en dos semillas. Las validaciones medias continúan bajando entre épocas 14 y 15:

| Modelo | Val loss media, época 14 | Val loss media, época 15 |
| :--- | ---: | ---: |
| `dense_ffn` | 1.92407 | 1.91693 |
| `lerp_fwht_dct` | 1.92738 | 1.91722 |
| `random_orthogonal` | 1.94479 | 1.93972 |
| `fwht` | 1.94756 | 1.93212 |
| `dct` | 1.95349 | 1.93904 |

La cercanía denso/Lerp en validación al final ($\Delta\approx0.00029$) y el hecho de que las curvas no han llegado a meseta impiden interpretar el pequeño margen de test como una derrota estructural de Lerp.

## 5. Conclusión

En Tiny Shakespeare de caracteres y este presupuesto, el FFN denso obtiene el mejor resultado numérico y el menor coste. Lerp, FWHT y DCT no aportan una mejora verificable sobre el denso. La tendencia Lerp frente al control ortogonal aleatorio es insuficiente para afirmar que FWHT/DCT aporten geometría específica.

El resultado válido es negativo respecto a la afirmación fuerte “las bases espectrales sustituyen y mejoran sistemáticamente los FFNs densos”; no es un cierre definitivo de toda arquitectura espectral, porque las curvas siguen descendiendo y no se ha hecho un barrido de duración/LR.

## 6. Auditoría posterior y amenazas a la validez (2026-08-10)

1. **Convergencia incompleta:** casi todos los checkpoints son la época 15 y la validación aún baja. Para distinguir capacidad final de velocidad de convergencia, repetir sólo Lerp, denso y aleatorio durante 30 épocas, manteniendo todos los demás hiperparámetros.
2. **Coste de implementación, no límite algorítmico:** las bases se materializan como matrices y se ejecutan con `F.linear` en CPU. Esta medición no representa una FHT compilada ni FLOPs normalizados. Antes de un claim de eficiencia, medir inferencia tras calentamiento y/o un kernel compilado bajo el mismo output de modelo.
3. **Una sola escala y corpus pequeño:** $d_{model}=64$, dos bloques y Tiny Shakespeare de caracteres no justifican extrapolación a BPE/TinyStories o LLMs. El siguiente benchmark de transferencia debe usar corpus y tokenizador reales; v307 no es válido para ese propósito porque era Zipf i.i.d.
4. **Prueba emparejada incompleta:** el JSON conserva medias de test por semilla, pero no las pérdidas por secuencia ni los pesos finales del router. La próxima versión debe persistir ambas cosas para auditar varianza intra-seed y qué mezcla FWHT/DCT aprendió cada capa.
5. **Identidad de experimento en ledger:** el piloto y Nivel 2 usan el mismo `experiment_id` (`v330_spectral_transfer_control`). El ledger debe incorporar `run_id` o diferenciar explícitamente `v330_pilot` y `v330_level2` antes de consolidar gráficos globales.

## 7. Artefacto reproducible

- Script: `scratch/prototype_v330_spectral_transfer_control.py`
- Plan: `docs/experiment_plan_v330_spectral_transfer_control.md`
- JSON de Nivel 2: `results/raw/v330_spectral_transfer_control_20260810T101228Z.json`

## 8. Extensión v330b: 30 épocas (2026-08-10)

La extensión `v330b_spectral_transfer_extended_30ep` mantiene el mismo corpus, arquitectura, semillas, ventanas retenidas y presupuesto de parámetros. Sólo duplica el presupuesto de entrenamiento de 15 a 30 épocas. El JSON usa una identidad de ejecución distinta, por lo que no debe agregarse al piloto ni al Nivel 2 original como si fueran la misma corrida.

| Modelo | Test loss media ± SE entre semillas | Test PPL media ± SE | Wall-clock medio | Cambio de loss vs. 15 épocas |
| :--- | ---: | ---: | ---: | ---: |
| **`dense_ffn`** | **1.95392 ± 0.00685** | **7.0570 ± 0.0485** | **157.48 s** | -0.04245 |
| `lerp_fwht_dct` | 1.95631 ± 0.00680 | 7.0738 ± 0.0480 | 217.17 s | -0.04517 |
| `random_orthogonal` | 1.97166 ± 0.00383 | 7.1828 ± 0.0275 | 191.90 s | -0.03907 |
| `fwht` | 1.97291 ± 0.01018 | 7.1931 ± 0.0729 | 190.13 s | -0.04022 |
| `dct` | 1.97589 ± 0.00533 | 7.2135 ± 0.0384 | 187.96 s | -0.04108 |

Las cinco variantes siguen mejorando de 15 a 30 épocas, pero el orden no se invierte: denso queda primero y Lerp segundo. El margen nominal Lerp–denso se reduce de +0.00511 a +0.00238, sin convertirse en una ventaja de Lerp.

### Comparaciones emparejadas v330b

Valores negativos favorecen la primera variante. Se conserva el criterio exploratorio predefinido de `2 × SE` como umbral de magnitud frente al ruido entre semillas.

| Comparación | Δ loss media | SE emparejado | 2 × SE | Lectura |
| :--- | ---: | ---: | ---: | :--- |
| Lerp − Denso | +0.00238 | 0.01068 | 0.02136 | Indistinguible; denso continúa nominalmente delante. |
| Lerp − Aleatorio | -0.01536 | 0.01041 | 0.02082 | Tendencia favorable a Lerp, aún insuficiente. |
| FWHT − Aleatorio | +0.00125 | 0.01381 | 0.02763 | Sin evidencia de efecto específico FWHT. |
| DCT − Aleatorio | +0.00423 | 0.00694 | 0.01388 | Sin evidencia de efecto específico DCT. |

Lerp cuesta 37.9% más wall-clock y 46.0% más tiempo de evaluación de función que denso. Por tanto, incluso si el empate de calidad se confirmase, la implementación actual no justifica una afirmación de eficiencia práctica.

## 9. Auditoría posterior y amenazas a la validez: extensión v330b

1. **La señal Lerp no identifica todavía la causa.** Lerp combina dos ramas, mientras `random_orthogonal`, FWHT y DCT son ramas únicas. Su tendencia frente al control aleatorio puede deberse a la fusión/router y no a la geometría FWHT+DCT. La ablación prioritaria es `lerp_random_a_random_b`, con dos bases ortogonales aleatorias fijas e independientes, mismo router y mismo presupuesto; `lerp_fwht_random` aislaría aún mejor la contribución de cada familia.
2. **Convergencia aún no demostrada como meseta.** Los checkpoints seleccionados están entre las épocas 26 y 30 para todas las variantes. Eso no invalida la comparación emparejada a 30 épocas, pero impide presentar estos números como rendimiento asintótico. No se recomienda otra extensión antes de resolver la confusión causal de dos ramas frente a bases específicas.
3. **Cinco semillas dan resolución limitada.** El efecto Lerp–aleatorio (-0.01536) es menor que `2 × SE` (0.02082). Debe mantenerse como [RUIDO-SOSPECHA], no como [SEÑAL], hasta aumentar semillas o reproducirse en la ablación causal.
4. **Iso-paramétrico no implica iso-cómputo.** La diferencia máxima sigue siendo de seis parámetros sobre ~59.2k, pero Lerp materializa y aplica dos transformadas. Los tiempos son de esta implementación PyTorch/CPU, no de un kernel FWHT/DCT compilado; no sostienen un claim algorítmico de FLOPs ni de inferencia optimizada.
5. **Alcance externo restringido.** Todo v330/v330b se limita a Tiny Shakespeare a nivel de carácter, `d_model=64` y dos bloques. No permite inferir comportamiento en BPE, contextos largos, escalas grandes ni LLMs.

## 10. Conclusión consolidada

La extensión elimina la explicación simple de que Lerp sólo estaba cerca de denso por falta de entrenamiento: tras 30 épocas sigue prácticamente empatado, pero no lo supera de forma verificable y es más lento. Tampoco demuestra una ventaja específica de FWHT o DCT-II sobre una rotación ortogonal aleatoria. La siguiente prueba informativa es una ablación de dos ramas, no otra prolongación directa de v330.

## 11. Artefactos de la extensión

- Script: `scratch/prototype_v330_spectral_transfer_control.py`
- Plan actualizado: `docs/experiment_plan_v330_spectral_transfer_control.md`
- JSON v330b: `results/raw/v330b_spectral_transfer_extended_30ep_20260810T120346Z.json`

## 12. Reconciliación posterior: ablación causal v331 (2026-08-11)

v331 elimina la principal ambigüedad que permanecía en v330b: compara `lerp_fwht_dct` con `lerp_random_pair`, que conserva las dos ramas, router, combinador y 59,205 parámetros de Lerp. El contraste emparejado es `-0.00137 ± 0.00446` (`2×SE=0.00893`), por lo que no hay ventaja verificable de FWHT+DCT sobre una pareja de bases ortogonales aleatorias.

También falla la explicación alternativa de diversidad de dos bases en este presupuesto: `RandomPair − RandomTied = -0.00562 ± 0.01591` (`2×SE=0.03182`). En consecuencia, la tendencia favorable de Lerp frente a una rama aleatoria en v330/v330b debe reclasificarse como [RUIDO-SOSPECHA], no como evidencia de geometría espectral ni como incentivo para extender directamente la misma línea.

El detalle reproducible de esta reconciliación está en `docs/findings_v331_two_branch_basis_ablation.md` y `results/raw/v331_two_branch_basis_ablation_20260811T113633Z.json`.
