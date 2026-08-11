# Plan de experimento v331 — Ablación causal de dos ramas y bases espectrales

> **Estado:** pre-registrado; aún no ejecutado. **Nivel previsto:** Nivel 1 para filtro de harness y Nivel 2 para la comparación que se documentará. No reutiliza `v231`, que ya identifica el experimento JPEG espectral.

## Reconciliación con v330/v330b

v330b deja una ambigüedad concreta: `lerp_fwht_dct` quedó nominalmente por delante de una rama ortogonal aleatoria única, pero Lerp tiene dos ramas, parámetros de modulación independientes y un combinador. El margen frente a `random_orthogonal` puede provenir de esa topología, de usar dos bases distintas, o de que FWHT+DCT aporten estructura. Por tanto, v331 no busca mejorar la loss; busca identificar cuál de esas tres explicaciones es compatible con los datos.

## Pregunta primaria

**A igual arquitectura de dos ramas, router, combinador, datos, entrenamiento y parámetros, ¿FWHT+DCT-II reducen la test loss frente a dos bases ortogonales aleatorias independientes?**

## Diseño: único cambio experimental

Se reutiliza el backbone de v330b sin cambios: Tiny Shakespeare a nivel de carácter, split temporal 70/15/15, `d_model=64`, 2 bloques de atención causal, 4 cabezas, secuencia 128, batch 16, AdamW, `LR=3e-3`, `weight_decay=0`, clip 1.0, 150 pasos/época, ventanas de validación/test fijas y no solapadas, y checkpoint de menor `valid_loss` con un único test posterior.

La única variable es el **par de matrices ortogonales fijas** de cada FFN de dos ramas. Todas las variantes Lerp usan exactamente la misma clase genérica:

`salida = combine(concat(w0 · branch(T0, x), w1 · branch(T1, x)))`

donde `w=softmax(logits)` es global y aprendible por capa; cada rama conserva sus propios parámetros de fase y amplitud. No se introduce routing token-dependiente ni capacidad nueva.

| Orden de ejecución | Variante | Pares de bases fijas | Papel causal |
| :--- | :--- | :--- | :--- |
| 1 | `lerp_fwht_dct` | FWHT, DCT-II | Candidato estructurado. Debe ejecutarse primero. |
| 2 | `lerp_random_pair` | R0, R1 ortogonales aleatorias e independientes | **Control crítico**: misma topología, diversidad de bases sin estructura espectral elegida. |
| 3 | `lerp_random_tied` | R0, R0 | Control de dos ramas/router sin diversidad de base. |
| 4 | `lerp_fwht_random` | FWHT, R0 | Atribuye una contribución posible a FWHT. |
| 5 | `lerp_dct_random` | DCT-II, R0 | Atribuye una contribución posible a DCT-II. |
| 6 | `dense_ffn` | — | Referencia externa de calidad y coste; no participa en la atribución primaria entre pares. |

`R0` y `R1` se generan por QR de matrices gaussianas de forma determinista a partir de `(training_seed, layer_index, branch_index)`. Así cada una de las cinco réplicas Nivel 2 prueba una realización aleatoria distinta, mientras las comparaciones dentro de semilla comparten las mismas matrices R0/R1 allí donde corresponda. Las matrices permanecen congeladas y se registran sus semillas y huellas SHA-256 en el JSON.

## Igualación y controles

- Las cinco variantes Lerp deben tener **exactamente** el mismo número de parámetros entrenables (59,205 con la configuración actual). Sólo cambian buffers congelados.
- `dense_ffn` queda dentro de seis parámetros (59,207) como en v330b; se reporta como referencia, no como control causal de la hipótesis de bases.
- Todas las variantes de una semilla comparten inicialización entrenable, batches de entrenamiento, ventanas de validación/test y scheduler.
- La comparación primaria es emparejada por `training_seed`: `lerp_fwht_dct − lerp_random_pair`. Loss menor es mejor.

## Ejecución prevista

### Nivel 1 — filtro de harness

- Una semilla (`42`), 10 épocas, 16 batches de validación/test (256 secuencias por partición).
- Confirmar antes de continuar: conteos de parámetros, igualdad de las rutas no aleatorias, semillas/huellas de R0/R1, finite forward/backward y que `lerp_random_tied` usa efectivamente el mismo buffer dos veces.
- El log debe emitir los cinco primeros batches de la época 1 y, después, el resumen detallado de cada época. No se interpreta la loss como comparación concluyente.

### Nivel 2 — decisión de la ablación

- Cinco semillas de entrenamiento: `[10, 20, 30, 42, 100]`; 30 épocas; mismas 1,024 secuencias retenidas de validación y test por semilla.
- Reportar test loss/PPL media, SD y SE entre semillas; y las diferencias emparejadas de todas las comparaciones causales.
- No elegir ni descartar variantes mirando test durante entrenamiento. Cada test se evalúa sólo tras seleccionar por validación.

## Criterios de lectura predefinidos

| Patrón observado en Nivel 2 | Interpretación permitida |
| :--- | :--- |
| `FWHT+DCT − RandomPair ≤ -2×SE` | Señal reproducible compatible con una contribución de la pareja estructurada frente a bases aleatorias de la misma topología; aún limitada a este corpus/escala. |
| `FWHT+DCT` y `RandomPair` no distinguibles, pero `RandomPair` mejora `RandomTied` | La evidencia favorece diversidad de dos bases, no una geometría espectral específica. |
| `FWHT+DCT`, `RandomPair` y `RandomTied` no distinguibles | No hay evidencia de que ni la diversidad de base ni FWHT/DCT expliquen la señal previa. |
| Sólo `FWHT+Random` o sólo `DCT+Random` mejora `RandomPair` | Señal exploratoria del componente correspondiente; requiere réplica, no atribución fuerte. |
| Denso es mejor que todos los pares | Referencia práctica adversa; no invalida por sí sola la pregunta causal entre pares. |

Para cada comparación, se informará `Δ`, SD emparejada, `SE=SD/sqrt(5)` y el umbral `2×SE`. Una diferencia menor en magnitud que `2×SE` se etiquetará [RUIDO-SOSPECHA]. No se declarará un cierre negativo definitivo: faltarían réplica en otro corpus/escala y un barrido de sensibilidad suficiente.

## Logging, JSON y ledger

El script es `scratch/prototype_v331_two_branch_basis_ablation.py`; no modifica `prototype_v330_spectral_transfer_control.py`. Debe cumplir el contrato de `GEMINI.md`:

1. Cada línea con timestamp relativo; cabecera con pregunta, metadatos, configuración completa y arquitectura por capas.
2. Historia no colapsada por época: train/valid loss, LR, normas media/final de gradiente, tokens, tiempos y checkpoint.
3. JSON con configuración, historial por época, resultados por semilla, diferencias emparejadas, semillas y huellas de las bases aleatorias, costes y resumen final.
4. Ledger sólo tras éxito y con `experiment_id=v331_two_branch_basis_ablation`; el resultado quedará como [SEÑAL] o [RUIDO-SOSPECHA], nunca [ANCLA] sin confirmar los criterios anteriores.

Comandos de ejecución:

```powershell
# Filtro de harness: candidato primero, una semilla y diez épocas.
python scratch/prototype_v331_two_branch_basis_ablation.py --mode pilot

# Comparación Nivel 2: 5 semillas, 30 épocas y 1,024 secuencias retenidas por partición.
python scratch/prototype_v331_two_branch_basis_ablation.py --mode level2
```

## Amenazas a la validez

1. **Sólo cinco realizaciones aleatorias de base.** Variar R0/R1 con la semilla evita depender de un único control afortunado, pero no estima con precisión toda la distribución de bases. Si hay señal, repetir con nuevas semillas de base separadas de las de entrenamiento.
2. **Coste de implementación.** Todas las ramas siguen siendo `F.linear` con buffers materializados en CPU. El resultado compara calidad causal, no la eficiencia de kernels FWHT/DCT compilados.
3. **Alcance estrecho.** La conclusión, incluso favorable, queda limitada a Tiny Shakespeare char, 2 bloques y `d_model=64`. La transferencia a BPE o escalas mayores requiere un experimento nuevo.

## Resultado registrado (Nivel 2, 2026-08-11)

La comparación terminó con cinco semillas, 30 épocas y 1,024 secuencias retenidas por partición. El contraste primario `lerp_fwht_dct − lerp_random_pair` fue `-0.00137 ± 0.00446` (SE emparejado; `2×SE=0.00893`): no distingue la pareja estructurada de dos bases aleatorias independientes. `lerp_random_pair − lerp_random_tied` fue `-0.00562 ± 0.01591`, también no distinguible.

`lerp_dct_random` obtuvo la menor media nominal (1.94472), pero su contraste contra RandomPair fue `-0.01031 ± 0.00905`, inferior al umbral de magnitud `2×SE=0.01811`. Se registra como [RUIDO-SOSPECHA]. Dense permanece como referencia práctica: `lerp_fwht_dct − dense_ffn = +0.00097 ± 0.00871` y Lerp FWHT+DCT cuesta 30.1% más wall-clock.

El informe, reconciliación y auditoría completa están en `docs/findings_v331_two_branch_basis_ablation.md`.
