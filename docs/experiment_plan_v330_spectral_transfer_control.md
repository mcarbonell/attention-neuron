# Plan de experimento v330b — Transferencia espectral controlada extendida

## Pregunta

¿Las bases fijas FWHT/DCT-II aportan una ventaja específica en modelado de lenguaje real, o el efecto observado es compatible con cualquier rotación ortogonal fija y con la capacidad/optimización del FFN?

## Diseño

- **Corpus:** Tiny Shakespeare real, caracteres, split temporal 70% train / 15% valid / 15% test, con ventanas retenidas no solapadas.
- **Backbone constante:** embedding + posición seno/coseno + 2 bloques de atención causal QK$^T$ + normalización + head.
- **Única variable:** el FFN de cada bloque.
- **Checkpoint:** mínimo `valid_loss`; test una única vez tras seleccionar el checkpoint.
- **Métrica primaria:** test loss medio por secuencia; se informan también PPL, accuracy, SE por secuencia y SE entre semillas.

| Orden de ejecución | Variante | Función como control |
| :--- | :--- | :--- |
| 1 | `lerp_fwht_dct` | Candidato: mezcla global aprendible entre FWHT y DCT-II. |
| 2 | `dense_ffn` | FFN denso con parámetros emparejados al espectral. |
| 3 | `random_orthogonal` | Control crítico para probar si importa una base concreta. |
| 4 | `fwht` | Base Walsh-Hadamard aislada. |
| 5 | `dct` | Base DCT-II aislada. |

Las variantes espectrales quedan igualadas salvo dos logits del router; el denso se ajusta al número de parámetros del FFN espectral. Con la configuración por defecto de dos bloques, la diferencia total máxima es de seis parámetros por el redondeo inevitable de la anchura densa; el script imprime y guarda las cuentas exactas.

## Ejecución

```powershell
# Filtro de harness: una semilla, diez épocas.
python scratch/prototype_v330_spectral_transfer_control.py --mode pilot

# Extensión Nivel 2: cinco semillas, treinta épocas y 1.024 secuencias en validación y test por configuración.
python scratch/prototype_v330_spectral_transfer_control.py --mode level2
```

El JSON completo se guarda en `results/raw/` con el identificador `v330b_spectral_transfer_extended_30ep` y el script añade una línea `[SEÑAL]` al ledger al finalizar correctamente. No debe afirmarse una mejora hasta inspeccionar el JSON, calcular diferencias entre semillas y comprobar que el candidato también supera al control `random_orthogonal`.

## Criterio interpretativo predefinido

Una diferencia frente al FFN denso no es evidencia de una base espectral específica. Para sostener esa hipótesis, `fwht`, `dct` o `lerp_fwht_dct` deben mejorar tanto al control denso como al control ortogonal aleatorio, con el mismo presupuesto y una diferencia de test loss de al menos $2\times SE$ entre semillas. Si sólo mejora frente a denso, el resultado se interpreta como compatible con la restricción ortogonal/parametrización compacta, no con una geometría FWHT/DCT especial.

## Amenazas anticipadas

1. Tiny Shakespeare es pequeño y de caracteres; un resultado positivo sólo justifica pasar a un corpus BPE real, no afirmaciones sobre LLMs.
2. El costo wall-clock de matrices ortogonales materializadas no representa un kernel FWHT compilado; se reporta como implementación PyTorch actual.
3. Un split temporal evita fuga de ventanas, pero no evalúa contextos mucho más largos ni recall asociativo; MQAR *on-the-fly* permanece como experimento separado.

## Resultado registrado (v330b, 2026-08-10)

La ejecución Nivel 2 extendida terminó con cinco semillas y 30 épocas. `dense_ffn` obtuvo la menor test loss media (1.95392 ± 0.00685); `lerp_fwht_dct` quedó segundo (1.95631 ± 0.00680), sin diferencia emparejada distinguible frente a denso (+0.00238 ± 0.01068). Frente a `random_orthogonal`, Lerp presenta una tendencia favorable (-0.01536 ± 0.01041), pero no alcanza el umbral predefinido de `2 × SE`.

La siguiente hipótesis a probar no es más duración: es si la ventaja nominal procede de las bases FWHT/DCT o de tener dos ramas con un router. Se prioriza una ablación `lerp_random_a_random_b` con dos bases aleatorias independientes, presupuesto y router igualados.

El detalle completo, auditoría y artefacto se registran en `docs/findings_v330_spectral_transfer_control.md`.
