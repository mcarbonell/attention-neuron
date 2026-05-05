# Findings V163f: Stress Test de Memoria Holográfica (Saturación)

## Objetivo
Determinar el punto de ruptura de la memoria holográfica basada en desplazamientos circulares (`Roll`) sin mecanismos de atención.

## Resultados
Se probó la recuperación de un token "aguja" en la posición 0 rodeado de $L$ tokens de ruido aleatorio.

| Dimensión (D) | Contexto (L) | Accuracy | SNR (Relación Señal-Ruido) |
| :--- | :--- | :--- | :--- |
| 512 | 128 | 5.0% | 2.10 |
| 1024 | 128 | 30.0% | 2.78 |
| 2048 | 128 | 55.0% | 3.61 |
| 2048 | 512 | 0.0% | 1.82 |

## Conclusiones
- **Colapso Rápido**: Sin filtrado, la memoria holográfica se satura extremadamente rápido. La acumulación uniforme de vectores de ruido eleva la varianza del holograma hasta que la señal original es indistinguible.
- **Escalado Insuficiente**: Doblar la dimensión ayuda, pero no resuelve el problema de fondo del ruido acumulado.
- **Necesidad de Saliencia**: Se hace evidente que un LLM no puede confiar en una suma simple para recordar contextos largos.
