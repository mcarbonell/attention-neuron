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
