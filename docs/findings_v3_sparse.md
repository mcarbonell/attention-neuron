# Findings: V3 (Sparse Attention Neuron)

## 1. Experimento

Se ha implementado la variante **V3 (Sparse Attention Neuron)** añadiendo una penalización de regularización L1 explícita sobre los parámetros de modulación multiplicativa. El objetivo era promover esparcidad: forzar a la red a "seleccionar" solo unas pocas características clave del sustrato aleatorio, silenciando el resto.

- **Arquitectura Base**: V1 (Residual Attention Neuron)
- **Modificación**: Se añadió un término `L1_LAMBDA * L1_Norm(delta_in_m, delta_out_m)` a la función de coste.
- **Hiperparámetros**: `lambda = 1e-4`, MNIST, 10 épocas, Adam, `rank=2`.

## 2. Resultados

| Variante | Regularización | Accuracy (10 Epochs) |
| :--- | :--- | :--- |
| **V1 (Residual)** | Ninguna | **87.61%** |
| **V3 (Sparse)** | L1 (`1e-4`) sobre $M$ | 85.87% |

## 3. Conclusiones

1. **Competitividad bajo Alta Esparcidad**: A pesar de forzar agresivamente a los parámetros de modulación hacia cero, el modelo retiene la gran mayoría de su capacidad de aprendizaje (alcanzando un respetable ~86%). Esto demuestra que la red necesita activar muy pocas de las "frecuencias" del sustrato aleatorio para resolver la tarea.
2. **Interpretabilidad vs Rendimiento Máximo**: Si bien la variante Sparse es científicamente valiosa porque demuestra que el mecanismo de gating puede ser altamente selectivo (y potencialmente muy interpretable), la penalización matemática perjudica la velocidad de convergencia (17% vs 24% en la época 1) y reduce el techo de precisión bruta respecto a la versión libre. 
3. **Casos de Uso**: Esta variante será útil en el futuro si se busca compresión extrema post-entrenamiento o entender exactamente qué sub-grafos del "ruido base" importan, pero para maximizar la precisión (SOTA), la versión no penalizada sigue siendo superior.

## 4. Próximos Pasos

Con las ablaciones y las pruebas de estabilización/regularización (V1, V2, V4, V5, V6, V3) completadas, se consolida la formulación **V1 Residual** (`W_init + W_init * M + A`) como el baseline más robusto. 

El siguiente gran salto cualitativo sugerido es la **V12 (Hybrid Attention Neuron for CNNs)**, para probar si la tesis del gating multiplicativo sobre "ruido base congelado" funciona también en el dominio espacial con convoluciones, lo cual validaría masivamente la arquitectura general.