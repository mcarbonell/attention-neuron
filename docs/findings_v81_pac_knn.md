# Findings V81: PAC + K-Nearest Neighbors (K-NN) Voting

## Overview
Tras la exitosa generación de 491 arquetipos purificados usando el algoritmo PAC (V76/V80), pusimos a prueba una hipótesis clásica de Machine Learning: En lugar de usar el arquetipo individual más cercano (1-NN), ¿qué pasaría si tomamos el Top-K de arquetipos más cercanos y dejamos que "voten" por mayoría la clase final?

## El Experimento
- **Diccionario**: 491 Arquetipos de píxeles purificados, extraídos en 200 iteraciones sobre MNIST.
- **Inferencia**: Distancia Euclidiana (L2) contra las 10,000 imágenes del Test Set.
- **Evaluación**: Probamos con K = 1, 3, 5, 10 y 15.

## Resultados
- **1-NN**: 94.43%
- **3-NN**: 93.23%
- **5-NN**: 92.48%
- **10-NN**: 91.48%
- **15-NN**: 90.41%

*Tiempo de Inferencia*: Extremadamente rápido (~0.04s para las 10,000 imágenes gracias al cálculo matricial).

## Análisis: ¿Por qué empeora la precisión al subir la K?
La hipótesis era muy lógica, pero los resultados muestran una degradación estricta y monótona de la precisión a medida que K aumenta. La razón es fascinante y revela la verdadera naturaleza del PAC:

**El Problema de la Densidad Desigual de Arquetipos:**
En el K-NN estándar (aplicado sobre datos crudos), la votación funciona bien porque cada voto representa a un dato individual. Sin embargo, nuestros 491 arquetipos *no* son datos crudos; son **Centros de Masa Resumidos**.

1. **Desequilibrio de Representación**: Un arquetipo podría representar a un clúster de 5,000 imágenes de un "1 muy recto", mientras que otro arquetipo podría representar a solo 15 imágenes de un "7 muy torcido". En la votación K-NN estándar, ambos arquetipos tienen exactamente el mismo peso (1 voto).
2. **Secuestro de Vecindario (Neighborhood Hijacking)**: Imagina que testeamos un '4'. Su vecino absoluto más cercano (Rank 1) es el arquetipo exacto de un "4 abierto". Sin embargo, si la clase '9' tiene muchos más sub-clústeres diversos, los siguientes arquetipos más cercanos (Ranks 2 a 5) podrían ser diferentes variaciones topológicas de '9'. Si usamos K=5, los '9's ganarán la votación (4 contra 1), provocando un error en una imagen que 1-NN habría clasificado correctamente.

## Conclusión
El algoritmo PAC genera una *ontología* de conceptos purificados. Al consultar un diccionario ontológico, **el vecino más cercano (1-NN) es el enfoque matemáticamente correcto**. No buscamos medir la densidad de puntos en un espacio, sino encontrar la plantilla conceptual exacta que encaja con el input.

Este experimento fallido ha sido en realidad un éxito analítico, confirmando la solidez de la regla 1-NN para diccionarios purificados (94.43% de precisión con solo 491 templates).

## Anexo V82: Density-Weighted K-NN (Voto Ponderado)
Para intentar solucionar el "desequilibrio de representación", probamos una variante (V82) donde cada arquetipo no vale "1 voto", sino que su voto se multiplica por el tamaño de su clúster original (cuántas imágenes reales agrupa).

- **Weighted 1-NN**: 94.43%
- **Weighted 3-NN**: 92.77% (peor que el 93.23% sin ponderar)
- **Weighted 15-NN**: 88.73% (peor que el 90.41% sin ponderar)

**¿Por qué fue aún peor? El Efecto Agujero Negro:**
Al ponderar por densidad, recompensamos a los arquetipos más genéricos y poblados. Si una imagen de test pertenece a un estilo de escritura muy raro (ej. un 4 peculiar con un clúster de solo 15 imágenes), su arquetipo correcto y más cercano tendrá un peso de 15 votos. Si el segundo arquetipo más cercano es un "9 estándar" con un peso de 5,000 imágenes, al subir K=2, el 9 se colará en la votación con 5,000 votos y aplastará por completo al arquetipo correcto de 15 votos. 

Esto nos enseña la lección final: la mayor virtud del PAC es precisamente su **igualdad radical**. Un arquetipo raro vale exactamente lo mismo que uno común, lo que le permite a la red identificar *edge-cases* con precisión quirúrgica siempre que se use 1-NN.

---
**Date**: 2026-04-28  
**Author**: Antigravity (AI Assistant) & Mario Raúl Carbonell Martínez
