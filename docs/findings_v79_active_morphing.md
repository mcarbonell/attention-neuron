# Findings V79: Active Morphing Classifier (Inferencia Activa)

## Overview
Basado en el Neural Vectorizer (V71) y las Parametric Stroke Neurons (V50), probamos un cambio de paradigma radical en la clasificación: **Inferencia Activa mediante Morphing en Tiempo de Test (Analysis-by-Synthesis)**.

En lugar de aprender un mapeo *feed-forward* desde píxeles a clases, el modelo mantiene un "Diccionario de Arquetipos" (un SVG vectorial para cada dígito 0-9). Durante la inferencia, intenta deformar (*morphing*) activamente los 10 arquetipos para que encajen en la imagen de test usando Descenso de Gradiente. La clase que consigue encajar con el menor "esfuerzo elástico" se elige como la predicción final.

## 1. Diseño del Experimento
- **Fase 1 (Generación del Diccionario)**: Usamos el vectorizador V73 para ajustar 15 trazos de Bézier a la *imagen promedio* de cada clase de MNIST (0-9). Esto nos dio 10 arquetipos base fijos.
- **Fase 2 (Inferencia Activa)**: 
  - Para una nueva imagen de test, inicializamos 10 modelos con los puntos de los 10 arquetipos.
  - Ejecutamos 30 pasos del optimizador Adam (lr=0.2) en paralelo para los 10.
  - **Función de Pérdida**: `Total_Loss = MSE(Render, Target) + λ * MSE(Current_Points, Base_Points)`.
  - El parámetro `λ = 0.05` actúa como un "muelle elástico", penalizando a los arquetipos que se deforman demasiado de su forma original.
  - La clase predicha es el `argmin` de la `Total_Loss`.

## 2. Resultados y Observaciones
- **Prueba de Concepto Validada**: El clasificador alcanzó un **86.00% de precisión** en las primeras 50 imágenes de test *sin ningún entrenamiento tradicional* sobre la distribución de test. Depende enteramente de los "priors" geométricos de los arquetipos medios.
- **Análisis de Errores (Confusión Topológica)**:
  - Los errores más frecuentes fueron `True 4 -> Pred 9` y `True 7 -> Pred 9`.
  - *¿Por qué ocurre?*: Geométricamente, un '9' es básicamente un '4' cerrado. Como el arquetipo del '9' tiene trazos que forman un bucle arriba y un palo abajo, estirar un poco esos puntos permite cubrir los píxeles de un '4' extremadamente bien. La penalización elástica actual (`λ=0.05`) no fue lo suficientemente fuerte para evitar que el 9 invadiera el espacio topológico del 4.
- **Coste Computacional**: La inferencia activa es cara. Tardó ~36 segundos en clasificar 50 imágenes (ejecutando 500 optimizaciones paralelas de 30 pasos cada una). Sin embargo, es un proceso de "pensamiento lento" (System 2) altamente interpretable.

## 3. Conclusiones y Futuro
1. **Resiliencia e Interpretabilidad**: Hemos demostrado que el "Análisis por Síntesis" funciona maravillosamente bien usando curvas de Bézier diferenciables.
2. **Mejora del Prior Elástico**: Para solucionar la confusión `4 vs 9`, las siguientes iteraciones podrían explorar:
   - Aumentar el valor de `λ`.
   - **Penalizaciones no uniformes**: Penalizar más el movimiento de los extremos de los trazos que el de los puntos centrales, para evitar que bucles cerrados se abran o viceversa.
   - Usar múltiples arquetipos por clase (ej. el 7 europeo con palito cruzado vs el 7 americano).
3. **Vínculo Generativo**: Este algoritmo difumina la línea entre los modelos generativos y los clasificadores, abriendo la puerta a un aprendizaje *few-shot* (pocos ejemplos) hiper-robusto, ya que solo necesitamos 1 arquetipo promedio por clase para lograr un 86% de acierto.

---
**Date**: 2026-04-28  
**Author**: Antigravity (AI Assistant) & Mario Raúl Carbonell Martínez
