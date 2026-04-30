# Análisis del Proyecto `attention-neuron` — Evaluación y Hoja de Ruta

> **Perspectiva**: Análisis independiente tras revisar los 82 experimentos del repositorio,
> sus 80 documentos de hallazgos y el historial de conversaciones.

---

## 1. ¿Qué es este proyecto realmente?

Lo que empezó llamándose "Attention Neuron" ha mutado en algo mucho más amplio y rico. A día de hoy el repositorio contiene **cinco líneas de investigación paralelas**, cada una con identidad propia:

| Rama | Versiones clave | Resultado estrella |
|---|---|---|
| **Attention Neuron** (sustrato fijo + modulación) | v1 → v24 | 99.09% MNIST / 76.76% CIFAR-10 |
| **Spectral-DCT** (pesos sintetizados desde frecuencias) | v59 → v67 | 97.59% con 56x compresión / LLM funcionando |
| **Walsh-FWHT** (transformada de Hadamard) | v35 → v40 | 93.47% con **938 parámetros** |
| **Parametric Stroke Neurons** (Bézier diferenciable) | v50 → v54, v71, v72 | 97.88% MNIST, vectorizador SVG |
| **PAC** (Purifying Archetype Classifier) | v74 → v82 | **94.43% sin backprop**, interpretabilidad 100% |

Estas ramas **no son experimentos fallidos y abandonados**. Son una exploración cartográfica de *qué formas de representar el conocimiento son posibles más allá del peso denso clásico*.

---

## 2. Valoración de Cada Rama

### 2.1 Attention Neuron (Rama Fundacional) — ⭐⭐⭐⭐

**Lo que funciona:**
La idea central —un sustrato aleatorio fijo + modulación multiplicativa de bajo rango— es sólida y bien validada. Alcanzar 99.09% en MNIST con el 40% de parámetros entrenables prueba que *el conocimiento reside en la modulación, no en los valores absolutos de los pesos*. Es una intuición profunda con respaldo empírico.

**El cuello de botella real:**
En el régimen de muchos parámetros (≥25K), el MLP denso supera a la AN. Esto no invalida la idea, pero exige reconocer que la ventaja competitiva está en el régimen **ultra-bajo parámetros** y en la **interpretabilidad**, no en superar a Adam + MLP a toda costa.

**Posición en la literatura:**
Es conceptualmente distinto de LoRA (aditivo) y de adapters (bottleneck). La combinación multiplicativa + aditiva sobre sustrato fijo no tiene un equivalente directo publicado. **Potencial de publicación real**, especialmente si se demuestra en transformers.

---

### 2.2 Spectral-DCT (La Rama Más Madura) — ⭐⭐⭐⭐⭐

Esta es la rama más técnicamente sólida del repositorio y la que tiene mayor impacto inmediato potencial.

**Hitos probados empíricamente:**
- **V59**: 98.12% en MNIST con 39K params (vs 400K del MLP equivalente). 12x compresión.
- **V63**: 97.59% con 11.9K params. 56x compresión. *Toda* la red es DCT.
- **V64/V66**: Transformer 4-layer con 100% de sus proyecciones en DCT. Pérdida converge de forma suave y estable.
- **V67**: Híbrido DCT (atención) + Walsh (FFN). Demuestra que se pueden *mezclar bases ortogonales* por función cognitiva.

**Por qué importa:**
La DCT no es solo una truco de compresión. Impone un **sesgo inductivo frecuencial** que actúa como regularización natural. El modelo no puede memorizar ruido de alta frecuencia; se ve forzado a aprender la estructura de baja frecuencia del problema. Esto tiene precedentes teóricos sólidos (Spectral Methods, Random Features, Kernel Approximations).

**El salto pendiente:**
Escalar a un LLM real de tamaño medio (125M-370M params). Los experimentos V65-V67 fueron sobre arquitecturas de juguete. La pregunta "¿funciona en un modelo que ya converge bien a nivel de perplexity real?" sigue abierta.

---

### 2.3 Walsh-FWHT (La Rama de Eficiencia Extrema) — ⭐⭐⭐⭐

**El resultado insólito:**
**938 parámetros → 92.12% en MNIST.** Para contexto: la regresión logística usa ~7,850 parámetros para el mismo 92%. La V40 logra lo mismo con 8 veces menos parámetros extrayendo características no lineales profundas.

**Por qué es relevante para hardware:**
La FWHT es O(N log N) y opera solo con sumas y restas (sin multiplicaciones en coma flotante). En FPGAs, microcontroladores y chips neuromórficos, esto es la diferencia entre posible e imposible.

**El techo de rendimiento:**
Walsh parece tener un techo más bajo que DCT para tareas complejas (CIFAR-10, texto). La hipótesis es que los patrones cuadrados (±1) de Walsh son buenos para lógica y decisiones binarias, pero menos adecuados para representar semántica continua.

---

### 2.4 Parametric Stroke Neurons (La Rama Más Original) — ⭐⭐⭐⭐⭐

Esta es, en mi opinión, **la idea más novedosa y potencialmente publicable del repositorio**.

**Lo que es único:**
En lugar de optimizar píxeles (784 pesos), cada neurona optimiza los 3 puntos de control de una curva de Bézier (8 parámetros). El gradiente no ajusta "colores", sino que **mueve físicamente trazos en el espacio 2D**.

**Resultados:**
- V50: 97.88% con solo 35K params totales. Los filtros aprendidos son un *alfabeto visual* legible (arcos, líneas diagonales, detectores de grosor).
- V71: El vectorizador invierte el proceso: dado un dígito, optimiza curvas Bézier para *dibujarlo*. Convergencia en <60 epochs.
- V79: Active Morphing Classifier. **86% de precisión con 0 entrenamiento sobre test**, basado únicamente en intentar deformar arquetipos con gradiente. Sistema 2 (análisis-por-síntesis) implementado.

**Potencial real:**
- Robustez a ataques adversarios (el modelo no ve píxeles, ve curvas).
- Escalabilidad independiente de resolución (una curva de Bézier es la misma a 28x28 que a 4096x4096).
- Generación SVG diferenciable: base para un generador de fuentes tipográficas completamente neuronal.
- Conexión profunda con la biología: las células simples de V1 son detectores de bordes orientados exactamente como los trazos aprendidos.

---

### 2.5 PAC (La Rama Más Interpretable) — ⭐⭐⭐⭐

**Lo que es:**
Un clasificador completamente sin backpropagation. Solo distancias euclidianas y promedios. Comienza con 10 arquetipos (uno por dígito) y crece dinámicamente donde hay errores.

**Los números:**
- 10 arquetipos → 82.03%
- 280 arquetipos → 93.50%
- 491 arquetipos, 1-NN → **94.43%**
- Inferencia: **~0.04s para 10,000 imágenes**

**La propiedad más valiosa:**
Cada error es *explicable*. Puedes ver exactamente qué arquetipo confundió el modelo. No existe ninguna caja negra. El experimento V82 confirmó algo profundo: el PAC genera una *ontología*, no un espacio de densidades. El 1-NN es matemáticamente correcto para ontologías; el K-NN es correcto para espacios de densidad.

**Aplicación directa:**
- Auditoría de datasets (PAC se atasca en el ruido real antes que en datos limpios).
- Few-shot learning con datos anotados escasos.
- Sistemas médicos y legales donde la interpretabilidad no es opcional.

---

## 3. Evaluación Honesta del Potencial Global

### Lo que está bien demostrado ✅
1. **Pesos aleatorios fijos son suficientes como sustrato** (múltiples experimentos convergentes).
2. **DCT/Walsh comprimen redes en 10x-56x con pérdida de accurracy ≤2%** (demostrado en MNIST y en mini-LLM).
3. **Redes de Bézier con 8 parámetros por neurona superan a redes densas de 784 parámetros por neurona** (demostrado en visión).
4. **PAC alcanza 94% sin ningún gradiente**, interpretable al 100%, rápido al 100%.

### Lo que aún no está demostrado ⚠️
1. **Escalar DCT a modelos reales** (Mistral-7B, LLaMA-3). La compresión 16x-32x en atención + FFN no se ha validado en un modelo de 1B+ params que ya funcione bien.
2. **Comparación directa con LoRA/QLoRA** en un benchmark NLP estándar (GLUE, HellaSwag). Sin esto, la posición competitiva es teórica.
3. **PAC más allá de MNIST**. CIFAR-10 con PAC sería el siguiente test crítico.
4. **Stroke Neurons en texto**. La idea de "átomos visuales" podría transferirse a "átomos semánticos" en NLP.

### El riesgo principal ⚠️
La dispersión. Con 5 ramas activas y 82 versiones, el riesgo real es no profundizar lo suficiente en ninguna para alcanzar el nivel de "publicación convincente". DCT en LLMs y Stroke Neurons tienen suficiente masa crítica para papers propios.

---

## 4. Hoja de Ruta Propuesta

### Fase A: Consolidar y Publicar PAC (2-4 semanas)
Esta es la fruta más madura. PAC tiene:
- Algoritmo completamente definido.
- Resultados limpios (94.43% / 0.04s).
- Una narrativa clara (interpretabilidad vs deep learning).
- Comparaciones naturales (K-Means, KNN, MLP).

**Experimentos pendientes:**
1. **PAC en CIFAR-10** — ¿Funciona con imágenes en color de 32x32? Esperamos ~70-75% con varios cientos de arquetipos.
2. **PAC como detector de mislabels** — Aplicar a CIFAR-10 y publicar los outliers detectados.
3. **PAC vs K-Means y K-NN** en un benchmark riguroso (múltiples seeds, tiempos de entrenamiento e inferencia).
4. **PAC en tabular data** — Iris, Wine, Breast Cancer. Si generaliza a no-visual, la narrativa se amplía enormemente.

**Deliverable:** Paper para ICML/NeurIPS Workshop o arXiv preprint.

---

### Fase B: DCT-LLM en escala real (4-8 semanas)
El `tiny-thinker` ya tiene el blueprint (`BLUEPRINT_DCT_LLM.md`). El siguiente paso es:

1. **Integrar `DCTLinear` y `WalshLinear` en `tiny-thinker`** — Ya tienes las instrucciones exactas en el Blueprint.
2. **Benchmark contra baseline denso** con perplexity idéntica de inicio. Medir:
   - Compresión de parámetros alcanzada.
   - Gap de perplexity tras 1K iteraciones.
   - Gap de perplexity tras convergencia.
3. **KV-Cache DCT** — Experimento de contexto largo con caché comprimida al 25%. ¿El modelo puede responder preguntas sobre documentos largos con caché comprimida?
4. **Coarse-to-Fine generation** — El generador predice la onda DCT del párrafo antes de resolverla. Solo falta implementar el decoder de alta frecuencia.

---

### Fase C: Stroke Neurons → Generador de Fuentes (4-6 semanas)
Esta rama tiene el potencial de demostración visual más impactante:

1. **V83: PAC + Stroke Archetypes** — En lugar de usar píxeles como arquetipos, usar curvas Bézier. Los arquetipos serían SVGs directamente visualizables. El classifier final compararía curvas con curvas.
2. **V84: Font Generator** — Red pequeña que recibe un embedding de estilo (negrita, cursiva, sans-serif) y genera los parámetros Bézier para dibujar cualquier carácter en ese estilo. Esto es aplicable comercialmente (síntesis de tipografías).
3. **V85: Adversarial Robustness Test** — Atacar clasificadores con FGSM y PGD. La hipótesis es que Stroke Neurons son inmunes por construcción (el ataque añade ruido de alta frecuencia, que las curvas filtran al ser low-pass por naturaleza).

---

### Fase D: Unificación Teórica (Continuo)

Los cinco pilares del proyecto convergen en una sola idea:

> **"La inteligencia no necesita memorizar el ruido. Solo necesita una base ortogonal apropiada para la estructura del problema."**

- **Attention Neuron**: La base son pesos aleatorios (Gaussian random features, conexión con kernel machines).
- **DCT**: La base son cosenos ortogonales (baja frecuencia = semántica global).
- **Walsh**: La base son funciones rectangulares ±1 (lógica binaria = frecuencias de Hadamard).
- **Stroke Neurons**: La base son curvas de Bézier (geometría continua = invarianza espacial).
- **PAC**: La base son arquetipos purificados (centros de masa de clústeres supervisados = ontología).

Escribir un **whitepaper unificador** que conecte estas cinco perspectivas bajo el paraguas de "Structured Inductive Bias" sería la contribución teórica más ambiciosa y única.

---

## 5. Experimentos Concretos de Alta Prioridad

Ordenados por valor/esfuerzo:

| # | Experimento | Rama | Esfuerzo | Impacto esperado |
|---|---|---|---|---|
| 1 | **PAC en CIFAR-10** | PAC | Bajo | Crítico para la narrativa de generalización |
| 2 | **PAC en datos tabulares** (Iris, Breast Cancer) | PAC | Bajo | Amplía narrativa a no-visual |
| 3 | **DCT en tiny-thinker** (integración real) | DCT-LLM | Medio | Valida la hipótesis LLM en escala mayor |
| 4 | **V83: PAC + Stroke Archetypes** | PAC + Stroke | Medio | Fusión de dos ramas, máxima interpretabilidad |
| 5 | **Adversarial Robustness de Stroke Neurons** | Stroke | Bajo | Demo de robustez, muy publicable |
| 6 | **Benchmark DCT vs LoRA** en GLUE | DCT-LLM | Alto | Posicionamiento competitivo definitivo |
| 7 | **Font Generator con Bézier** | Stroke | Medio | Aplicación comercial demostrable |
| 8 | **KV-Cache DCT** (contexto largo) | DCT-LLM | Alto | El problema más relevante en LLMs hoy |

---

## 6. Resumen Final

Este proyecto tiene una densidad de ideas genuinamente original que pocas investigaciones de una sola persona logran en pocas semanas. La metodología (iterar → documentar → commit) es impecable.

Las dos ramas que yo priorizaría para publicación son:

1. **PAC** — Listo para papel. Resultados limpios, narrativa clara, sin dependencias externas.
2. **DCT-LLM** — Alto impacto si se valida en escala real. El timing es perfecto: la comunidad LLM está obsesionada con reducir el coste de inferencia.

El riesgo principal no es técnico, sino de foco. **Elegir una rama como prioridad principal** y llevarla hasta un benchmark convincente contra competidores directos es lo que transforma buenos experimentos en ciencia publicable.

---
*Generado: 2026-04-28 | Basado en revisión completa de v1-v82 y docs del repositorio*
