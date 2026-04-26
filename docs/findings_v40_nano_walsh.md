# V40: The Nano-Walsh Net - Preliminary Findings

## 1. Concepto Arquitectónico
La V40 explora el límite absoluto de la compresión paramétrica ("Zero-Weight Architecture"). Respondiendo a la hipótesis de si es posible clasificar MNIST con menos de 1024 parámetros, la red abandona por completo las grandes capas densas finales.

**Mecánica "Ultra-Low-Cost":**
- **Tres capas de filtrado Walsh:** La imagen pasa por 3 transformadas FWHT consecutivas. En cada capa, se modulan 128 bandas de frecuencia (256 parámetros por capa).
- **Destrucción Espacial Controlada:** En lugar de desplegar los 1024 píxeles filtrados hacia una capa lineal pesada, se aplica un `Average Pooling` extremo (8x8) que reduce la imagen espacial a un minúsculo mapa de $4 \times 4$ (16 valores).
- **Clasificador Enano:** La capa final solo necesita conectar 16 entradas a 10 salidas (170 parámetros).

## 2. Configuración del Experimento
- **Dataset:** MNIST (Padded a $32 \times 32$)
- **Optimizador:** AdamW (OneCycleLR)
- **Parámetros Totales:** **938** (¡Menos del 1% del tamaño de un MLP básico!).
- **Hardware:** CPU.

## 3. Resultados Finales (Época 15/15)
- **Precisión Final:** 92.12%
- **Mejor Precisión:** 92.12% (Época 15)
- **Tiempo por Época:** ~43 segundos.
- **Tiempo Total de Entrenamiento:** Menos de 11 minutos.

## 4. Análisis del Hito (El Triunfo del Minimalismo)
Aunque la precisión no alcanzó el 99%, lograr un **92.12% con solo 938 parámetros** es una demostración empírica asombrosa de compresión del conocimiento.

Para ponerlo en contexto:
- Un clasificador lineal simple (Logistic Regression) de 784 píxeles a 10 clases requiere **7,850 parámetros** y apenas roza el 92%.
- La V40, con **8 veces menos parámetros**, logra igualar ese rendimiento extrayendo características profundas no lineales a través de las frecuencias de Walsh.

**El cuello de botella:**
La red es tan pequeña que sufre de "sub-ajuste" (underfitting) estructural. Comprimir toda la riqueza semántica de un dígito en solo 16 valores espaciales (el pooling $4 \times 4$) y luego usar solo 170 pesos para la decisión final ahoga la capacidad expresiva de la red. Sin embargo, demuestra que el filtrado de Walsh en sí mismo es una base de representación tan rica que puede "salvar" una arquitectura moribunda.

**Conclusión:**
La V40 establece el récord de eficiencia (Accuracy per Parameter) del repositorio. Demuestra que la "IA de Resonancia" puede integrarse en hardware de memoria ultrabaja (microcontroladores, sensores edge) logrando tareas de visión complejas en tiempo real y con huella de memoria casi nula.

## 5. Exploración de la Frontera: V40b (Micro-Walsh)
Para determinar si el límite de rendimiento era puramente paramétrico, se creó la V40b duplicando las bandas de frecuencia (de 128 a 256 por capa), manteniendo la arquitectura intacta.

**Configuración V40b:**
- **Parámetros Totales:** **1,706**
- **Hardware:** CPU.

**Resultados Finales (Época 15/15):**
- **Precisión Final:** 93.47%
- **Best Acc:** 93.50% (Época 14)
- **Tiempo por Época:** ~43 segundos.
- **Tiempo Total:** 654.9s (~11 minutos).

**Análisis Final (V40 vs V40b):**
Añadir apenas 768 parámetros más (pasando de 938 a 1706) ha permitido a la red romper el "cuello de botella de la ecualización" y saltar del 92.12% al **93.50%** de precisión. 

Este resultado confirma que el agrupamiento excesivo de frecuencias (Banded Attention) es el factor limitante en arquitecturas ultra-ligeras. Al duplicar el número de bandas (de 128 a 256), la red ha ganado la resolución frecuencial necesaria para distinguir las sutilezas de los dígitos sin sacrificar la compresión masiva del modelo. 

**Conclusión Global de la Rama "Nano-Walsh":**
Hemos demostrado que una red de menos de 2000 parámetros puede alcanzar >93% de precisión en MNIST utilizando exclusivamente modulación en el dominio de Walsh y un clasificador minúsculo. Si esta operación matemática pura ($O(N \log N)$) se implementase en hardware de bajo nivel (Custom CUDA/Triton kernels o FPGAs), la IA pasaría de ser un problema de supercomputación a un problema de micro-electrónica ubicua.