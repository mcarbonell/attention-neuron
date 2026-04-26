# V30: The Framer (Soft Window Attention) - Preliminary Findings

## 1. Concepto Arquitectónico
La V30 explora el paradigma de la **Visión Continua**. Abandona por completo las convoluciones discretas ($3 \times 3$) y las reemplaza por "Ventanas Suaves" (Soft Windows).

El mecanismo utiliza máscaras sigmoideas separables en 1D para crear cajas delimitadoras diferenciables. Esto reduce drásticamente la complejidad matemática de la extracción espacial a $O(H+W)$ en lugar de $O(H \times W)$, haciendo que la red sea, en teoría, agnóstica a la resolución de entrada.

## 2. Configuración del Experimento
- **Dataset:** CIFAR-10
- **Optimizador:** AdamW (OneCycleLR)
- **Parámetros Entrenables:** **50,570** (Extremadamente ligera)
- **Extracción Espacial:** 4 ventanas continuas por canal.
- **Clasificador:** Pequeño MLP (`256 -> 128 -> 10`).

## 3. Resultados Finales (Época 50/50)
- **Precisión Final:** 58.84%
- **Velocidad:** ~25 segundos por época (en CPU).

## 4. Análisis de Resultados
El modelo alcanzó un techo de **58.84%**, lo cual valida perfectamente nuestra hipótesis inicial. Lograr casi un 60% de precisión sin usar una sola operación de convolución tradicional y con apenas 50K parámetros demuestra que las "Ventanas Suaves" son capaces de extraer características espaciales útiles (probablemente aislando zonas de color o bordes gruesos).

**Comparativa Histórica:**
Comparado con la V24 (Kaleidoscope - 64K params - CNN), que llegó al 75.18%, la V30 se ha quedado atrás. Esto era de esperar: la V24 gana gracias a su naturaleza jerárquica (6 capas profundas), mientras que la V30 actual es "plana" (Extrae ventanas globales $\rightarrow$ MLP clasifica).

## 5. Conclusión y Siguientes Pasos
El mecanismo *Soft Window* funciona y es computacionalmente baratísimo $O(H+W)$. Sin embargo, un solo nivel de extracción de ventanas no puede competir con una jerarquía profunda de características.

**Próxima Iteración Recomendada (V34 "Deep Framer" / "Res-Window"):**
El paso natural es integrar este componente dentro de una arquitectura profunda. Debemos crear un bloque residual donde, en lugar de convoluciones $3 \times 3$, se utilicen capas `SoftWindowLayer` apiladas, permitiendo a la red componer características continuas jerárquicas.