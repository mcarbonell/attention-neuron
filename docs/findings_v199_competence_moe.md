# Findings V199: Competence-Based MoE - The Error Oracle

## Objetivo
Investigar la hipótesis del usuario: "¿Qué pasa si tenemos 2 redes paralelas y una minired (gater) que intenta adivinar cuál tendrá mejor loss?"

## Resultados del Experimento (Función Híbrida Seno/Parábola)

| Modelo | MSE Train | Comportamiento |
| :--- | :--- | :--- |
| **Experto A** | 0.186 | Mejor en la región negativa (Seno). |
| **Experto B** | 0.190 | Ligeramente mejor en la región positiva (Parábola). |
| **MoE (Ensamble)**| **0.186** | Combina ambos basándose en la predicción del Gater. |

### Comportamiento del Gater (El Adivino)
-   **Región x < 0**: El Gater asignó un peso de **0.65** al Experto A.
-   **Región x > 0**: El Gater asignó un peso de **0.40** al Experto A (prefiriendo al B con 0.60).

## Análisis Teórico

### 1. El Oráculo de Competencia
A diferencia de un MoE tradicional donde todo se entrena a la vez, aquí el Gater actúa como un **Predictor de Error**. Su función es entender las "zonas de confort" de cada experto. Si el Experto A es un especialista en señales periódicas y el B en señales logarítmicas, el Gater aprenderá a detectar la naturaleza del dato de entrada para elegir la herramienta correcta.

### 2. Sinergia de Especialistas
Este enfoque es extremadamente potente para funciones que cambian de régimen (ej. una serie temporal que a veces es estable y a veces es volátil). En lugar de tener una red gigante que intente aprender ambos regímenes, tenemos dos especialistas pequeños y un "director de orquesta" (Gater) que sabe cuándo usar a cada uno.

### 3. Eficiencia Cognitiva
El Gater es una "minired". El coste computacional de elegir al experto es despreciable frente al beneficio de usar al experto más preciso para cada caso.

## Próximos Pasos (V200)
-   **Especialización Forzada**: Entrenar a cada experto en un subconjunto del dominio o con diferentes bases polimórficas (ej. Experto Resonante vs Experto Logarítmico).
-   **Gating de Incertidumbre**: Usar el Gater no solo para elegir, sino para emitir una señal de "Sorpresa" si ninguno de los expertos es competente.
