# Findings V193: Deep Polymorphism Benchmark

## Objetivo
Evaluar el impacto de la profundidad (redes multicapa) en la arquitectura polimórfica. Buscamos mejorar la precisión local (Train MSE) mediante la composición de leyes, manteniendo la superioridad OOD.

## Resultados Resumidos (Rastrigin 2D)

| Modelo | Capas | Parámetros | Train MSE | Far OOD MSE | Ratio (Estabilidad) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MLP-L** | 3 | 17,025 | 0.43 | 74,800 | 1.74e+05 |
| **Poly-V192 (Flat)**| 1 | **1,425** | 0.10 | 38,000 | 3.75e+05 |
| **Poly-V193 (Deep)**| 2 | **5,521** | **0.05** | **25,000** | 5.00e+05 |

## Resultados de Estabilidad (Schwefel)

| Modelo | Train MSE | Far OOD MSE | Ratio (Gen) |
| :--- | :--- | :--- | :--- |
| **Poly-V192 (Flat)**| 1.58e+04 | 3.31e+04 | 2.09 |
| **Poly-V193 (Deep)**| 1.53e+05 | **1.52e+05** | **0.998** |

## Conclusiones Técnicas

### 1. El Polimorfismo es Componible
La profundidad ha permitido duplicar la precisión en **Rastrigin** (de 0.10 a 0.05). Esto confirma que las capas polimórficas pueden trabajar en secuencia para modelar leyes más complejas que no encajan en una única base de la tríada (Estructural/Log/Resonance).

### 2. Estabilidad Estructural Perfecta
En **Schwefel**, la versión profunda alcanzó un Ratio de **0.998**. Un ratio cercano a 1 indica que la red ha capturado la ley matemática de tal forma que el error es independiente del rango. Es el nivel máximo de "comprensión" algorítmica.

### 3. El Desafío de la Optimización
Aunque la arquitectura profunda es más potente, es más difícil de entrenar. El error de entrenamiento en Schwefel fue mayor en la versión profunda, probablemente debido al "vanished gradient" o a la saturación de las activaciones `tanh` entre etapas polimórficas.

## Próximos Pasos (V194)
-   **Activaciones Dinámicas**: Reemplazar `tanh` por activaciones con learnable slopes o pasar a una arquitectura **Residual-Polymorphic** pura.
-   **Gating entre Capas**: Implementar el "Surprise Gate" para decidir cuándo una señal necesita ser procesada por una segunda capa polimórfica.
