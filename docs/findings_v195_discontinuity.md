# Findings V195: Discontinuity Branch - Breaking the Modulus Wall

## Objetivo
Implementar una rama de discontinuidad nativa (`SawtoothLayer`) utilizando un **Straight-Through Estimator (STE)** para permitir que la red aprenda leyes con saltos abruptos, como el módulo.

## Resultados (x % y)

| Modelo | Parámetros | Train MSE | Far OOD MSE | Ratio (Estabilidad) |
| :--- | :--- | :--- | :--- | :--- |
| **MLP-Huge (V194)** | 1,052,673 | **0.0055** | **4.11** | **743** |
| **Poly-V193 (V194)** | 28,385 | 0.0678 | 15.53 | 229 |
| **Poly-V195 (Discont)**| **2,513** | **0.0095** | 19.15 | 2,020 |

## Conclusiones Técnicas

### 1. Eficiencia Paramétrica Extrema
La **Poly-V195** ha logrado una precisión de entrenamiento similar a un MLP de 1 millón de parámetros usando solo **2,500 parámetros** (un ahorro de 400x). Esto demuestra que la rama de discontinuidad es la herramienta correcta para funciones algorítmicas.

### 2. El Poder del STE
El uso de `STEFloor` permitió que Adam optimizara las frecuencias y fases de los "dientes de sierra" a pesar de que la función real tiene gradiente cero. Sin esta técnica, la red nunca habría podido sintonizar sus bases discontinuas.

### 3. El Desafío de la Extrapolación en Saltos
Aunque la precisión local es excelente, el ratio de estabilidad aumentó. Esto sugiere que cuando hay discontinuidades, un pequeño error en la frecuencia aprendida se traduce en un error masivo en OOD (desincronización de los saltos). Para corregir esto, se necesitaría un mecanismo de **Sincronización de Fase** o una arquitectura más rígida para leyes puramente algorítmicas.

## Próximos Pasos (V196)
-   **Gating de Discontinuidad**: Solo activar la rama sawtooth cuando se detecten gradientes infinitos o errores residuales de alta frecuencia.
-   **Normalización de Fase**: Investigar cómo estabilizar la fase en OOD para evitar la desincronización del módulo.
