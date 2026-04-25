# Findings: V13 (Polymorphic Attention Neuron)

## 1. Experimento: "El Dial Neuronal"

Se ha puesto a prueba una arquitectura propuesta iterativamente para explorar si las neuronas pueden "elegir" su función de agregación lógica. En lugar de forzar a todas las neuronas a integrar sus entradas mediante una suma ponderada, se dotó a cada neurona de un parámetro entrenable (`alpha`) que actúa como un dial entre dos funciones:

- `y_sum = sum(w * x)` (Comportamiento de agregador de evidencia / "Soft OR")
- `y_max = max(w * x)` (Comportamiento de detector estricto de rasgos)
- **Fórmula**: `y_eff = alpha * y_sum + (1 - alpha) * y_max`

Entrenamiento: MNIST, 10 épocas, Adam, y arquitectura base Residual (V1).

## 2. Resultados

| Variante | Accuracy (10 Epochs) | Tiempo / Época |
| :--- | :--- | :--- |
| **V1 (Residual, Solo SUM)** | 87.61% | ~11s |
| **V13 (Polymorphic SUM/MAX)** | 86.51% | ~38s |

**Análisis de Identidad Neuronal (Auto-organización del dial `alpha`):**
- **Capa Oculta (512 neuronas)**: 
  - 225 neuronas eligieron ser sumadoras (alpha > 0.6).
  - 122 neuronas eligieron ser filtros MAX (alpha < 0.4).
  - 165 se mantuvieron híbridas.
- **Capa de Salida (10 neuronas)**:
  - 10 neuronas eligieron ser sumadoras.
  - 0 eligieron MAX o híbrido.

## 3. Conclusiones

1. **Auto-Organización Orgánica**: El hallazgo más importante es cómo la red ha asignado roles. La capa oculta generó un ecosistema diverso de detectores estrictos (MAX) y acumuladores (SUM) para extraer características. Sin embargo, la capa final de clasificación "descubrió" de forma totalmente autónoma que necesitaba sumar toda la evidencia para tomar una decisión final (clasificación), moviendo sus 10 diales hacia SUM.
2. **Viabilidad Matemática**: El gradiente fluye perfectamente a través del dial polimórfico, permitiendo arquitecturas que adaptan su "álgebra" a la tarea.
3. **El Cuello de Botella Computacional**: El tiempo de ejecución se triplicó (de 11s a 38s) debido a que calcular el `max` verdadero requiere materializar las conexiones intermedias (tensor 3D) en lugar de usar multiplicaciones de matrices puras. 

## 4. Próximos Pasos (Optimización)

Para futuras iteraciones, la idea de "suma de cuadrados" (Norma $L_p$) surge como la solución perfecta al problema computacional. Se podría aproximar el comportamiento MAX utilizando `sqrt((X^2) * (W^2)^T)`, lo cual simula la amplificación de los valores máximos manteniendo la velocidad nativa de la multiplicación de matrices en hardware.