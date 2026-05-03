# Hallazgos V219: Conscious Attention Neuron (Auto-Confianza)

## Objetivo
Explorar la capacidad de una red para predecir su propio error (Self-Confidence) mediante una arquitectura de doble cabeza (Valor y Confianza) y validar su comportamiento en escenarios Out-of-Distribution (OOD).

## Configuracion del Experimento
- **Arquitectura:** CAN de 2 capas con cabeza de valor (MSE) y cabeza de confianza (prediccion del log-MSE).
- **Entrenamiento:** Rango $[1, 2]$.
- **Test OOD:** Rango $[2, 5]$.
- **Mecanismo:** Desacoplamiento de gradientes (`detach`) para que la cabeza de confianza no afecte al aprendizaje de la tarea principal.

## Resultados
| Metrica | Train | OOD (Test) |
| :--- | :--- | :--- |
| **MSE Real** | 6.1298e-04 | 1.3060e+00 |
| **Confianza (MSE Previsto)** | 1.7569e-04 | **1.5932e-05** |
| **Correlacion Real vs Previsto** | 0.2508 | N/D |
| **PEI** | **0.2878** | - |

## Analisis del "Efecto Arrogancia"
El hallazgo mas importante de este experimento es el colapso de la confianza en OOD:
1.  **Exceso de Confianza (Dunning-Kruger Neural):** En lugar de predecir un error mayor cuando los datos salen de rango, la red predice un error **menor**. Se vuelve mas "arrogante" conforme mas se equivoca.
2.  **Fallo de Extrapolacion:** La cabeza de confianza sufre el mismo problema que la de valor: no sabe que el mundo ha cambiado. Al ser un MLP entrenado en $[1, 2]$, simplemente proyecta lo que sabe hacia el infinito, ignorando que su propia arquitectura de valor esta fallando.
3.  **Correlacion Positiva en Train:** El hecho de que en entrenamiento la correlacion sea de 0.25 indica que la red *si* es capaz de identificar que ejemplos son mas dificiles dentro de su distribucion conocida.

## Conclusiones
1.  **La Confianza Predictiva es Insuficiente:** Predecir el loss mediante una capa densa no sirve para detectar OOD. La confianza debe ser una propiedad **emergente o estructural**, no una salida entrenada.
2.  **Necesidad de Memoria/Novedad:** Para que la red "sepa que no sabe", necesita un mecanismo para comparar el input actual con lo que ha visto antes.

---
**Siguiente Paso:** Implementar un sistema de **Memoria de Prototipos** (V220) para detectar novedad mediante la distancia estructural al conjunto de entrenamiento.
