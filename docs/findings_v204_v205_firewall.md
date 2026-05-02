# Hallazgos V204 & V205: El Firewall Biológico y el Phase Jitter

## Objetivo
Probar la hipótesis biológica del "Firewall": ¿Son las Neuronas de Resonancia inherentemente más robustas al ruido (fases desincronizadas) que las neuronas clásicas (sumas lineales)?

## El Error Conceptual (V204)
En la primera prueba, la red resonante colapsó con ruido moderado (std=0.5), mientras la MLP aguantó. El análisis matemático reveló el problema del **"Phase Jitter"**:
Al mapear la intensidad directamente a fase mediante `x_phase = x * math.pi`, un ruido de `std=1.0` se convierte en un desfase de $\pi$ (180º). En una onda de coseno, 180º invierte la señal completamente. El ruido se transforma de aditivo a un "Jitter" destructivo hiper-sensible que anula el gradiente y vuelve a la red inentrenable con ruido.

## La Solución (V205)
Para lograr un verdadero "Firewall", la red necesita tener "bandas de resonancia" más holgadas, no puntos microscópicos de fallo.
1. **Reducción de Escala Angular:** En vez de mapear la intensidad de 0 a $\pi$, la mapeamos de 0 a $\pi/4$. Así, un ruido extremo no llega a "dar la vuelta" a la onda.
2. **Vacunación por Ruido:** Se entrenó la red directamente con ruido (std=0.5) para que aprendiera a usar frecuencias más estables.

## Resultados del Experimento

| Noise (std) | Resonant PI (Colapso) | Resonant PI/4 (Robusta) |
|-------------|-----------------------|-------------------------|
| 0.0         | 11.86%                | **87.72%**              |
| 0.5         | 68.75%                | **87.59%**              |
| 1.0         | 10.14%                | **74.88%**              |

## Conclusión
El concepto de **Phase Jitter** es real y crítico en arquitecturas resonantes. Si la escala de fase es demasiado amplia, el ruido produce interferencia destructiva incontrolable. Sin embargo, al escalar correctamente la sensibilidad ($\pi/4$) y entrenar bajo estrés, la Neurona de Fase demuestra que puede actuar como un Firewall, aguantando tormentas de ruido con desviaciones estándar de 1.0 sin perder su sintonización básica. ¡Otra propiedad biológica replicada con éxito!
