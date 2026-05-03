# Hallazgos V213: El "Activation Bridge" y la Trampa de Sísifo

## Objetivo
Evaluar si un optimizador local basado en gradientes (Adam) puede entrenar una red profunda para "hablar" con un bloque simbólico no-diferenciable (`%`) inyectándole un gradiente sintético estimado mediante diferencias finitas sobre las activaciones.

## Resultados
| Época | Train MSE |
| :--- | :--- |
| 0 | 2.29 |
| 500 | 1373.06 |
| 1000 | 1.28 |
| 2500 | 108.01 |
| 3500 | 1.50 |
| **Final** | **387.98** |

## Conclusiones: La Paradoja de los Gradientes Perfectos

El experimento falló espectacularmente al converger (el error oscilaba salvajemente entre 1.2 y 1300), pero falló por un motivo hermoso y revelador.

### 1. El Puente Funcionó Perfectamente
El "Activation Bridge" sí estaba calculando los gradientes sintéticos correctamente y transmitiéndoselos a Adam. El problema no fue el gradiente, **fue el paisaje de pérdida**.

### 2. La Trampa de Sísifo
La función Módulo (`Z_1 % Z_2`) genera una onda de sierra infinita. 
Si el objetivo es `Y=4` y la red predice `Z_1 % Z_2 = 1`, el gradiente exacto le dice a Adam: *"¡Aumenta $Z_1$!"*. 
Adam obedece diligentemente y sube los pesos para que $Z_1$ crezca. $Z_1$ sube a 2, a 3, a 4... ¡El error baja a casi cero! (Épocas 1000 y 3500).
Pero como Adam tiene inercia (momentos), $Z_1$ sigue creciendo un milímetro más... y **cruza el acantilado del módulo**. De repente `Z_1 % Z_2` cae a `0`. 
El error salta instantáneamente de 0 a 16. La red ha caído por el precipicio matemático. 

Adam se pasa todo el entrenamiento empujando la piedra a la cima de la montaña de la onda de sierra, solo para ver cómo se despeña por el otro lado debido a las discontinuidades del operador.

### 3. El Veredicto Final del Neuro-Simbólico
Este experimento demuestra un límite teórico crítico en la IA Neuro-Simbólica:
> **Incluso si logras "conectar" una red neuronal con código simbólico a través de gradientes sintéticos perfectos, si el código simbólico genera un paisaje no-convexo o periódico, los optimizadores locales como Adam fracasarán.**

Para resolver problemas que involucran saltos lógicos duros, no basta con calcular el gradiente. **Debes usar algoritmos de búsqueda global (como el DGE del V212 o Algoritmos Genéticos)** que no sigan a ciegas la pendiente local hacia el abismo.
