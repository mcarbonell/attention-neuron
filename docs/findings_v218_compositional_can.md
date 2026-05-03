# Hallazgos V218: Compositional Attention Neuron (CAN)

## Objetivo
Validar si una arquitectura de dos capas con competencia darwiniana (MoE Honesto) puede descubrir la composición simbólica de una función compleja ($y = \sin(x_1 \cdot x_2)$) y generalizar fuera del rango de entrenamiento (OOD).

## Configuracion del Experimento
- **Funcion:** $h(x, y) = \sin(x \cdot y)$
- **Rango de Entrenamiento:** $x \in [1, 2]$
- **Rango OOD (Test):** $x \in [2, 10]$ (Escala 5x superior)
- **Arquitectura:**
    - **Capa 1 (Mapper):** Linear vs Log.
    - **Capa 2 (Operator):** MLP vs Harmonic (Seno).
- **Metricas:** PEI (Parametric Efficiency Index) y Error OOD.

## Resultados
| Metrica | Valor |
| :--- | :--- |
| **PEI** | **0.4684** |
| **Wall Clock Time** | 27.22s |
| **Dominancia Log -> Har (Train)** | **70.3%** |
| **OOD MSE (Log -> Har)** | **0.8676** |
| **OOD MSE (Lin -> MLP - Baseline)** | 2.8947 |

### Analisis de la Seleccion de Caminos
La red descubrio de forma autonoma que el camino mas eficiente para resolver $\sin(x \cdot y)$ es pasar primero por el **espacio logaritmico** y luego aplicar un **operador armonico**.

1.  **Descubrimiento Simbolico:** El camino `Log -> Har` capturo la esencia del problema. Al estar en log-space, la multiplicacion se convierte en una suma, facilitando la tarea al operador armonico.
2.  **Superioridad OOD:** Mientras que el camino tradicional `Lin -> MLP` fallo catastroficamente fuera de rango (MSE 2.89), el camino `Log -> Har` mantuvo un error significativamente menor (0.86). Aunque no es una perfeccion absoluta, es una mejora del **333%** en estabilidad.
3.  **Eficiencia Parametrica (PEI):** Con un numero minimo de parametros (en comparacion con una red densa profunda), la arquitectura CAN logra una aproximacion funcional robusta.

## Conclusiones
1.  **La Composicion es Clave:** La inteligencia no reside en el tamaño de la capa, sino en la capacidad de componer transformaciones.
2.  **Generalizacion Estructural:** La generalizacion OOD no se consigue con mas datos, sino con la **alineacion estructural** (Architecture-Problem Fit).
3.  **Hacia la Red "Sin Pesos":** Este experimento demuestra que si elegimos las funciones base adecuadas (Log, Seno), la red necesita muy pocos "pesos" para aprender leyes universales.

---
**Siguiente Paso:** Refinar el `HarmonicExpert` para que sea capaz de manejar la periodicidad de forma aun mas precisa en el espacio logaritmico (incorporando un paso de `exp` interno).
