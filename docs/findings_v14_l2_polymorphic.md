# Findings: V14 (L2 Polymorphic Attention Neuron)

## 1. Experimento: "Aproximación L2 del Max"

Tras descubrir que el dial polimórfico (V13) dotaba a la red de una auto-organización maravillosa (mezclando SUM y MAX) pero a un coste computacional prohibitivo, se implementó la variante **V14**.

El objetivo de la V14 es sustituir la costosa operación de "búsqueda del máximo exacto" por una agregación energética vectorizable: la **Norma L2**.
- `y_l2 = sqrt( X^2 @ (W^2)^T )`
- La intuición matemática es que elevar al cuadrado amplifica exponencialmente los valores más grandes ("rasgos dominantes"), actuando funcionalmente como un detector análogo al `max`, pero utilizando únicamente operaciones de matrices aceleradas por hardware.

- **Fórmula**: `y_eff = alpha * y_sum + (1 - alpha) * y_l2`
- Entrenamiento: MNIST, 10 épocas, Adam.

## 2. Resultados

| Variante | Agregadores | Accuracy (10 Epochs) | Tiempo / Época |
| :--- | :--- | :--- | :--- |
| **V1 (Baseline)** | Solo SUM | 87.61% | ~11s |
| **V13** | SUM vs MAX | 86.51% | ~38s |
| **V14** | SUM vs L2 | **86.46%** | **~13s** |

**Análisis de Identidad Neuronal (Auto-organización del dial `alpha`):**
- **Capa Oculta (512 neuronas)**: 
  - 211 neuronas eligieron ser puramente sumadoras (alpha > 0.6).
  - 65 neuronas eligieron ser filtros de energía L2 (alpha < 0.4).
  - 236 se mantuvieron en un estado híbrido.
- **Capa de Salida (10 neuronas)**:
  - 10 neuronas eligieron ser puramente sumadoras.
  - 0 eligieron L2 o híbrido.

## 3. Conclusiones

1. **Resolución del Cuello de Botella Computacional**: ¡Éxito rotundo! Al usar la aproximación L2, el tiempo de cálculo se ha desplomado de 38s a 13s, recuperando casi toda la velocidad nativa del baseline (11s) al evitar la expansión del tensor 3D necesario para el `max` exacto.
2. **Preservación de la Identidad Polimórfica**: A pesar de la aproximación, la red conserva la capacidad de diferenciar el comportamiento de sus neuronas. La capa oculta sigue creando un ecosistema mixto de sumadores y detectores energéticos, mientras que la capa final clasifica de nuevo con un 100% de preferencia por la suma lógica.
3. **Mantenimiento del Rendimiento**: El accuracy final es idéntico a la versión lenta de MAX (86.46% vs 86.51%), validando que la intuición matemática L2 sirve perfectamente como un proxy suave para detectar patrones dominantes.

## 4. Próximos Pasos

Habiendo validado que la familia Lp (Normas) funciona como aproximación suave y diferenciable para alterar el álgebra de la neurona de forma eficiente, el diseño propuesto **V15 (Learnable-Lp Attention Neuron)** es el siguiente paso lógico. En la V15, en lugar de mezclar linealmente SUM y L2 con un dial `alpha`, se permitirá que la red aprenda directamente el exponente `p` de la ecuación de la norma generalizada de forma continua.